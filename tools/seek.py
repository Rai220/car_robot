#!/usr/bin/env python3
"""
LLM target-seeking for the ESP32-CAM car: find a described object and drive to it.

  set -a; source .env; set +a
  .venv/bin/python tools/seek.py "toy road roller / steamroller"

CONTROL PHILOSOPHY (learned the hard way):
  * Gemini DRIVES the navigation. It reliably understands the scene — where the
    target is (bearing), how it moves (from a few frames), whether it's safe to
    go forward (rug / open / away from furniture). All steering & search comes
    from Gemini.
  * The local monocular depth model is NOT a reliable navigator: it only sees an
    obstacle once it's almost touching it (~25 cm), and reads "open" at a
    distance. So depth is used ONLY as a last-resort EMERGENCY BUMPER: if
    something is right in front (center openness very low), override and back/turn
    — even if Gemini wanted forward. At normal distances depth is ignored.
  * The control loop turns Gemini's guidance into smooth motion (forward + trim
    arcs) and runs every tick; Gemini guidance refreshes in the background.

Env: CAR_IP, GEMINI_API_KEY (or GOOGLE_API_KEY)
"""
import os, sys, time, json, base64, argparse, threading, urllib.request, urllib.error
from io import BytesIO
from collections import deque
from PIL import Image

CAR_IP = os.environ.get("CAR_IP")
KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpg = None
        self.ring = deque(maxlen=60)    # (t, jpg) history for motion frames
        self.dec = None
        self.dec_t = 0.0
        self.hist = deque(maxlen=4)     # recent decision summaries (memory)
        self.running = True


S = State()


def ctl(q, timeout=4):
    try:
        urllib.request.urlopen(f"http://{CAR_IP}/control?{q}", timeout=timeout).read()
    except Exception as e:
        print(f"   [ctl err] {q}: {e}")


def stop():
    ctl("var=car&val=5")


def capture_reader():
    """Poll /capture (port 80). Robust across runs, unlike the single-client :81 stream."""
    cap_url = f"http://{CAR_IP}/capture"
    last_push = 0.0
    while S.running:
        try:
            jpg = urllib.request.urlopen(cap_url, timeout=5).read()
            if jpg[:2] == b"\xff\xd8":
                now = time.time()
                with S.lock:
                    S.jpg = jpg
                    if now - last_push >= 0.25:
                        S.ring.append((now, jpg)); last_push = now
        except Exception as e:
            if S.running:
                print(f"[capture] retry: {e}"); time.sleep(0.3)


def pick_frames(n, space):
    with S.lock:
        ring = list(S.ring)
    if not ring:
        return []
    now = ring[-1][0]
    picks = []
    for k in range(n - 1, -1, -1):
        tgt = now - k * space
        best = min(ring, key=lambda r: abs(r[0] - tgt))
        if not picks or best[1] is not picks[-1][1]:
            picks.append(best)
    return picks


SEEK_SCHEMA = {
    "type": "object",
    "properties": {
        "target_visible": {"type": "boolean"},
        "target_bearing": {"type": "string",
                           "enum": ["far_left", "left", "center", "right", "far_right", "none"]},
        "target_size": {"type": "string", "enum": ["none", "small", "medium", "large"]},
        "target_motion": {"type": "string",
                          "enum": ["approaching", "receding", "steady", "lost", "unknown"]},
        "turn_strength": {"type": "string", "enum": ["none", "slight", "medium", "hard"]},
        "search_strategy": {"type": "string",
                            "enum": ["spin_left", "spin_right", "forward", "turn_around", "stop"]},
        "safe_forward": {"type": "boolean"},
        "arrived": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["target_visible", "target_bearing", "target_size", "target_motion",
                 "turn_strength", "search_strategy", "safe_forward", "arrived", "reason"],
}


def build_prompt(target, hist_lines):
    mem = "\n".join(hist_lines) if hist_lines else "none yet"
    return (
        "You are the navigation brain of a small ground robot car (forward camera) "
        "that rolls slowly forward. You get the last few frames in time order "
        "(oldest first, last = current) so you can judge MOTION, plus a memory of "
        "your recent decisions.\n"
        f'TASK: find this target and drive right up to it: "{target}".\n'
        "SAFETY RULES (obey even if it delays reaching the target):\n"
        " - Stay ON THE RUG/CARPET; never head toward bare floor past the rug edge.\n"
        " - Keep to OPEN floor; do NOT go under furniture or into narrow gaps; keep "
        "distance from walls/furniture (only the target itself may be approached closely).\n"
        " - set safe_forward=false whenever furniture/wall/clutter is ahead, even if the "
        "target is also ahead — we will steer around rather than ram.\n"
        f"YOUR RECENT DECISIONS (newest first):\n{mem}\n"
        "Fill ALL fields:\n"
        " - target_visible: visible in the CURRENT frame?\n"
        " - target_bearing: far_left/left/center/right/far_right, or none.\n"
        " - target_size: none/small(far)/medium/large(right in front, fills lower-center).\n"
        " - target_motion: approaching/receding/steady/lost/unknown (compare frames).\n"
        " - turn_strength: how hard to steer toward it: none(centered)/slight/medium/hard.\n"
        " - search_strategy: if not visible, where to look using MEMORY: spin_left, "
        "spin_right, forward, turn_around, or stop.\n"
        " - safe_forward: is open rug clear ahead right now?\n"
        " - arrived: true ONLY when target_size is large and roughly centered.\n"
        " - reason: one short sentence."
    )


def ask_seek(model, target, frames, hist_lines, timeout=25, retries=2):
    now = frames[-1][0] if frames else time.time()
    parts = [{"text": build_prompt(target, hist_lines)}]
    for i, (t, jpg) in enumerate(frames):
        tag = "CURRENT" if i == len(frames) - 1 else f"{now - t:.1f}s ago"
        parts.append({"text": f"Frame {i + 1}/{len(frames)} ({tag}):"})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpg).decode()}})
    body = {"contents": [{"parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": SEEK_SCHEMA,
                                 "temperature": 0.3, "thinkingConfig": {"thinkingBudget": 0}}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={KEY}"
    data = json.dumps(body).encode()
    last = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=timeout))
            return json.loads(d["candidates"][0]["content"]["parts"][0]["text"])
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            last = e; time.sleep(1.0)
        except Exception as e:
            last = e; time.sleep(1.0)
    raise last


def summarize(dec):
    if dec.get("target_visible"):
        return (f"saw target bearing={dec.get('target_bearing')} size={dec.get('target_size')} "
                f"motion={dec.get('target_motion')}")
    return f"target lost -> searched {dec.get('search_strategy')}"


def vlm_thread(model, target, gap, n_frames, frame_space):
    first = True
    while S.running:
        frames = pick_frames(n_frames, frame_space)
        if not frames:
            time.sleep(0.1); continue
        with S.lock:
            hist = [f"- {s}" for s in list(S.hist)]
        try:
            t0 = time.time()
            dec = ask_seek(model, target, frames, hist)
            with S.lock:
                S.dec = dec; S.dec_t = time.time()
                S.hist.appendleft(summarize(dec))
            if first:
                print(f"   [vlm] first decision in {time.time()-t0:.1f}s ({len(frames)} frame(s))")
                first = False
        except Exception as e:
            print(f"   [vlm err] {e}")
        end = time.time() + gap
        while S.running and time.time() < end:
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description="LLM target-seeking: Gemini-led, depth as emergency bumper")
    ap.add_argument("target")
    ap.add_argument("--vlm-sec", type=float, default=0.4, help="extra wait between Gemini calls")
    ap.add_argument("--frames", type=int, default=2, help="recent frames sent to Gemini (motion cue)")
    ap.add_argument("--frame-space", type=float, default=0.6, help="seconds between those frames")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--speed", type=int, default=2, help="forward speed 0-8")
    ap.add_argument("--trim-slight", type=int, default=12)
    ap.add_argument("--trim-medium", type=int, default=24)
    ap.add_argument("--trim-hard", type=int, default=40)
    ap.add_argument("--bump", type=float, default=0.12,
                    help="depth center-openness below this = something within ~25cm -> emergency bumper")
    ap.add_argument("--turn-pulse", type=float, default=0.12, help="spin seconds per steering tick")
    ap.add_argument("--search-pulse", type=float, default=0.16, help="spin seconds per search tick")
    ap.add_argument("--scan-max", type=int, default=3, help="max scan-turns in a row before forced forward relocate")
    ap.add_argument("--hz", type=float, default=5.0, help="control loop rate")
    ap.add_argument("--secs", type=float, default=150.0)
    ap.add_argument("--invert-steer", action="store_true")
    ap.add_argument("--no-depth", action="store_true", help="disable the depth emergency bumper")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not CAR_IP:
        sys.exit("set CAR_IP:  set -a; source .env; set +a")
    if not KEY:
        sys.exit("set GEMINI_API_KEY")

    DP = None
    if not args.no_depth:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import depth_perception as DP
        mpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "da2-small")
        print(f"loading depth model from {mpath} ...")
        print("depth model ready on", DP.load(model_id=mpath))

    threading.Thread(target=capture_reader, daemon=True).start()
    threading.Thread(target=vlm_thread,
                     args=(args.model, args.target, args.vlm_sec, args.frames, args.frame_space),
                     daemon=True).start()

    sgn = -1 if args.invert_steer else 1
    MAG = {"none": 0, "slight": args.trim_slight, "medium": args.trim_medium, "hard": args.trim_hard}
    BEAR = {"far_left": -1, "left": -1, "center": 0, "right": 1, "far_right": 1, "none": 0}
    print(f"SEEK {args.target!r} | model={args.model} | frames={args.frames}@{args.frame_space}s + memory | "
          f"{'DRY-RUN' if args.dry_run else 'DRIVING'} | bumper={'off' if args.no_depth else f'<{args.bump}'}")
    print("Ctrl-C to stop\n")

    def cruise(trim):
        if args.dry_run:
            return
        ctl(f"var=trim&val={trim}"); ctl(f"var=speed&val={args.speed}"); ctl("var=car&val=1")

    def spin(direction, secs):
        if args.dry_run:
            return
        ctl("var=car&val=" + ("3" if direction == "left" else "4"))
        time.sleep(secs); stop()

    cur_trim = None; cruising = False; last_t = -1.0
    bump_dir = None; bump_n = 0; wait_log = 0.0
    spins_in_row = 0
    t_end = time.time() + args.secs
    try:
        while S.running and time.time() < t_end:
            with S.lock:
                jpg = S.jpg; dec = S.dec; dec_t = S.dec_t
            if jpg is None or dec is None:
                if time.time() - wait_log > 2.0:
                    print(f"[waiting] frames={'ok' if jpg is not None else 'none'} "
                          f"decision={'ok' if dec is not None else 'pending'}")
                    wait_log = time.time()
                time.sleep(0.05); continue

            # depth: ONLY a close-range emergency signal (center openness)
            center_open = None
            if DP is not None:
                try:
                    center_open = DP.free_space(DP.depth(Image.open(BytesIO(jpg)).convert("RGB")))["center"]
                except Exception:
                    center_open = None

            visible = bool(dec.get("target_visible"))
            bearing = dec.get("target_bearing", "none")
            size = dec.get("target_size", "none")
            strength = dec.get("turn_strength", "medium")
            strat = dec.get("search_strategy", "spin_right")
            safe_fwd = bool(dec.get("safe_forward", True))
            motion = dec.get("target_motion", "unknown")
            fresh = dec_t != last_t

            # arrival: Gemini says large+centered (depth不 needed; it's unreliable far out)
            if visible and size == "large" and bearing in ("center", "left", "right"):
                stop(); print(f"\n*** REACHED TARGET: {dec.get('reason','')} ***"); break

            # EMERGENCY BUMPER: something within ~25cm dead ahead. Overrides everything except
            # the final approach to a large target. Commit to one turn dir; reverse if stuck.
            emergency = (center_open is not None and center_open < args.bump
                         and not (visible and size == "large"))
            if emergency:
                if bump_dir is None:
                    # turn toward target side if known, else default right
                    if bearing in ("left", "far_left"):
                        bump_dir = "left"
                    elif bearing in ("right", "far_right"):
                        bump_dir = "right"
                    else:
                        bump_dir = "right"
                bump_n += 1
                if bump_n >= 5:
                    if args.debug:
                        print(f"[escape] reverse + hard turn {bump_dir}")
                    if not args.dry_run:
                        ctl(f"var=speed&val={max(3, args.speed)}"); ctl("var=car&val=2")
                        time.sleep(0.6); stop(); spin(bump_dir, 0.5)
                    bump_n = 0; bump_dir = None; cruising = False
                    time.sleep(1.0 / args.hz); continue
                if args.debug and fresh:
                    print(f"[BUMPER->{bump_dir}] center_open={center_open:.2f} < {args.bump}")
                spin(bump_dir, args.turn_pulse); cruising = False
                time.sleep(1.0 / args.hz); continue
            bump_dir = None; bump_n = 0

            # ---- Gemini-led navigation ----
            act = ""
            if visible:
                spins_in_row = 0
                mag = MAG.get(strength, args.trim_medium)
                # far to a side + hard -> spin in place to bring it toward center
                if bearing in ("far_left", "far_right") and strength == "hard":
                    side = "left" if bearing == "far_left" else "right"
                    act = f"spin-{side}"
                    spin(side, args.turn_pulse); cruising = False
                    time.sleep(1.0 / args.hz)
                    if args.debug and fresh:
                        print(f"[vis|{size:6s}|{motion:10s}] bearing={bearing:8s} {act}")
                    last_t = dec_t; continue
                # if Gemini vetoes forward (furniture ahead) but target is centered, nudge aside
                if bearing == "center" and not safe_fwd:
                    act = "veto-steer"
                    spin("right", args.turn_pulse); cruising = False
                    time.sleep(1.0 / args.hz)
                    if args.debug and fresh:
                        print(f"[vis|{size:6s}|{motion:10s}] center but unsafe -> {act}")
                    last_t = dec_t; continue
                # otherwise cruise forward with a steering-trim arc toward the target
                trim = BEAR.get(bearing, 0) * mag * sgn
                act = f"arc trim={trim:+d}"
                if not cruising or trim != cur_trim:
                    cruise(trim); cruising = True; cur_trim = trim
            else:
                # SEARCH = explore, don't spin in place forever. Spinning on one spot rarely
                # brings a small target into view and just racks up turns. So: after a couple of
                # scan-turns, if the way ahead is open, DRIVE FORWARD to change vantage point.
                want_spin = strat in ("spin_left", "spin_right", "turn_around")
                if want_spin and spins_in_row >= args.scan_max and safe_fwd:
                    want_spin = False        # forced relocate: stop pirouetting, go look elsewhere
                if want_spin:
                    side = "left" if strat == "spin_left" else "right"
                    secs = args.search_pulse * (3 if strat == "turn_around" else 1)
                    act = f"search {strat}"
                    spin(side, secs); cruising = False; spins_in_row += 1
                    time.sleep(1.0 / args.hz)
                    if args.debug and fresh:
                        print(f"[---|search] {act} ({spins_in_row}) | safe_fwd={safe_fwd}")
                    last_t = dec_t; continue
                elif safe_fwd:
                    act = "search forward"
                    spins_in_row = 0
                    if not cruising or cur_trim != 0:
                        cruise(0); cruising = True; cur_trim = 0
                else:
                    # can't go forward and shouldn't keep spinning -> one nudge turn then reassess
                    act = "search nudge"
                    spin("right", args.search_pulse); cruising = False; spins_in_row += 1
                    time.sleep(1.0 / args.hz)
                    if args.debug and fresh:
                        print(f"[---|search] {act} | blocked, safe_fwd=False")
                    last_t = dec_t; continue

            if args.debug and fresh:
                tag = f"vis|{size:6s}|{motion:10s}" if visible else "---|search "
                co = f"co={center_open:.2f}" if center_open is not None else "co=--"
                print(f"[{tag}] bearing={bearing:8s} {act:14s} | {co} | {dec.get('reason','')[:34]}")
            last_t = dec_t
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        S.running = False
        stop()
        print("STOP sent. done.")


if __name__ == "__main__":
    main()
