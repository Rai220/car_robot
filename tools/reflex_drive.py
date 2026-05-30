#!/usr/bin/env python3
"""
Near-real-time reflex driver for the Keyestudio ESP32-CAM car.

Two concurrent loops:
  * REFLEX (local, ~6 Hz): a background thread reads the MJPEG stream (:81),
    a light numpy heuristic estimates free space ahead in three zones
    (left / center / right), and the car is driven accordingly. No cloud in
    this path -> reactions are instant.
  * GOAL (Gemini, every --goal-sec): suggests a high-level explore direction,
    used only as a TIE-BREAKER when left/right look equally open.

Free-space heuristic: compares each zone of the near-floor band to a reference
strip right in front of the wheels (almost always floor). Zones that look like
that floor (similar brightness, little extra clutter) are "open". This adapts to
whatever the floor looks like, incl. patterned floors.

Control uses send-on-change: 'forward' persists (continuous drive), a 'left'/'right'
tick spins until the next tick re-decides, 'stop' halts. Always stops on exit.

Env: CAR_IP, GEMINI_API_KEY (or GOOGLE_API_KEY)

Examples:
  set -a; source .env; set +a
  .venv/bin/python tools/reflex_drive.py --dry-run --debug     # scores only, NO movement (tune here)
  .venv/bin/python tools/reflex_drive.py                       # drive: reflex + Gemini goals
  .venv/bin/python tools/reflex_drive.py --no-gemini --secs 30 # pure local reflex, 30s
"""
import os, sys, time, json, base64, argparse, threading, urllib.request, urllib.error
from io import BytesIO
import numpy as np
from PIL import Image

CAR_IP = os.environ.get("CAR_IP")
KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpg = None
        self.gray = None
        self.fps = 0.0
        self.bias = None          # 'left' | 'right' | 'forward' | None  (from Gemini)
        self.bias_reason = ""
        self.running = True


S = State()


# ---------------- car HTTP control ----------------
def ctl(q, timeout=4):
    try:
        urllib.request.urlopen(f"http://{CAR_IP}/control?{q}", timeout=timeout).read()
    except Exception as e:
        print(f"   [ctl err] {q}: {e}")


def stop():
    ctl("var=car&val=5")


# ---------------- MJPEG stream reader (background) ----------------
def stream_reader(url):
    while S.running:
        try:
            resp = urllib.request.urlopen(url, timeout=10)
            buf = b""
            t0 = time.time(); n = 0
            while S.running:
                chunk = resp.read(8192)
                if not chunk:
                    break
                buf += chunk
                while True:
                    soi = buf.find(b"\xff\xd8")
                    eoi = buf.find(b"\xff\xd9", soi + 2) if soi != -1 else -1
                    if soi == -1 or eoi == -1:
                        break
                    jpg = buf[soi:eoi + 2]
                    buf = buf[eoi + 2:]
                    try:
                        gray = np.asarray(Image.open(BytesIO(jpg)).convert("L"), dtype=np.float32)
                    except Exception:
                        continue
                    n += 1
                    dt = time.time() - t0
                    with S.lock:
                        S.jpg, S.gray = jpg, gray
                        if dt >= 1.0:
                            S.fps = n / dt; n = 0; t0 = time.time()
        except Exception as e:
            if S.running:
                print(f"[stream] reconnect: {e}")
                time.sleep(0.5)


# ---------------- free-space heuristic ----------------
def free_space(gray):
    h, w = gray.shape
    mid = gray[int(h * 0.45):int(h * 0.92), :]           # near-floor band ahead
    ref = gray[int(h * 0.85):, int(w * 0.35):int(w * 0.65)]  # floor strip in front of wheels
    ref_mean = float(ref.mean())
    ref_tex = float(np.abs(np.diff(ref, axis=1)).mean()) + 1e-3
    H, W = mid.shape
    scores = []
    for a, b in ((0, W // 3), (W // 3, 2 * W // 3), (2 * W // 3, W)):
        reg = mid[:, a:b]
        bright_diff = abs(float(reg.mean()) - ref_mean) / (ref_mean + 1e-3)
        clutter = (np.abs(np.diff(reg, axis=1)).mean() + np.abs(np.diff(reg, axis=0)).mean())
        openness = 1.0 / (1.0 + 3.0 * bright_diff + clutter / (ref_tex * 6.0))
        scores.append(round(openness, 3))
    return {"left": scores[0], "center": scores[1], "right": scores[2]}


def decide(sc, bias, fwd_min):
    l, c, r = sc["left"], sc["center"], sc["right"]
    mx = max(l, c, r)
    if mx < fwd_min * 0.6:
        return "stop"
    if c >= fwd_min and c >= mx * 0.9:
        return "forward"
    if abs(l - r) < 0.04 and bias in ("left", "right"):
        return bias
    return "left" if l > r else "right"


# ---------------- Gemini goal thread ----------------
GOAL_SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {"type": "string", "enum": ["forward", "left", "right"]},
        "reason": {"type": "string"},
    },
    "required": ["direction", "reason"],
}
GOAL_PROMPT = ("A small ground robot is exploring indoors (forward camera). "
               "Which direction is most interesting/open to explore next? "
               "Reply with direction (forward/left/right) and a short reason.")


def ask_gemini(model, jpg, timeout=20):
    img = base64.b64encode(jpg).decode()
    body = {"contents": [{"parts": [{"text": GOAL_PROMPT},
                                    {"inline_data": {"mime_type": "image/jpeg", "data": img}}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": GOAL_SCHEMA,
                                 "temperature": 0.5, "thinkingConfig": {"thinkingBudget": 0}}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={KEY}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    return json.loads(d["candidates"][0]["content"]["parts"][0]["text"])


def goal_thread(model, period):
    while S.running:
        with S.lock:
            jpg = S.jpg
        if jpg:
            try:
                d = ask_gemini(model, jpg)
                with S.lock:
                    S.bias = d.get("direction"); S.bias_reason = d.get("reason", "")
            except Exception as e:
                with S.lock:
                    S.bias_reason = f"(gemini err: {e})"
        for _ in range(int(period * 10)):
            if not S.running:
                return
            time.sleep(0.1)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Near-real-time reflex driver (local CV + Gemini goals)")
    ap.add_argument("--hz", type=float, default=6.0, help="reflex decisions per second")
    ap.add_argument("--speed", type=int, default=3, help="drive speed 0-8")
    ap.add_argument("--fwd-min", type=float, default=0.5, help="min center openness to go forward (tune with --debug)")
    ap.add_argument("--smooth", type=float, default=0.6, help="temporal smoothing 0..0.9 (higher = steadier, less jittery)")
    ap.add_argument("--perception", choices=["heuristic", "depth"], default="depth",
                    help="free-space source: 'depth' = local Depth Anything V2 ML model (default), 'heuristic' = brightness")
    ap.add_argument("--turn-pulse", type=float, default=0.08, help="seconds of in-place spin per turn tick (small = gentle, stepwise turns)")
    ap.add_argument("--turnspeed", type=int, default=60, help="set the car's in-place turn power 0-255 at startup (60 = gentle; needs firmware with var=turnspeed)")
    ap.add_argument("--escape-after", type=int, default=4, help="consecutive 'stop' ticks before an un-stuck maneuver")
    ap.add_argument("--back-secs", type=float, default=0.3, help="reverse duration in the un-stuck maneuver")
    ap.add_argument("--scan-secs", type=float, default=0.45, help="scan-turn duration in the un-stuck maneuver")
    ap.add_argument("--goal-sec", type=float, default=4.0, help="seconds between Gemini goal updates")
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--secs", type=float, default=30.0, help="max run seconds")
    ap.add_argument("--no-gemini", action="store_true", help="pure local reflex, no cloud goals")
    ap.add_argument("--dry-run", action="store_true", help="decide but do NOT move the wheels")
    ap.add_argument("--debug", action="store_true", help="print per-tick scores")
    args = ap.parse_args()

    if not CAR_IP:
        sys.exit("set CAR_IP first: set -a; source .env; set +a")
    if not args.no_gemini and not KEY:
        sys.exit("set GEMINI_API_KEY (or use --no-gemini)")

    DP = None
    if args.perception == "depth":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import depth_perception as DP
        mpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "da2-small")
        print(f"loading depth model from {mpath} ...")
        dev = DP.load(model_id=mpath)
        print(f"depth model ready on {dev}")

    threading.Thread(target=stream_reader, args=(f"http://{CAR_IP}:81/stream",), daemon=True).start()
    if not args.no_gemini:
        threading.Thread(target=goal_thread, args=(args.model, args.goal_sec), daemon=True).start()

    print(f"reflex_drive | {'DRY-RUN' if args.dry_run else 'DRIVING'} | hz={args.hz} speed={args.speed} "
          f"fwd_min={args.fwd_min} gemini={'off' if args.no_gemini else args.model}")
    print("waiting for first frame... (Ctrl-C to stop)")

    if args.turnspeed is not None:
        ctl(f"var=turnspeed&val={args.turnspeed}")
        print(f"set turnspeed={args.turnspeed}")

    mode = {"v": None}  # 'fwd' | 'stopped'

    def drive(action):
        # forward = continuous; turns = short stepwise spin pulse then stop (gentle); stop = idempotent
        if action == "forward":
            if mode["v"] != "fwd":
                ctl(f"var=speed&val={args.speed}"); ctl("var=car&val=1"); mode["v"] = "fwd"
        elif action in ("left", "right"):
            ctl("var=car&val=" + ("3" if action == "left" else "4"))
            time.sleep(args.turn_pulse)
            ctl("var=car&val=5")
            mode["v"] = "stopped"
        else:
            if mode["v"] != "stopped":
                ctl("var=car&val=5"); mode["v"] = "stopped"

    t_end = time.time() + args.secs
    sm = None  # EMA-smoothed scores
    stuck = 0
    try:
        while S.running and time.time() < t_end:
            with S.lock:
                gray = S.gray; jpg = S.jpg; fps = S.fps; bias = S.bias; reason = S.bias_reason
            if gray is None:
                time.sleep(0.05); continue
            if args.perception == "depth":
                sc = DP.free_space(DP.depth(Image.open(BytesIO(jpg)).convert("RGB")))
            else:
                sc = free_space(gray)
            a = max(0.0, min(0.9, args.smooth))
            sm = sc if sm is None else {k: a * sm[k] + (1 - a) * sc[k] for k in sc}
            act = decide(sm, bias, args.fwd_min)
            stuck = stuck + 1 if act == "stop" else 0
            if args.debug:
                print(f"  L{sm['left']:.2f} C{sm['center']:.2f} R{sm['right']:.2f} | "
                      f"{act:7s} | fps={fps:.0f} goal={bias} {reason[:40]}")
            if not args.dry_run:
                if stuck >= args.escape_after:
                    t = "3" if sm["left"] >= sm["right"] else "4"   # scan toward the more open side
                    print(f"  [escape] stuck x{stuck} -> back up + scan {'left' if t == '3' else 'right'}")
                    ctl(f"var=speed&val={max(3, args.speed)}"); ctl("var=car&val=2")
                    time.sleep(args.back_secs); ctl("var=car&val=5")
                    ctl(f"var=car&val={t}"); time.sleep(args.scan_secs); ctl("var=car&val=5")
                    mode["v"] = "stopped"; stuck = 0
                else:
                    drive(act)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        S.running = False
        stop()
        print("STOP sent. done.")


if __name__ == "__main__":
    main()
