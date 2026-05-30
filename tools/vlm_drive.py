#!/usr/bin/env python3
"""
Autonomous VLM exploration loop for the Keyestudio ESP32-CAM car.

Each step:  capture a frame  ->  ask Gemini for ONE action (strict JSON)
            ->  execute via the car's /control HTTP API  ->  short pause  ->  repeat.

The car moves in short pulses (move, then immediately stop), so a crash/Ctrl-C
leaves it stationary. The loop always sends STOP on exit (finally).

Env (from .env):  CAR_IP, GEMINI_API_KEY (or GOOGLE_API_KEY)

Examples:
    set -a; source .env; set +a
    .venv/bin/python tools/vlm_drive.py --dry-run        # narrate only, wheels do NOT move
    .venv/bin/python tools/vlm_drive.py                  # autonomous explore (default 30 steps)
    .venv/bin/python tools/vlm_drive.py --steps 50 --speed 3
"""
import os, sys, json, time, base64, argparse, urllib.request, urllib.error

CAR_IP = os.environ.get("CAR_IP")
KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

PROMPT = (
    "You are the autonomous control system of a small ground robot car with a "
    "forward-facing camera, exploring an indoor space. Look at the camera image "
    "and pick the single best next action to keep exploring while AVOIDING "
    "collisions with walls, furniture, table edges and other obstacles.\n"
    "Actions:\n"
    " - 'forward': only if the path straight ahead is clearly open.\n"
    " - 'left' / 'right': turn to avoid an obstacle ahead or to explore a new direction.\n"
    " - 'stop': if the way is blocked/unsafe or you might fall off an edge.\n"
    "Prefer turning over stopping when something is ahead. Be cautious near edges."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["forward", "left", "right", "stop"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}


def ctl(q, timeout=5):
    try:
        urllib.request.urlopen(f"http://{CAR_IP}/control?{q}", timeout=timeout).read()
    except Exception as e:
        print(f"   [ctl error] {q}: {e}")


def capture(timeout=10, retries=2):
    last = None
    for _ in range(retries + 1):
        try:
            return urllib.request.urlopen(f"http://{CAR_IP}/capture", timeout=timeout).read()
        except Exception as e:
            last = e
            time.sleep(0.5)
    raise last


def stop():
    ctl("var=car&val=5")


def ask_vlm(model, jpg, timeout=30, retries=2, think=0):
    img = base64.b64encode(jpg).decode()
    gen = {"responseMimeType": "application/json", "responseSchema": SCHEMA, "temperature": 0.4}
    # thinkingBudget=0 disables extended "thinking" -> ~2x faster, no latency spikes.
    # gemini-2.0* models don't accept thinkingConfig, so skip it there.
    if think is not None and not model.startswith("gemini-2.0"):
        gen["thinkingConfig"] = {"thinkingBudget": think}
    body = {
        "contents": [{"parts": [{"text": PROMPT},
                                 {"inline_data": {"mime_type": "image/jpeg", "data": img}}]}],
        "generationConfig": gen,
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={KEY}"
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=timeout))
            txt = d["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(txt)
        except urllib.error.HTTPError as e:
            # 429/5xx are transient -> retry; other 4xx are real -> raise immediately
            if e.code not in (429, 500, 502, 503, 504):
                raise
            last = e
            time.sleep(1.5)
        except Exception as e:  # timeouts, SSL handshake, connection resets
            last = e
            time.sleep(1.5)
    raise last


def do_action(action, args):
    if action == "forward":
        ctl(f"var=speed&val={args.speed}"); ctl("var=car&val=1"); time.sleep(args.pulse); stop()
    elif action == "left":
        ctl(f"var=speed&val={args.speed}"); ctl("var=car&val=3"); time.sleep(args.turn); stop()
    elif action == "right":
        ctl(f"var=speed&val={args.speed}"); ctl("var=car&val=4"); time.sleep(args.turn); stop()
    else:  # stop / unknown
        stop()


def main():
    ap = argparse.ArgumentParser(description="Autonomous VLM explore loop for the ESP32-CAM car")
    ap.add_argument("--steps", type=int, default=30, help="max decision steps")
    ap.add_argument("--speed", type=int, default=3, help="drive speed 0-8 (low recommended)")
    ap.add_argument("--pulse", type=float, default=0.4, help="forward pulse seconds")
    ap.add_argument("--turn", type=float, default=0.12, help="turn pulse seconds (small = gentle turns; turning spins in place at fixed power, so angle is set by duration)")
    ap.add_argument("--interval", type=float, default=0.4, help="pause between decisions")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--think", type=int, default=0,
                    help="Gemini thinking budget: 0 = off (fastest, ~1.5s), -1 = dynamic (smarter, slower)")
    ap.add_argument("--dry-run", action="store_true", help="narrate only; do NOT move the wheels")
    args = ap.parse_args()

    if not CAR_IP or not KEY:
        sys.exit("ERROR: set CAR_IP and GEMINI_API_KEY first:  set -a; source .env; set +a")

    mode = "DRY-RUN (no movement)" if args.dry_run else "DRIVING"
    print(f"VLM explore | model={args.model} | {mode} | steps={args.steps} "
          f"speed={args.speed} pulse={args.pulse}s turn={args.turn}s")
    print(f"car={CAR_IP}  —  Ctrl-C to stop\n")

    consec_fail = 0
    try:
        for i in range(1, args.steps + 1):
            try:
                jpg = capture()
            except Exception as e:
                consec_fail += 1
                print(f"[{i}] capture failed ({consec_fail}/3): {e}")
                stop()
                if consec_fail >= 3:
                    print("too many consecutive failures -> abort"); break
                time.sleep(1.0); continue
            t = time.time()
            try:
                dec = ask_vlm(args.model, jpg, think=args.think)
            except urllib.error.HTTPError as e:
                print(f"[{i}] VLM HTTP {e.code}: {e.read().decode()[:200]}  -> abort"); stop(); break
            except Exception as e:
                consec_fail += 1
                print(f"[{i}] VLM error after retries ({consec_fail}/3): {e}")
                stop()
                if consec_fail >= 3:
                    print("too many consecutive failures -> abort"); break
                time.sleep(1.0); continue
            consec_fail = 0
            action = dec.get("action", "stop")
            reason = dec.get("reason", "")
            print(f"[{i}/{args.steps}] {action.upper():7s} ({time.time()-t:.1f}s)  {reason}")
            if not args.dry_run:
                do_action(action, args)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[interrupted by user]")
    finally:
        stop()
        print("STOP sent. done.")


if __name__ == "__main__":
    main()
