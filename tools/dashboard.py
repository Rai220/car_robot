#!/usr/bin/env python3
"""
Live observability dashboard for the ESP32-CAM car.

Three live panels in your browser (default http://127.0.0.1:8000):
  1. Camera — the raw forward-facing camera frame.
  2. Depth  — Depth Anything V2 heatmap (near = warm/red) + the left/center/right
              free-space openness scores vs the block/slow/open thresholds the
              real controller (reflex_drive.py) uses to decide recover-vs-arc.
  3. VLM    — the current Gemini decision (action/goal + reason + latency +
              a short rolling history), so you can see WHY it would steer.

This tool only OBSERVES by default — it never drives autonomously. Optional
manual drive buttons in the UI (forward/left/right/back/stop + speed) let you
move the car around and watch how depth and the VLM react in real time.

Reuses tools/depth_perception.py for depth and the same Gemini call shape as
tools/vlm_drive.py / tools/seek.py. No new dependencies (FastAPI + uvicorn +
opencv already in .venv).

  set -a; source .env; set +a
  tools/dashboard.py                                  # camera + depth + explore-VLM
  tools/dashboard.py --target "yellow rubber bath duck"  # richer seek-style VLM output
  tools/dashboard.py --no-depth                       # skip the local depth model
  tools/dashboard.py --no-vlm                          # camera + depth only (no API calls)
  tools/dashboard.py --port 8080 --vlm-sec 1.5
"""
from __future__ import annotations

import os
import sys

# Re-exec through the project venv so the local ML stack (torch/transformers,
# opencv, fastapi) is on the path even when launched as `tools/dashboard.py`.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
VENV_PYTHON = os.path.join(REPO_ROOT, ".venv", "bin", "python")
VENV_ROOT = os.path.join(REPO_ROOT, ".venv")
if (
    os.path.exists(VENV_PYTHON)
    and os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON)
    and os.path.abspath(sys.prefix) != os.path.abspath(VENV_ROOT)
    and not os.environ.get("CAR_ROBOT_NO_VENV_REEXEC")
):
    env = os.environ.copy()
    env["CAR_ROBOT_NO_VENV_REEXEC"] = "1"
    os.execve(VENV_PYTHON, [VENV_PYTHON, __file__, *sys.argv[1:]], env)

import argparse
import base64
import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn


# --------------------------------------------------------------------------- #
# env / config
# --------------------------------------------------------------------------- #
def load_dotenv(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def enum(value: object, allowed: set, default: str) -> str:
    s = str(value or "").strip().lower()
    return s if s in allowed else default


# --------------------------------------------------------------------------- #
# Gemini call (same shape as vlm_drive.py / seek.py)
# --------------------------------------------------------------------------- #
EXPLORE_PROMPT = (
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
EXPLORE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["forward", "left", "right", "stop"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}

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


def build_seek_prompt(target: str) -> str:
    return (
        "You are the navigation brain of a small ground robot car (forward camera) "
        "that rolls slowly forward. You get the last few frames in time order "
        "(oldest first, last = current) so you can judge MOTION.\n"
        f'TASK: find this target and describe how to drive to it: "{target}".\n'
        "Fill ALL fields:\n"
        " - target_visible: visible in the CURRENT frame?\n"
        " - target_bearing: far_left/left/center/right/far_right, or none.\n"
        " - target_size: none/small(far)/medium/large(right in front).\n"
        " - target_motion: approaching/receding/steady/lost/unknown (compare frames).\n"
        " - turn_strength: how hard to steer toward it: none/slight/medium/hard.\n"
        " - search_strategy: if not visible, where to look: spin_left/spin_right/forward/turn_around/stop.\n"
        " - safe_forward: is open floor clear ahead right now?\n"
        " - arrived: true ONLY when target_size is large and roughly centered.\n"
        " - reason: one short sentence."
    )


def ask_gemini(model: str, key: str, prompt: str, schema: dict,
               jpgs: list[bytes], timeout: float = 25.0, retries: int = 1) -> dict:
    parts: list[dict] = [{"text": prompt}]
    for i, jpg in enumerate(jpgs):
        if len(jpgs) > 1:
            tag = "CURRENT" if i == len(jpgs) - 1 else f"{len(jpgs) - 1 - i} step(s) ago"
            parts.append({"text": f"Frame {i + 1}/{len(jpgs)} ({tag}):"})
        parts.append({"inline_data": {"mime_type": "image/jpeg",
                                      "data": base64.b64encode(jpg).decode("ascii")}})
    gen: dict = {"responseMimeType": "application/json", "responseSchema": schema,
                 "temperature": 0.3}
    if not model.startswith("gemini-2.0"):
        gen["thinkingConfig"] = {"thinkingBudget": 0}
    body = {"contents": [{"parts": parts}], "generationConfig": gen}
    data = json.dumps(body).encode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    last: Optional[BaseException] = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            doc = json.load(urllib.request.urlopen(req, timeout=timeout))
            txt = doc["candidates"][0]["content"]["parts"][0]["text"]
            try:
                return json.loads(txt)
            except Exception:
                m = re.search(r"\{.*\}", txt, re.DOTALL)   # tolerate ```json fences / prose
                if m:
                    return json.loads(m.group(0))
                raise
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            last = e
            time.sleep(1.0)
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise RuntimeError(f"Gemini failed: {last}")


# --------------------------------------------------------------------------- #
# shared state
# --------------------------------------------------------------------------- #
class Shared:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = True

        # camera
        self.frame: Optional[bytes] = None
        self.frame_t = 0.0
        self.frame_dims = (0, 0)          # (w, h)
        self.ring: deque = deque(maxlen=48)     # (t, jpg)
        self.cap_stamps: deque = deque(maxlen=30)
        self.cap_errors = 0

        # depth
        self.depth_jpg: Optional[bytes] = None
        self.depth_scores = {"left": None, "center": None, "right": None}
        self.depth_debug: Optional[dict] = None
        self.depth_latency = 0.0
        self.depth_t = 0.0
        self.depth_errors = 0

        # vlm
        self.vlm_decision: Optional[dict] = None
        self.vlm_label: Optional[str] = None
        self.vlm_latency = 0.0
        self.vlm_t = 0.0
        self.vlm_errors = 0
        self.vlm_frames = 0
        self.vlm_history: deque = deque(maxlen=8)

        # autonomy / task execution (VLM drives the car toward S.task)
        self.task: Optional[str] = None        # current task/target prompt
        self.autonomy_on = False               # True between Start and Stop
        self.auto_dry = False                  # dry-run: decide but don't move wheels
        self.auto_status = "idle"              # human-readable status for the UI
        self.auto_action = ""                  # last drive action taken

        # closed-loop robotics mode (ER-1.6: parametric command + measured feedback)
        self.rb_target_point: Optional[list] = None   # [y,x] 0-1000 from ER pointing
        self.rb_cmd: Optional[dict] = None            # last commanded {action,turn_degrees,duration_ms,speed,trim}
        self.rb_measured: Optional[dict] = None       # {yaw_deg, img_change, stall, ...}
        self.rb_calib: dict = {"ms_per_deg": None}    # live self-calibration

        # local-AI explore mode (--vlm explore): carpet staying + Gemini coarse directive
        self.carpet = {"left": None, "center": None, "right": None}   # on-rug confidence 0..1
        self.directive = "explore"                    # Gemini coarse directive
        self.directive_reason = ""
        self.directive_t = 0.0
        self.directive_target_visible = False         # supervisor sees the operator's target ahead
        self.directive_target_close = False           # supervisor says target reached (large/centered)
        self.directive_target_point: Optional[list] = None   # [y,x] 0-1000 (ER pointing) for steering
        self.drift_trim = 0.0                         # learned constant trim that makes 'forward' go straight
        self.explore = {"summary": "", "bias": "center", "visited": 0, "looping": False}


S = Shared()


# --------------------------------------------------------------------------- #
# loops
# --------------------------------------------------------------------------- #
def capture_loop(car_ip: str, hz: float, timeout: float) -> None:
    url = f"http://{car_ip}/capture"
    period = 1.0 / max(0.5, hz)
    push = 0.0
    while S.running:
        t0 = time.time()
        try:
            jpg = urllib.request.urlopen(url, timeout=timeout).read()
            if jpg[:2] != b"\xff\xd8":
                raise ValueError("capture is not JPEG")
            try:
                w, h = Image.open(BytesIO(jpg)).size
            except Exception:
                w, h = (0, 0)
            now = time.time()
            with S.lock:
                S.frame = jpg
                S.frame_t = now
                S.frame_dims = (w, h)
                S.cap_stamps.append(now)
                S.cap_errors = 0
                if now - push >= 0.25:
                    S.ring.append((now, jpg))
                    push = now
        except Exception as e:
            with S.lock:
                S.cap_errors += 1
                n = S.cap_errors
            if n == 1 or n % 10 == 0:
                print(f"[capture] retry {n}: {e}")
            time.sleep(0.25)
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def colorize_depth(d: np.ndarray, dims: tuple[int, int], dbg: Optional[dict] = None) -> bytes:
    lo, hi = float(d.min()), float(d.max())
    g = ((d - lo) / (hi - lo + 1e-6) * 255.0).astype(np.uint8)   # near = bright
    heat = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)              # near = warm/red
    w, h = dims
    if w > 0 and h > 0:
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
    hh, ww = heat.shape[:2]
    # vertical thirds = the left/center/right free_space zones
    for x in (ww // 3, 2 * ww // 3):
        cv2.line(heat, (x, 0), (x, hh), (255, 255, 255), 1)
    if dbg:
        # the actual pixels the openness is derived from (colors are BGR):
        #   cyan   = look-ahead band where "near ahead" is sampled
        #   magenta = floor-reference patch (the close-floor baseline)
        ly0, ly1 = int(hh * dbg["look_rows"][0]), int(hh * dbg["look_rows"][1])
        cv2.rectangle(heat, (0, ly0), (ww - 1, ly1), (255, 255, 0), 2)
        rx0, rx1 = int(ww * dbg["ref_cols"][0]), int(ww * dbg["ref_cols"][1])
        ry0, ry1 = int(hh * dbg["ref_rows"][0]), int(hh * dbg["ref_rows"][1])
        cv2.rectangle(heat, (rx0, ry0), (rx1, ry1), (255, 0, 255), 2)
    ok, buf = cv2.imencode(".jpg", heat, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes()


def depth_loop(depth_model: str, depth_size: int, hz: float) -> None:
    sys.path.insert(0, SCRIPT_DIR)
    import depth_perception as DP  # type: ignore

    print(f"[depth] loading {depth_model} ...")
    print(f"[depth] ready on {DP.load(model_id=depth_model, size=depth_size)}")
    period = 1.0 / max(0.5, hz)
    last_t = -1.0
    while S.running:
        t0 = time.time()
        with S.lock:
            jpg = S.frame
            ft = S.frame_t
            dims = S.frame_dims
        if jpg is None or ft == last_t:
            time.sleep(0.02)
            continue
        try:
            pil = Image.open(BytesIO(jpg)).convert("RGB")
            d = DP.depth(pil)
            dbg = DP.free_space_debug(d)
            scores = {k: dbg["zones"][k]["openness"] for k in ("left", "center", "right")}
            heat = colorize_depth(d, dims, dbg)
            carpet = carpet_scores(pil)            # on-rug confidence per zone (local CV)
            with S.lock:
                S.depth_jpg = heat
                S.depth_scores = scores
                S.depth_debug = dbg
                S.carpet = carpet
                S.depth_latency = time.time() - t0
                S.depth_t = time.time()
                S.depth_errors = 0
            last_t = ft
        except Exception as e:
            with S.lock:
                S.depth_errors += 1
                n = S.depth_errors
            if n == 1 or n % 10 == 0:
                print(f"[depth] retry {n}: {e}")
            time.sleep(0.1)
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def pick_frames(n: int, spacing: float) -> list[tuple[float, bytes]]:
    """Return up to n recent (t, jpg) samples spaced ~`spacing` seconds apart."""
    with S.lock:
        ring = list(S.ring)
    if not ring:
        return []
    latest = ring[-1][0]
    picks: list = []
    for k in range(n - 1, -1, -1):
        target_t = latest - k * spacing
        best = min(ring, key=lambda r: abs(r[0] - target_t))
        if not picks or picks[-1] is not best:
            picks.append(best)          # (t, jpg)
    return picks


def vlm_loop(args: argparse.Namespace, key: Optional[str]) -> None:
    """Produce seek-style decisions for the CURRENT task, using Gemini (cloud) or
    a local mlx-vlm model (FastVLM / Qwen2.5-VL). Calls the VLM ONLY while a task
    is running (Start/Stop from the UI) — idle = no calls = no API spend."""
    backend = args.vlm
    if backend == "off":
        return

    local_mod = None
    if backend == "local":
        sys.path.insert(0, SCRIPT_DIR)
        import vlm_local as local_mod  # type: ignore
        print(f"[vlm] loading local model from {args.local_vlm_model_dir} ...")
        try:
            label = local_mod.load_local(args.local_vlm_model_dir)
        except Exception as e:
            print(f"[vlm] FAILED to load local model: {e}")
            return
        print(f"[vlm] local ready: {label}")
        with S.lock:
            S.vlm_label = label

    hist: deque = deque(maxlen=5)
    while S.running:
        with S.lock:
            active = S.autonomy_on
            task = S.task
        if not active or not task:
            if hist:
                hist.clear()
            time.sleep(0.15)
            continue

        frames = pick_frames(args.vlm_frames, args.vlm_frame_space)
        if not frames:
            time.sleep(0.1)
            continue
        try:
            t0 = time.time()
            if backend == "local":
                dec = local_mod.ask_seek_local(task, frames, list(hist), max_frames=args.vlm_frames)
            else:
                jpgs = [jpg for _, jpg in frames]
                dec = ask_gemini(args.model, key, build_seek_prompt(task), SEEK_SCHEMA, jpgs)
            now = time.time()
            line = (f"vis={dec.get('target_visible')} bearing={dec.get('target_bearing')} "
                    f"size={dec.get('target_size')} safe={dec.get('safe_forward')}")
            hist.appendleft(f"{line} :: {str(dec.get('reason', ''))[:60]}")
            with S.lock:
                S.vlm_decision = dec
                S.vlm_latency = now - t0
                S.vlm_t = now
                S.vlm_frames = len(frames)
                S.vlm_errors = 0
                S.vlm_history.appendleft(
                    {"t": now, "line": line, "reason": str(dec.get("reason", ""))[:90]})
        except Exception as e:
            with S.lock:
                S.vlm_errors += 1
                n = S.vlm_errors
            if n == 1 or n % 3 == 0:
                print(f"[vlm] error {n}: {e}")
        # wait between calls, but bail out promptly if the task is stopped
        end = time.time() + args.vlm_sec
        while S.running and time.time() < end:
            with S.lock:
                if not S.autonomy_on:
                    break
            time.sleep(0.05)


def autonomy_loop(args: argparse.Namespace) -> None:
    """Turn the latest VLM seek-decision (+ depth as a close-range emergency
    bumper) into motion — but only while a task is running. Ported from seek.py;
    firmware turns are now correct (val3=left, val4=right, +trim=right)."""
    ip = args.car_ip
    MAG = {"none": 0, "slight": args.trim_slight, "medium": args.trim_medium, "hard": args.trim_hard}
    BEAR = {"far_left": -1, "left": -1, "center": 0, "right": 1, "far_right": 1, "none": 0}

    def stop():
        car_ctl(ip, "var=car&val=5")

    def cruise(trim):
        car_ctl(ip, f"var=trim&val={int(trim)}")
        car_ctl(ip, f"var=speed&val={args.drive_speed}")
        car_ctl(ip, "var=car&val=1")

    def spin(side, secs):
        car_ctl(ip, "var=car&val=" + ("3" if side == "left" else "4"))
        time.sleep(secs)
        stop()

    def set_action(a):
        with S.lock:
            S.auto_action = a
            S.auto_status = "running"

    def finish(msg):
        with S.lock:
            S.autonomy_on = False
            S.auto_status = msg
            S.auto_action = ""

    was_on = False
    spins = 0
    centered_streak = 0
    while S.running:
        with S.lock:
            on = S.autonomy_on
            dry = S.auto_dry
            dec = S.vlm_decision
            dec_t = S.vlm_t
            center_open = S.depth_scores.get("center")
        if not on:
            if was_on:
                stop()
                was_on = False
            time.sleep(0.08)
            continue
        if not was_on:           # just started: clear stale decision
            was_on = True
            spins = 0
            centered_streak = 0
        if dec is None or dec_t <= 0:
            with S.lock:
                S.auto_status = "waiting for first VLM decision…"
            time.sleep(0.1)
            continue
        if time.time() - dec_t > args.goal_stale_sec:
            set_action("VLM stale -> holding")
            stop()
            time.sleep(0.1)
            continue

        visible = bool(dec.get("target_visible"))
        bearing = dec.get("target_bearing", "none")
        size = dec.get("target_size", "none")
        strength = dec.get("turn_strength", "medium")
        strat = dec.get("search_strategy", "spin_right")
        safe_fwd = bool(dec.get("safe_forward", True))

        # --- arrival: stop the car AND stop querying the VLM ---
        # Gemini often caps the target at "medium" and never says "large", so we
        # accept several signals: its own `arrived` flag, large+roughly-centered,
        # or medium+centered sustained for a few frames.
        if visible and bearing == "center" and size in ("medium", "large"):
            centered_streak += 1
        else:
            centered_streak = 0
        reached = visible and (
            bool(dec.get("arrived"))
            or (size == "large" and bearing in ("center", "left", "right"))
            or (size == "medium" and bearing == "center" and centered_streak >= args.arrive_medium_frames)
        )
        if reached:
            stop()
            finish(f"reached target ({size}/{bearing}): {str(dec.get('reason',''))[:55]}")
            continue

        # close-range emergency bumper (depth), overrides everything but final approach
        if center_open is not None and center_open < args.bump and not (visible and size == "large"):
            side = "left" if bearing in ("left", "far_left") else "right"
            set_action(f"BUMPER spin {side} (center_open={center_open:.2f})")
            if not dry:
                spin(side, args.turn_pulse)
            time.sleep(1.0 / args.drive_hz)
            continue

        if visible:
            spins = 0
            if bearing in ("far_left", "far_right") and strength == "hard":
                side = "left" if bearing == "far_left" else "right"
                set_action(f"spin {side} (target at far edge)")
                if not dry:
                    spin(side, args.turn_pulse)
                time.sleep(1.0 / args.drive_hz)
                continue
            if bearing == "center" and not safe_fwd:
                set_action("centered but blocked -> nudge")
                if not dry:
                    spin("right", args.turn_pulse)
                time.sleep(1.0 / args.drive_hz)
                continue
            trim = BEAR.get(bearing, 0) * MAG.get(strength, args.trim_medium)
            set_action(f"approach {bearing}/{size}: arc trim={trim:+d}")
            if not dry:
                cruise(trim)
        else:
            want_spin = strat in ("spin_left", "spin_right", "turn_around")
            if want_spin and spins >= args.scan_max and safe_fwd:
                want_spin = False        # stop pirouetting; relocate to a new vantage
            if want_spin:
                side = "left" if strat == "spin_left" else "right"
                secs = args.search_pulse * (3 if strat == "turn_around" else 1)
                set_action(f"search {strat}")
                if not dry:
                    spin(side, secs)
                spins += 1
                time.sleep(1.0 / args.drive_hz)
                continue
            elif safe_fwd:
                set_action("search forward")
                spins = 0
                if not dry:
                    cruise(0)
            else:
                set_action("search nudge")
                if not dry:
                    spin("right", args.search_pulse)
                spins += 1
                time.sleep(1.0 / args.drive_hz)
                continue
        time.sleep(1.0 / args.drive_hz)
    stop()


# --------------------------------------------------------------------------- #
# car control proxy (manual drive only; never autonomous)
# --------------------------------------------------------------------------- #
def car_ctl(car_ip: str, query: str, timeout: float = 4.0) -> bool:
    try:
        urllib.request.urlopen(f"http://{car_ip}/control?{query}", timeout=timeout).read()
        return True
    except Exception as e:
        print(f"[ctl] {query}: {e}")
        return False


# --------------------------------------------------------------------------- #
# closed-loop robotics mode: Gemini Robotics-ER 1.6 emits parametric commands,
# we execute timed pulses, measure the actual motion from the camera, feed that
# back, and self-calibrate ms-per-degree. (Reuses gemini_drive.py patterns.)
# --------------------------------------------------------------------------- #
ROBOTICS_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["forward", "back", "turn_left", "turn_right", "stop"]},
        "speed": {"type": "integer"},
        "duration_ms": {"type": "integer"},
        "turn_degrees": {"type": "integer"},
        "trim": {"type": "integer"},
        "target_point": {"type": "array", "items": {"type": "integer"}},   # [y,x] 0-1000
        "target_visible": {"type": "boolean"},
        "target_size": {"type": "string", "enum": ["none", "small", "medium", "large"]},
        "path_clear": {"type": "boolean"},
        "arrived": {"type": "boolean"},
        "reason": {"type": "string"},
        "expected_effect": {"type": "string"},
    },
    "required": ["action", "speed", "duration_ms", "turn_degrees", "trim", "target_visible",
                 "target_size", "path_clear", "arrived", "reason", "expected_effect"],
}


def build_robotics_prompt(task: str, feedback: Optional[str], ms_per_deg: float, lim: dict) -> str:
    return (
        "You directly drive a small ESP32-CAM robot car from its forward camera. "
        "Pick ONE bounded motor command. The program executes it, then reports back how "
        "the car ACTUALLY moved (measured from the camera) so you correct the next one.\n"
        f'TASK: find and drive right up to: "{task}".\n'
        "target_point is [y,x] normalized 0-1000 (origin top-left); point at the target "
        "if visible, else set target_visible=false.\n"
        "Movement model: forward/back use speed + duration_ms (trim>0 arcs right, <0 arcs left); "
        "turn_left/turn_right turn in place by turn_degrees; stop when arrived.\n"
        f"Limits: speed 0..{lim['max_speed']}; forward {lim['min_ms']}..{lim['max_ms']} ms; "
        f"turn_degrees up to {lim['max_deg']}. (~{ms_per_deg:.0f} ms per degree is applied for you.)\n"
        "Adapt using the feedback: if a turn under/overshot, change turn_degrees next time; "
        "if forward reported NO PROGRESS you are blocked — turn or back up rather than push. "
        "arrived=true only when the target is large and roughly centered.\n"
        f"FEEDBACK from your last command: {feedback or 'none yet (first step)'}\n"
        "Reply ONLY with the required JSON schema; keep reason and expected_effect short."
    )


def _gray_arr(jpg: bytes, w: int = 64, h: int = 48) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(jpg)).convert("L").resize((w, h)), dtype=np.float32)


def image_change(prev: bytes, cur: bytes) -> Optional[float]:
    try:
        a = Image.open(BytesIO(prev)).convert("L").resize((32, 24)).tobytes()
        b = Image.open(BytesIO(cur)).convert("L").resize((32, 24)).tobytes()
        return sum(abs(x - y) for x, y in zip(a, b)) / (255.0 * len(a))
    except Exception:
        return None


def measure_yaw_deg(a_jpg: bytes, b_jpg: bytes, hfov: float = 60.0) -> float:
    """+deg => the car rotated RIGHT. Estimated from the horizontal shift of the
    scene between frames a->b (column-mean cross-correlation)."""
    try:
        a = _gray_arr(a_jpg); b = _gray_arr(b_jpg); W = a.shape[1]
        ca = a.mean(axis=0); cb = b.mean(axis=0)
        ca = ca - ca.mean(); cb = cb - cb.mean()
        cc = np.correlate(cb, ca, mode="full")
        lag = int(cc.argmax() - (W - 1))    # +lag => scene moved RIGHT => car turned LEFT
        return -(lag / W) * hfov            # flip so +deg = car turned right
    except Exception:
        return 0.0


def measure_forward(a_jpg: bytes, b_jpg: bytes) -> tuple[float, Optional[float]]:
    """(vertical_shift_fraction, image_change). Forward motion expands/scrolls the
    scene; we report the vertical row-shift plus the scalar image change."""
    ic = image_change(a_jpg, b_jpg)
    try:
        a = _gray_arr(a_jpg); b = _gray_arr(b_jpg); H = a.shape[0]
        ra = a.mean(axis=1); rb = b.mean(axis=1)
        ra = ra - ra.mean(); rb = rb - rb.mean()
        cc = np.correlate(rb, ra, mode="full")
        vlag = int(cc.argmax() - (H - 1))
        return (vlag / H), ic
    except Exception:
        return 0.0, ic


def clamp_robotics(raw: dict, args: argparse.Namespace, ms_per_deg: float) -> dict:
    action = enum(raw.get("action"), {"forward", "back", "turn_left", "turn_right", "stop"}, "stop")
    visible = bool(raw.get("target_visible", False))
    size = enum(raw.get("target_size"), {"none", "small", "medium", "large"}, "none")
    arrived = bool(raw.get("arrived", False))
    if visible and size == "large":
        arrived = True
    if arrived:
        action = "stop"
    tp = raw.get("target_point")
    point = None
    if isinstance(tp, (list, tuple)) and len(tp) == 2:
        try:
            point = [int(clamp(float(tp[0]), 0, 1000)), int(clamp(float(tp[1]), 0, 1000))]
        except Exception:
            point = None

    def _int(v, d):
        try:
            return int(round(float(v)))
        except Exception:
            return d

    speed = _int(raw.get("speed"), args.drive_speed)
    trim = int(clamp(_int(raw.get("trim"), 0), -args.rb_max_trim, args.rb_max_trim))
    deg = int(clamp(abs(_int(raw.get("turn_degrees"), args.rb_default_deg)), 0, args.rb_max_deg))
    ms = _int(raw.get("duration_ms"), args.rb_default_ms)

    if action == "stop":
        speed = trim = deg = 0
        ms = 0
    elif action in ("turn_left", "turn_right"):
        speed = 0
        trim = 0
        ms = int(clamp(deg * ms_per_deg, args.rb_min_turn_ms, args.rb_max_turn_ms))
    elif action == "back":
        speed = int(clamp(speed or args.rb_escape_speed, args.drive_speed, args.rb_escape_speed))
        trim = 0
        deg = 0
        ms = int(clamp(ms, args.rb_min_ms, args.rb_max_back_ms))
    else:  # forward
        speed = int(clamp(speed or args.drive_speed, args.drive_speed, args.rb_max_speed))
        deg = 0
        ms = int(clamp(ms, args.rb_min_ms, args.rb_max_ms))
        if visible and size in ("medium", "large"):
            trim = int(clamp(trim, -args.rb_near_trim, args.rb_near_trim))

    return {"action": action, "speed": speed, "duration_ms": ms, "turn_degrees": deg, "trim": trim,
            "target_point": point, "target_visible": visible, "target_size": size,
            "path_clear": bool(raw.get("path_clear", True)), "arrived": arrived,
            "reason": str(raw.get("reason", ""))[:160],
            "expected_effect": str(raw.get("expected_effect", ""))[:160]}


def closedloop_loop(args: argparse.Namespace, key: Optional[str]) -> None:
    """Sequential act -> measure -> feedback loop driven by Gemini Robotics-ER 1.6,
    active only while a task is running."""
    ip = args.car_ip
    model = args.robotics_model
    ms_per_deg = args.ms_per_90 / 90.0
    lim = {"max_speed": args.rb_max_speed, "min_ms": args.rb_min_ms,
           "max_ms": args.rb_max_ms, "max_deg": args.rb_max_deg}

    def stop():
        car_ctl(ip, "var=car&val=5")

    def pulse(action, speed, trim, ms):
        secs = max(0.0, ms / 1000.0)
        if action == "stop" or ms <= 0:
            stop(); return
        if action == "forward":
            car_ctl(ip, f"var=speed&val={speed}"); car_ctl(ip, f"var=trim&val={trim}"); car_ctl(ip, "var=car&val=1")
        elif action == "back":
            car_ctl(ip, f"var=speed&val={speed}"); car_ctl(ip, "var=trim&val=0"); car_ctl(ip, "var=car&val=2")
        elif action == "turn_left":
            car_ctl(ip, "var=car&val=3")
        elif action == "turn_right":
            car_ctl(ip, "var=car&val=4")
        else:
            stop(); return
        time.sleep(secs)
        stop()

    def set_action(a):
        with S.lock:
            S.auto_action = a
            S.auto_status = "running"

    def finish(msg):
        with S.lock:
            S.autonomy_on = False
            S.auto_status = msg
            S.auto_action = ""

    was_on = False
    feedback = None
    while S.running:
        with S.lock:
            on = S.autonomy_on
            dry = S.auto_dry
            task = S.task
            cur = S.frame
            center_open = S.depth_scores.get("center")
        if not on:
            if was_on:
                stop(); was_on = False
            time.sleep(0.1)
            continue
        if not was_on:
            was_on = True
            feedback = None
            car_ctl(ip, f"var=turnspeed&val={int(clamp(args.turnspeed, 0, 255))}")
        if cur is None:
            with S.lock:
                S.auto_status = "waiting for camera…"
            time.sleep(0.1)
            continue

        # 1) decide — ER 1.6 with the last command's measured feedback
        try:
            t0 = time.time()
            prompt = build_robotics_prompt(task, feedback, ms_per_deg, lim)
            raw = ask_gemini(model, key, prompt, ROBOTICS_SCHEMA, [cur])
            dec = clamp_robotics(raw, args, ms_per_deg)
            lat = time.time() - t0
        except Exception as e:
            with S.lock:
                S.vlm_errors += 1
                n = S.vlm_errors
            if n == 1 or n % 3 == 0:
                print(f"[robotics] error {n}: {e}")
            time.sleep(0.5)
            continue
        with S.lock:
            if not S.autonomy_on:
                stop(); continue
            S.vlm_decision = dec
            S.vlm_latency = lat
            S.vlm_t = time.time()
            S.vlm_errors = 0
            S.rb_target_point = dec.get("target_point")
        tp = dec.get("target_point")

        # 2) arrival -> stop + stop querying
        if dec.get("arrived") or (dec.get("target_visible") and dec.get("target_size") == "large"):
            stop()
            finish(f"reached target: {str(dec.get('reason',''))[:60]}")
            continue

        # 3) depth emergency bumper overrides
        if center_open is not None and center_open < args.bump and not (
                dec.get("target_visible") and dec.get("target_size") == "large"):
            side = "turn_right" if (tp and tp[1] >= 500) else "turn_left"
            set_action(f"BUMPER {side} (center_open={center_open:.2f})")
            if not dry:
                pulse(side, 0, 0, int(clamp(25 * ms_per_deg, args.rb_min_turn_ms, args.rb_max_turn_ms)))
            feedback = f"emergency: obstacle close ahead (center_open={center_open:.2f}); auto-turned {side}"
            with S.lock:
                S.rb_cmd = {"action": side, "turn_degrees": 25, "duration_ms": 0, "speed": 0, "trim": 0}
                S.rb_measured = None
            time.sleep(args.settle_sec)
            continue

        # 4) execute the parametric command, then measure what actually happened
        action = dec["action"]; speed = dec["speed"]; trim = dec["trim"]
        ms = dec["duration_ms"]; deg = dec["turn_degrees"]
        with S.lock:
            S.rb_cmd = {"action": action, "turn_degrees": (deg if action in ("turn_left", "turn_right") else 0),
                        "duration_ms": ms, "speed": speed, "trim": trim}
            S.auto_action = f"{action} deg={deg} ms={ms} spd={speed} trim={trim:+d}"
        before = cur
        after = None
        if not dry:
            pulse(action, speed, trim, ms)
            time.sleep(args.settle_sec)
            with S.lock:
                after = S.frame

        measured = None
        if (not dry) and before and after:
            yaw = measure_yaw_deg(before, after, args.cam_hfov)
            vfrac, ic = measure_forward(before, after)
            stall = (action == "forward" and ic is not None and ic < args.stuck_change and abs(yaw) < 3.0)
            measured = {"yaw_deg": round(yaw, 1),
                        "img_change": (round(ic, 3) if ic is not None else None),
                        "vshift": round(vfrac, 3), "stall": stall}
            if action in ("turn_left", "turn_right") and abs(yaw) >= 3.0 and ms > 0:
                obs = ms / abs(yaw)
                ms_per_deg = float(clamp(0.7 * ms_per_deg + 0.3 * obs, 4.0, 40.0))
            if action in ("turn_left", "turn_right"):
                want = deg * (1 if action == "turn_right" else -1)
                ratio = (yaw / want) if want else 0.0
                feedback = f"turn {action.split('_')[1]} cmd={deg}deg -> measured {yaw:+.0f}deg ({ratio:.1f}x of intended)"
            elif action == "forward":
                feedback = (f"forward {ms}ms -> NO PROGRESS (blocked/stall)" if stall
                            else f"forward {ms}ms -> moved (img_change={ic:.3f})")
            elif action == "back":
                feedback = f"back {ms}ms -> img_change={ic:.3f}"
            else:
                feedback = "stopped"
            if tp:
                feedback += f"; target_point was [{tp[0]},{tp[1]}]"
            feedback += f"; calib ms/deg={ms_per_deg:.1f}"
        elif dry:
            feedback = "dry-run: command shown but NOT executed (no measurement)"

        with S.lock:
            S.rb_measured = measured
            S.rb_calib = {"ms_per_deg": round(ms_per_deg, 1)}
            S.vlm_history.appendleft({"t": time.time(), "line": f"{action} deg={deg} ms={ms}",
                                      "reason": (feedback or "")[:90]})

        endt = time.time() + args.vlm_sec
        while S.running and time.time() < endt:
            with S.lock:
                if not S.autonomy_on:
                    break
            time.sleep(0.05)
    stop()


# --------------------------------------------------------------------------- #
# local-AI explore mode: the local controller roams (stays on the rug, keeps
# clear of objects, covers the room via a frontier map); Gemini only issues a
# coarse directive (explore / search_here / move_on / turn_around / stop).
# --------------------------------------------------------------------------- #
def carpet_scores(pil, ref_rows=(0.82, 0.98), ref_cols=(0.30, 0.70),
                  look_rows=(0.55, 0.80), hue_w=2.0, sat_w=1.0) -> dict:
    """On-rug confidence per left/center/right look-ahead zone, comparing each to the
    rug patch directly under the camera (always 'the surface I'm on now'). exp(-dist)
    over mean hue (circular) + saturation in HSV; brightness ignored for lighting
    robustness. 1.0 = same surface as under the wheels, low = bare floor / rug edge /
    object ahead. Fast classical CV (validated on the saved frames)."""
    rgb = np.asarray(pil.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, w, _ = hsv.shape
    ref = hsv[int(h * ref_rows[0]):int(h * ref_rows[1]), int(w * ref_cols[0]):int(w * ref_cols[1])]
    rh, rs = float(ref[..., 0].mean()), float(ref[..., 1].mean())
    lo = hsv[int(h * look_rows[0]):int(h * look_rows[1]), :]
    out = {}
    for name, (a, b) in (("left", (0, w // 3)), ("center", (w // 3, 2 * w // 3)),
                         ("right", (2 * w // 3, w))):
        z = lo[:, a:b]
        zh, zs = float(z[..., 0].mean()), float(z[..., 1].mean())
        dh = min(abs(rh - zh), 180 - abs(rh - zh)) / 90.0   # circular hue distance
        ds = abs(rs - zs) / 128.0
        out[name] = round(float(math.exp(-(hue_w * dh + sat_w * ds))), 3)
    return out


class ExploreMap:
    """Tiny egocentric topological map for coverage exploration (slim port of
    reflex_drive.ExplorerMemory): approximate pose from executed motions, visited-cell
    counts, frontier bonus for unvisited directions, blocked memory, commit-to-direction,
    and loop detection. Operates on a per-zone `drivable` score (depth AND on-rug)."""

    def __init__(self, args):
        self.a = args
        self.cell_size = args.map_cell_size
        self.x = self.y = self.heading = 0.0
        self.entries = {}
        self.blocked = {}
        self.last_t = None
        self.last_cell = None
        self.last_new_cell_t = 0.0
        self.committed = "center"
        self.committed_until = 0.0
        self.looping = False
        self.loop_reason = ""

    def cell(self, x=None, y=None):
        xx = self.x if x is None else x
        yy = self.y if y is None else y
        return (int(round(xx / self.cell_size)), int(round(yy / self.cell_size)))

    def dir_cell(self, rel, dist=1.0):
        off = {"left": -0.7, "center": 0.0, "right": 0.7}[rel]
        a = self.heading + off
        return self.cell(self.x + math.cos(a) * dist, self.y + math.sin(a) * dist)

    def update(self, action, speed, trim, blocked_ahead, now):
        dt = 0.0 if self.last_t is None else max(0.0, min(1.0, now - self.last_t))
        self.last_t = now
        if self.last_new_cell_t == 0.0:
            self.last_new_cell_t = now
        if action.startswith("arc") or action == "forward":
            dist = dt * (0.12 + 0.16 * max(1, speed))
            self.heading += (trim / max(1, self.a.max_trim)) * dt * 0.9
            self.x += math.cos(self.heading) * dist
            self.y += math.sin(self.heading) * dist
        elif "back" in action:
            dist = dt * 0.2
            self.x -= math.cos(self.heading) * dist
            self.y -= math.sin(self.heading) * dist
        elif "left" in action:
            self.heading -= dt * 1.8
        elif "right" in action:
            self.heading += dt * 1.8
        cur = self.cell()
        if cur != self.last_cell:
            self.entries[cur] = self.entries.get(cur, 0) + 1
            if self.entries[cur] == 1:
                self.last_new_cell_t = now
            self.last_cell = cur
        if blocked_ahead:
            self.blocked[self.dir_cell("center")] = now
        no_new = now - self.last_new_cell_t
        revisits = self.entries.get(cur, 0)
        if no_new >= self.a.loop_no_new_cell_sec:
            self.looping, self.loop_reason = True, f"no new cell {no_new:.0f}s"
        elif revisits >= self.a.loop_cell_entries:
            self.looping, self.loop_reason = True, f"cell revisited {revisits}x"
        else:
            self.looping, self.loop_reason = False, ""

    def dir_score(self, rel, drivable, now):
        cell = self.dir_cell(rel)
        entries = self.entries.get(cell, 0)
        blocked_recent = cell in self.blocked and now - self.blocked[cell] < self.a.block_memory_sec
        return (drivable[rel] * 2.0
                + (self.a.frontier_bonus if entries == 0 else 0.0)
                - entries * self.a.revisit_penalty
                - (self.a.blocked_penalty if blocked_recent else 0.0))

    def choose_bias(self, drivable, now, prefer_new=False):
        choices = ("left", "right") if (self.looping or prefer_new) else ("left", "center", "right")
        scored = dict((r, self.dir_score(r, drivable, now)) for r in choices)
        best = max(scored, key=scored.get)
        if (now < self.committed_until and self.committed in scored
                and scored[self.committed] >= scored[best] - self.a.commit_switch_margin):
            return self.committed
        self.committed = best
        self.committed_until = now + self.a.explore_commit_sec
        return best

    def summary(self):
        cur = self.cell()
        hd = int((math.degrees(self.heading) % 360 + 22.5) // 45 * 45) % 360
        return (f"cell={cur} heading~{hd}deg visited={len(self.entries)} "
                f"bias={self.committed} looping={self.looping}"
                f"{('(' + self.loop_reason + ')') if self.looping else ''}")


DIRECTIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "directive": {"type": "string",
                      "enum": ["explore", "search_here", "move_on", "turn_around", "stop"]},
        "target_visible": {"type": "boolean"},   # the operator's target is in view, roughly ahead
        "target_close": {"type": "boolean"},      # target is large & centered -> reached it
        "target_point": {"type": "array", "items": {"type": "integer"}},   # [y,x] 0-1000 for steering
        "reason": {"type": "string"},
    },
    "required": ["directive", "target_visible", "target_close", "reason"],
}


def build_directive_prompt(task: Optional[str], map_summary: str, hist) -> str:
    if task:
        extra = (
            f'The operator is looking for a specific target: "{task}".\n'
            " - Set target_visible=true whenever that target is in view roughly ahead of the robot.\n"
            " - When target_visible, also POINT at it: target_point=[y,x] normalized 0-1000 "
            "(origin top-left). The controller uses x to steer straight at the target.\n"
            " - Set target_close=true when that target is LARGE and roughly CENTERED (the robot has "
            "essentially reached it) — the local controller will then stop right at it.\n"
            " - Use search_here when you can see the target and want a closer approach.\n"
        )
    else:
        extra = " - (No specific target: keep target_visible and target_close false.)\n"
    mem = "\n".join(hist) if hist else "none yet"
    return (
        "You SUPERVISE a small robot car that explores a room on its own. A local "
        "controller already handles ALL driving and safety — it stays on the rug, keeps "
        "clear of objects, and stops at the target on its own — so do NOT give motion "
        "commands. From the camera and the exploration map, give ONE high-level directive:\n"
        " - explore: keep roaming to cover new areas.\n"
        " - search_here: something here is worth a closer look; the robot will slow and scan.\n"
        " - move_on: this area looks done/uninteresting; go find a new area.\n"
        " - turn_around: explore behind the robot.\n"
        " - stop: stop exploring (only if asked to, or clearly nothing more to do).\n"
        f"{extra}"
        f"Exploration map: {map_summary}\n"
        f"Your recent directives (newest first):\n{mem}\n"
        "Reply with ONLY the JSON schema; keep reason short."
    )


def directive_loop(args: argparse.Namespace, key: Optional[str]) -> None:
    """Slow cloud-Gemini supervisor: returns one coarse directive while roaming."""
    hist: deque = deque(maxlen=5)
    while S.running:
        with S.lock:
            active = S.autonomy_on
            task = S.task
            summary = S.explore.get("summary", "")
        if not active:
            time.sleep(0.2)
            continue
        frames = pick_frames(1, args.vlm_frame_space)
        if not frames:
            time.sleep(0.1)
            continue
        tvis = False
        try:
            t0 = time.time()
            jpgs = [jpg for _, jpg in frames]
            # Gemini Robotics-ER supervises explore: coarse directive + a precise point
            # at the target (target_point) used by the local loop to steer straight at it.
            dec = ask_gemini(args.robotics_model, key, build_directive_prompt(task, summary, list(hist)),
                             DIRECTIVE_SCHEMA, jpgs)
            d = enum(dec.get("directive"), {"explore", "search_here", "move_on", "turn_around", "stop"}, "explore")
            reason = str(dec.get("reason", ""))[:120]
            tvis = bool(dec.get("target_visible", False))
            tclose = bool(dec.get("target_close", False))
            tp = dec.get("target_point")
            point = None
            if isinstance(tp, (list, tuple)) and len(tp) == 2:
                try:
                    point = [int(clamp(float(tp[0]), 0, 1000)), int(clamp(float(tp[1]), 0, 1000))]
                except Exception:
                    point = None
            with S.lock:
                S.directive = d
                S.directive_reason = reason
                S.directive_target_visible = tvis
                S.directive_target_close = tclose
                S.directive_target_point = point if tvis else None
                S.directive_t = time.time()
                S.vlm_latency = time.time() - t0
                S.vlm_t = time.time()
                S.vlm_errors = 0
                tag = (" [TARGET CLOSE]" if tclose else " [target ahead]" if tvis else "")
                S.vlm_history.appendleft({"t": time.time(), "line": d.upper() + tag, "reason": reason})
            hist.appendleft(f"{d}{' (target close)' if tclose else ' (target ahead)' if tvis else ''}: {reason[:50]}")
        except Exception as e:
            with S.lock:
                S.vlm_errors += 1
                n = S.vlm_errors
            if n == 1 or n % 3 == 0:
                print(f"[directive] error {n}: {e}")
        # poll faster while approaching a visible target so the steering point stays fresh
        interval = args.approach_vlm_sec if tvis else args.vlm_sec
        endt = time.time() + interval
        while S.running and time.time() < endt:
            with S.lock:
                if not S.autonomy_on:
                    break
            time.sleep(0.05)


def explore_loop(args: argparse.Namespace) -> None:
    """Fast local controller: roams while staying on the rug and keeping clear of
    objects, biased by the frontier map and the current Gemini directive. Owns motors."""
    ip = args.car_ip
    mem = ExploreMap(args)
    was_on = False
    last_turn_around = 0.0
    drift_trim = 0.0              # learned constant trim that makes 'forward' go straight
    prev_straight_jpg = None      # last frame captured while driving (near-)straight, for drift measurement

    def stop():
        car_ctl(ip, "var=car&val=5")

    def cruise(trim, speed):
        car_ctl(ip, f"var=trim&val={int(trim)}")
        car_ctl(ip, f"var=speed&val={int(speed)}")
        car_ctl(ip, "var=car&val=1")

    def spin(side, secs):
        car_ctl(ip, "var=car&val=" + ("3" if side == "left" else "4"))
        time.sleep(secs)
        stop()

    def reverse(secs):
        car_ctl(ip, f"var=speed&val={args.escape_speed}")
        car_ctl(ip, "var=trim&val=0")
        car_ctl(ip, "var=car&val=2")
        time.sleep(secs)
        stop()

    def publish(status, action):
        with S.lock:
            S.auto_status = status
            S.auto_action = action
            S.explore = {"summary": mem.summary(), "bias": mem.committed,
                         "visited": len(mem.entries), "looping": mem.looping}

    while S.running:
        with S.lock:
            on = S.autonomy_on
            dry = S.auto_dry
            op = dict(S.depth_scores)
            cp = dict(S.carpet)
            directive = S.directive
            task = S.task
            tvis = S.directive_target_visible
            tclose = S.directive_target_close
            tp = S.directive_target_point
            dir_t = S.directive_t
            frame = S.frame
            ft = S.frame_t
        if not on:
            if was_on:
                stop(); was_on = False
            time.sleep(0.1)
            continue
        if not was_on:
            was_on = True
            prev_straight_jpg = None
            car_ctl(ip, f"var=turnspeed&val={int(clamp(args.turnspeed, 0, 255))}")
        now = time.time()
        oc = op.get("center")
        if oc is None or ft == 0 or now - ft > args.stale_view_sec:
            with S.lock:
                S.auto_status = "waiting for perception…"
            stop()
            time.sleep(0.1)
            continue

        # learn the constant drift trim from a (near-)straight forward segment: if the
        # scene yawed while we intended to go straight, that's drift -> compensate.
        if (not dry) and prev_straight_jpg is not None and frame is not None and frame is not prev_straight_jpg:
            yaw = measure_yaw_deg(prev_straight_jpg, frame, args.cam_hfov)   # +deg = drifted right
            if abs(yaw) < 25.0:
                step = math.copysign(min(abs(yaw), 6.0), yaw)
                drift_trim = float(clamp(0.85 * drift_trim - 0.15 * step * args.drift_gain,
                                         -args.max_drift_trim, args.max_drift_trim))
        prev_straight_jpg = None   # re-armed below only when we drive ~straight this tick
        if directive == "stop":
            publish("directive: stop", "")
            stop()
            time.sleep(0.15)
            continue

        # drivable per zone = clear (depth) AND on the rug (carpet)
        drivable = {}
        for z in ("left", "center", "right"):
            o = op.get(z); c = cp.get(z)
            drivable[z] = min(0.0 if o is None else o, 1.0 if c is None else c)
        cc = cp.get("center", 1.0)

        # --- target approach & LOCAL arrival stop (when the operator gave a target) ---
        # The slow supervisor only flags target_visible/target_close; the fast local
        # loop decides the actual stop so the robot can't coast past the target.
        target_recent = bool(task) and tvis and (dir_t > 0) and (now - dir_t <= args.target_memory_sec)
        # Stop is gated on DEPTH proximity (so an over-eager supervisor can't stop us far
        # away, and we still stop the instant the target is genuinely close).
        arrived = bool(task) and (
            (target_recent and oc < args.arrive_open)        # very close, target confirmed ahead
            or (tclose and oc < args.keep_clear)             # supervisor says reached + moderately close
        )
        approaching = bool(task) and (target_recent or tclose) and not arrived
        if arrived:
            stop()
            with S.lock:
                S.autonomy_on = False
                S.auto_status = "reached target — stopped"
                S.auto_action = ""
            continue

        # turn_around maneuver (rate-limited)
        if directive == "turn_around" and now - last_turn_around > args.turn_around_cooldown:
            publish("directive: turn_around", "spin ~180")
            if not dry:
                spin("right", args.turn_around_sec)
            last_turn_around = now
            mem.update("right", 0, 0, False, now)
            time.sleep(1.0 / args.drive_hz)
            continue

        best = mem.choose_bias(drivable, now, prefer_new=(directive == "move_on"))
        best_side = best if best in ("left", "right") else (
            "left" if drivable["left"] >= drivable["right"] else "right")
        # while approaching the target, a close object dead-ahead IS the target — don't
        # treat low openness as "blocked" and pivot away (only the rug edge may divert).
        center_blocked = (oc < args.block and not approaching) or cc < args.carpet_block

        if center_blocked:
            if max(drivable["left"], drivable["right"]) < args.side_open:
                action = f"escape back+{best_side}"
                if not dry:
                    reverse(args.back_secs); spin(best_side, args.escape_turn_sec)
                mem.update(action, args.escape_speed, 0, True, now)
            else:
                action = f"pivot {best_side}"
                if not dry:
                    spin(best_side, args.turn_pulse)
                mem.update("left" if best_side == "left" else "right", 0, 0, True, now)
            publish("running", action)
            time.sleep(1.0 / args.drive_hz)
            continue

        # forward arc: steer toward the more-drivable side + frontier bias.
        # drift_trim (learned) is added to every forward command so 'straight' is straight.
        open_bias = clamp((drivable["right"] - drivable["left"]) * args.open_gain, -1.0, 1.0)
        frame_bias = {"left": -0.6, "center": 0.0, "right": 0.6}[best]
        steer = clamp(open_bias + args.map_weight * frame_bias, -1.0, 1.0)
        speed = args.drive_speed
        intent_straight = False
        if approaching:
            # steer to keep the target centered (visual servo on the ER point) and crawl,
            # so it heads straight at the target and the local arrival-stop lands in time.
            speed = min(speed, args.target_approach_speed)
            if tp and len(tp) == 2:
                off = clamp((tp[1] - 500.0) / 500.0, -1.0, 1.0)   # target x: + = right of center
                trim = int(clamp(off * args.approach_gain * args.max_trim + drift_trim,
                                 -args.max_trim, args.max_trim))
                action = f"steer→target x={tp[1]} trim={trim:+d}"
            else:
                trim = int(clamp(steer * args.max_trim * 0.3 + drift_trim, -args.max_trim, args.max_trim))
                action = f"approach (no point) trim={trim:+d}"
            if cc < args.carpet_edge:                 # never leave the rug, even chasing the target
                push = args.max_trim * 0.5 * (1 if best_side == "right" else -1)
                trim = int(clamp(trim + push, -args.max_trim, args.max_trim))
                action = f"approach, hug rug trim={trim:+d}"
        elif oc < args.keep_clear or cc < args.carpet_edge:
            # keep-clear: object/edge getting close ahead -> slow and steer away EARLY
            speed = max(args.min_drive_speed, speed - 1)
            push = args.max_trim * 0.5 * (1 if best_side == "right" else -1)
            trim = int(clamp(steer * args.max_trim * 1.4 + push + drift_trim, -args.max_trim, args.max_trim))
            action = f"keep-clear arc trim={trim:+d}"
        else:
            trim = int(clamp(steer * args.max_trim + drift_trim, -args.max_trim, args.max_trim))
            action = f"arc trim={trim:+d}"
            intent_straight = abs(steer) < 0.15
        if directive == "search_here" and not approaching:
            speed = min(speed, args.search_speed)
            action = "search-here " + action
        if not dry:
            cruise(trim, speed)
            if intent_straight:
                prev_straight_jpg = frame     # arm drift measurement over this straight segment
        mem.update("arc", speed, trim, False, now)
        with S.lock:
            S.drift_trim = round(drift_trim, 1)
        publish("running", f"{action} (drift {drift_trim:+.0f})")
        time.sleep(1.0 / args.drive_hz)
    stop()


# --------------------------------------------------------------------------- #
# web app
# --------------------------------------------------------------------------- #
def make_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="car_robot dashboard")
    THR = {"block": args.block, "slow": args.slow, "open": args.open}

    default_task = args.target or ("yellow rubber bath duck / toy duckie "
                                   "(жёлтая резиновая уточка для ванной)")
    default_task_html = (default_task.replace("&", "&amp;").replace('"', "&quot;")
                         .replace("<", "&lt;").replace(">", "&gt;"))

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE.replace("__DEFAULT_TASK__", default_task_html)

    @app.get("/frame.jpg")
    def frame() -> Response:
        with S.lock:
            jpg = S.frame
        if jpg is None:
            return Response(status_code=503)
        return Response(content=jpg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/depth.jpg")
    def depth_img() -> Response:
        with S.lock:
            jpg = S.depth_jpg
        if jpg is None:
            return Response(status_code=503)
        return Response(content=jpg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/state")
    def state() -> JSONResponse:
        now = time.time()
        with S.lock:
            stamps = list(S.cap_stamps)
            cap_fps = 0.0
            if len(stamps) >= 2:
                span = stamps[-1] - stamps[0]
                if span > 0:
                    cap_fps = (len(stamps) - 1) / span
            out = {
                "car_ip": args.car_ip,
                "now": now,
                "camera": {
                    "fps": round(cap_fps, 1),
                    "age": round(now - S.frame_t, 2) if S.frame_t else None,
                    "dims": S.frame_dims,
                    "errors": S.cap_errors,
                },
                "depth": {
                    "enabled": not args.no_depth,
                    "scores": S.depth_scores,
                    "ratios": {k: (S.depth_debug["zones"][k]["ratio"] if S.depth_debug else None)
                               for k in ("left", "center", "right")},
                    "ref_disp": S.depth_debug["ref_disp"] if S.depth_debug else None,
                    "ratio_open": S.depth_debug["ratio_open"] if S.depth_debug else None,
                    "ratio_block": S.depth_debug["ratio_block"] if S.depth_debug else None,
                    "latency_ms": round(S.depth_latency * 1000) if S.depth_t else None,
                    "fps": round(1.0 / S.depth_latency, 1) if S.depth_latency else None,
                    "age": round(now - S.depth_t, 2) if S.depth_t else None,
                    "errors": S.depth_errors,
                    "thresholds": THR,
                },
                "vlm": {
                    "enabled": args.vlm != "off",
                    "backend": args.vlm,
                    "model": ((S.vlm_label or os.path.basename(args.local_vlm_model_dir)) if args.vlm == "local"
                              else args.robotics_model if args.vlm == "robotics" else args.model),
                    "mode": args.vlm if args.vlm in ("robotics", "explore") else "seek",
                    "decision": S.vlm_decision,
                    "latency_ms": round(S.vlm_latency * 1000) if S.vlm_t else None,
                    "age": round(now - S.vlm_t, 2) if S.vlm_t else None,
                    "frames": S.vlm_frames,
                    "errors": S.vlm_errors,
                    "history": [
                        {"ago": round(now - h["t"], 1), "line": h["line"], "reason": h["reason"]}
                        for h in list(S.vlm_history)
                    ],
                },
                "task": {
                    "vlm_available": args.vlm != "off",
                    "running": S.autonomy_on,
                    "dry_run": S.auto_dry,
                    "task": S.task,
                    "status": S.auto_status,
                    "action": S.auto_action,
                },
                "robotics": {
                    "active": args.vlm == "robotics",
                    "target_point": S.rb_target_point,
                    "cmd": S.rb_cmd,
                    "measured": S.rb_measured,
                    "calib": S.rb_calib,
                },
                "carpet": {"enabled": not args.no_depth, "scores": S.carpet,
                           "block": args.carpet_block, "edge": args.carpet_edge},
                "explore": {
                    "active": args.vlm == "explore",
                    "directive": S.directive,
                    "directive_reason": S.directive_reason,
                    "directive_age": round(now - S.directive_t, 1) if S.directive_t else None,
                    "target_visible": S.directive_target_visible,
                    "target_close": S.directive_target_close,
                    "target_point": S.directive_target_point,
                    "drift_trim": S.drift_trim,
                    "summary": S.explore.get("summary", ""),
                    "bias": S.explore.get("bias", "center"),
                    "visited": S.explore.get("visited", 0),
                    "looping": S.explore.get("looping", False),
                    "keep_clear": args.keep_clear,
                },
            }
        return JSONResponse(out)

    @app.get("/drive")
    def drive(cmd: str = Query(...), speed: int = 3, ms: int = 400) -> JSONResponse:
        """Manual, user-initiated pulse driving. Moves then auto-stops.
        Any manual command also cancels an active task (manual override)."""
        car = args.car_ip
        with S.lock:
            if S.autonomy_on:
                S.autonomy_on = False
                S.auto_status = "stopped (manual override)"
        speed = max(0, min(8, speed))
        val = {"forward": 1, "back": 2, "left": 3, "right": 4, "stop": 5}.get(cmd)
        if val is None:
            return JSONResponse({"ok": False, "error": f"unknown cmd {cmd}"}, status_code=400)
        if val == 5:
            car_ctl(car, "var=car&val=5")
            return JSONResponse({"ok": True, "cmd": "stop"})
        car_ctl(car, f"var=speed&val={speed}")
        car_ctl(car, f"var=car&val={val}")
        time.sleep(max(0.0, min(2.0, ms / 1000.0)))
        car_ctl(car, "var=car&val=5")
        return JSONResponse({"ok": True, "cmd": cmd, "speed": speed, "ms": ms})

    @app.get("/task/start")
    def task_start(prompt: str = Query(...), dry: int = 0) -> JSONResponse:
        """Start executing a task: the VLM seeks `prompt` and drives the car."""
        if args.vlm == "off":
            return JSONResponse(
                {"ok": False, "error": "VLM is off — relaunch dashboard with --vlm gemini or --vlm local"},
                status_code=409)
        prompt = (prompt or "").strip()
        if not prompt:
            return JSONResponse({"ok": False, "error": "empty task prompt"}, status_code=400)
        with S.lock:
            S.task = prompt
            S.auto_dry = bool(dry)
            S.vlm_decision = None        # drop any stale decision from a previous run
            S.vlm_t = 0.0
            S.vlm_history.clear()
            S.auto_action = ""
            S.auto_status = "starting…"
            S.autonomy_on = True
        return JSONResponse({"ok": True, "task": prompt, "dry_run": bool(dry)})

    @app.get("/task/stop")
    def task_stop() -> JSONResponse:
        with S.lock:
            S.autonomy_on = False
            S.auto_status = "stopped"
            S.auto_action = ""
        car_ctl(args.car_ip, "var=car&val=5")
        return JSONResponse({"ok": True})

    return app


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>car_robot — live dashboard</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#30363d; --fg:#e6edf3; --mut:#8b949e;
          --grn:#3fb950; --yel:#d29922; --red:#f85149; --acc:#58a6ff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  header { display:flex; gap:18px; flex-wrap:wrap; align-items:center;
           padding:10px 16px; border-bottom:1px solid var(--line); background:var(--panel); }
  header b { color:var(--acc); }
  .stat { color:var(--mut); }
  .stat span { color:var(--fg); }
  main { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; padding:12px; }
  @media (max-width:1100px){ main { grid-template-columns:1fr; } }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  .card h2 { margin:0; padding:8px 12px; font-size:13px; border-bottom:1px solid var(--line);
             color:var(--mut); letter-spacing:.5px; text-transform:uppercase; }
  .imgwrap { position:relative; background:#000; aspect-ratio:4/3; display:flex; align-items:center; justify-content:center; }
  .imgwrap img { width:100%; height:100%; object-fit:contain; display:block; }
  .xhair { position:absolute; width:20px; height:20px; margin:-10px 0 0 -10px; border:2px solid var(--acc);
           border-radius:50%; box-shadow:0 0 0 2px rgba(0,0,0,.6); pointer-events:none; display:none; }
  .xhair::before, .xhair::after { content:""; position:absolute; background:var(--acc); }
  .xhair::before { left:50%; top:-7px; width:2px; height:7px; margin-left:-1px; }
  .xhair::after { top:50%; left:-7px; height:2px; width:7px; margin-top:-1px; }
  .rb { margin-top:10px; border-top:1px solid var(--line); padding-top:8px; }
  .body { padding:12px; }
  .bars { display:flex; flex-direction:column; gap:8px; margin-top:4px; }
  .bar { display:grid; grid-template-columns:64px 1fr 52px; align-items:center; gap:8px; }
  .bar .lbl { color:var(--mut); text-transform:uppercase; font-size:12px; }
  .track { height:16px; background:#0d1117; border:1px solid var(--line); border-radius:4px; overflow:hidden; }
  .fill { height:100%; width:0%; transition:width .2s, background .2s; }
  .bar .val { text-align:right; }
  .thr { margin-top:10px; color:var(--mut); font-size:12px; }
  .thr i { font-style:normal; padding:1px 6px; border-radius:3px; }
  .kv { display:grid; grid-template-columns:auto 1fr; gap:4px 12px; }
  .kv .k { color:var(--mut); }
  .kv .v { color:var(--fg); word-break:break-word; }
  .reason { margin-top:10px; padding:8px 10px; background:#0d1117; border-left:3px solid var(--acc);
            border-radius:4px; color:var(--fg); min-height:18px; }
  .action { font-size:20px; font-weight:700; color:var(--acc); }
  .hist { margin-top:12px; border-top:1px solid var(--line); padding-top:8px; }
  .hist .row { display:flex; gap:8px; padding:2px 0; color:var(--mut); font-size:12px; }
  .hist .row .ago { width:46px; color:#6e7681; }
  .hist .row .txt { color:var(--fg); }
  .pill { padding:1px 7px; border-radius:10px; font-size:11px; }
  .ok { background:rgba(63,185,80,.15); color:var(--grn); }
  .warn { background:rgba(210,153,34,.15); color:var(--yel); }
  .err { background:rgba(248,81,73,.15); color:var(--red); }
  footer { padding:10px 16px; border-top:1px solid var(--line); background:var(--panel);
           display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  button { background:#21262d; color:var(--fg); border:1px solid var(--line); border-radius:6px;
           padding:7px 14px; cursor:pointer; font:inherit; }
  button:hover { border-color:var(--acc); }
  button.stop { color:var(--red); border-color:var(--red); }
  footer .muted { color:var(--mut); margin-left:auto; }
  input[type=range]{ accent-color:var(--acc); }
  .taskbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap;
             padding:10px 16px; border-bottom:1px solid var(--line); background:#11161d; }
  .taskbar label.t { color:var(--mut); }
  #task { flex:1; min-width:220px; background:#0d1117; color:var(--fg);
          border:1px solid var(--line); border-radius:6px; padding:7px 10px; font:inherit; }
  #task:focus { outline:none; border-color:var(--acc); }
  button.run { color:var(--grn); border-color:var(--grn); }
  .taskbar .dry { color:var(--mut); display:flex; align-items:center; gap:4px; }
  .taskbar .tstat { margin-left:auto; color:var(--fg); }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#6e7681; margin-right:6px; }
  .dot.on { background:var(--grn); box-shadow:0 0 6px var(--grn); }
</style>
</head>
<body>
<header>
  <b>car_robot</b>
  <span class="stat">car <span id="s-ip">—</span></span>
  <span class="stat">cam <span id="s-cam">—</span></span>
  <span class="stat">depth <span id="s-depth">—</span></span>
  <span class="stat">vlm <span id="s-vlm">—</span></span>
  <span id="s-warn"></span>
</header>

<section class="taskbar">
  <label class="t">Задача:</label>
  <input id="task" type="text" value="__DEFAULT_TASK__"/>
  <button class="run" id="btn-run" onclick="startTask()">▶ Запустить</button>
  <button class="stop" onclick="stopTask()">■ Стоп</button>
  <label class="dry"><input type="checkbox" id="dry"/> dry-run (не двигать)</label>
  <span class="tstat"><span class="dot" id="task-dot"></span><span id="task-status">idle</span> <span id="task-action" style="color:var(--mut)"></span></span>
</section>

<main>
  <div class="card">
    <h2>Camera — what the car sees</h2>
    <div class="imgwrap"><img id="cam" alt="camera"/><div class="xhair" id="xhair"></div></div>
  </div>

  <div class="card">
    <h2>Depth — free space (near = red)</h2>
    <div class="imgwrap"><img id="depth" alt="depth"/></div>
    <div class="body">
      <div class="bars">
        <div class="bar"><span class="lbl">left</span><div class="track"><div class="fill" id="f-left"></div></div><span class="val" id="v-left">—</span></div>
        <div class="bar"><span class="lbl">center</span><div class="track"><div class="fill" id="f-center"></div></div><span class="val" id="v-center">—</span></div>
        <div class="bar"><span class="lbl">right</span><div class="track"><div class="fill" id="f-right"></div></div><span class="val" id="v-right">—</span></div>
      </div>
      <div class="thr">openness 0..1 · <i class="err">&lt;block</i> blocked · <i class="warn">&lt;slow</i> crawl · <i class="ok">&ge;open</i> clear &nbsp;(<span id="thr">…</span>)</div>
      <div class="thr">ratio near/floor: L <span id="r-left">—</span> · C <span id="r-center">—</span> · R <span id="r-right">—</span> · floor <span id="r-ref">—</span> &nbsp;(open&lt;<span id="r-o">—</span>, block&gt;<span id="r-b">—</span>)</div>
      <div class="thr"><i style="color:#2ee6e6">▭ look-ahead band</i> &nbsp;&nbsp; <i style="color:#e62ee6">▭ floor reference</i> — the pixels openness is computed from</div>
      <div class="bars" style="margin-top:10px">
        <div class="bar"><span class="lbl">rug L</span><div class="track"><div class="fill" id="cf-left"></div></div><span class="val" id="cv-left">—</span></div>
        <div class="bar"><span class="lbl">rug C</span><div class="track"><div class="fill" id="cf-center"></div></div><span class="val" id="cv-center">—</span></div>
        <div class="bar"><span class="lbl">rug R</span><div class="track"><div class="fill" id="cf-right"></div></div><span class="val" id="cv-right">—</span></div>
      </div>
      <div class="thr">on-rug confidence (1 = same surface as under the wheels) — used by <b>explore</b> to stay on the rug</div>
    </div>
  </div>

  <div class="card">
    <h2>VLM — decision</h2>
    <div class="body">
      <div id="vlm-head"><span class="action" id="vlm-action">—</span></div>
      <div class="kv" id="vlm-kv"></div>
      <div class="reason" id="vlm-reason">waiting for first decision…</div>
      <div class="rb" id="rb" style="display:none">
        <div class="kv">
          <div class="k">command</div><div class="v" id="rb-cmd">—</div>
          <div class="k">measured</div><div class="v" id="rb-meas">—</div>
          <div class="k">calibration</div><div class="v" id="rb-calib">—</div>
        </div>
      </div>
      <div class="hist" id="vlm-hist"></div>
    </div>
  </div>
</main>

<footer>
  <span class="muted-left">manual drive:</span>
  <button onclick="drive('left')">◀ left</button>
  <button onclick="drive('forward')">▲ forward</button>
  <button onclick="drive('right')">right ▶</button>
  <button onclick="drive('back')">▼ back</button>
  <button class="stop" onclick="drive('stop')">■ STOP</button>
  speed <input type="range" id="speed" min="0" max="8" value="3" oninput="sp.textContent=this.value"/>
  <span id="sp">3</span>
  <span class="muted">pulse <input type="range" id="pulse" min="100" max="1000" step="50" value="400" oninput="pl.textContent=this.value+'ms'"/> <span id="pl">400ms</span></span>
</footer>

<script>
const $ = id => document.getElementById(id);
function band(v, thr){ if(v==null) return ''; if(v<thr.block) return 'var(--red)'; if(v<thr.slow) return 'var(--yel)'; if(v>=thr.open) return 'var(--grn)'; return '#58a6ff'; }
function pill(txt, cls){ return `<span class="pill ${cls}">${txt}</span>`; }

let curThr = {block:0.18, slow:0.36, open:0.55};

function refreshImgs(){
  const t = Date.now();
  $('cam').src = '/frame.jpg?t=' + t;
  $('depth').src = '/depth.jpg?t=' + t;
}

async function poll(){
  try {
    const r = await fetch('/state', {cache:'no-store'});
    const s = await r.json();
    // header
    $('s-ip').textContent = s.car_ip || '—';
    const cam = s.camera;
    $('s-cam').textContent = `${cam.fps||0}fps ${cam.dims[0]}×${cam.dims[1]} age ${cam.age==null?'—':cam.age+'s'}`
      + (cam.errors? ` ✕${cam.errors}`:'');
    const d = s.depth;
    $('s-depth').textContent = d.enabled ? `${d.fps==null?'—':d.fps+'fps'} ${d.latency_ms==null?'':d.latency_ms+'ms'} age ${d.age==null?'—':d.age+'s'}` : 'off';
    const v = s.vlm;
    $('s-vlm').textContent = v.enabled ? `${v.model} ${v.latency_ms==null?'—':v.latency_ms+'ms'} age ${v.age==null?'—':v.age+'s'}${v.errors?` ✕${v.errors}`:''}` : 'off';

    let warn = '';
    if (cam.age!=null && cam.age>2) warn += pill('stale camera','err');
    if (d.enabled && d.age!=null && d.age>3) warn += pill('stale depth','warn');
    if (v.enabled && v.errors>0) warn += pill('vlm errors','warn');
    $('s-warn').innerHTML = warn;

    // depth bars
    curThr = d.thresholds || curThr;
    $('thr').textContent = `block ${curThr.block} · slow ${curThr.slow} · open ${curThr.open}`;
    for (const z of ['left','center','right']){
      const val = d.scores ? d.scores[z] : null;
      const f = $('f-'+z), lab = $('v-'+z);
      if (val==null){ f.style.width='0%'; lab.textContent='—'; }
      else { f.style.width = Math.round(val*100)+'%'; f.style.background = band(val, curThr); lab.textContent = val.toFixed(2); }
      const rat = d.ratios ? d.ratios[z] : null;
      $('r-'+z).textContent = rat==null ? '—' : rat.toFixed(2);
    }
    $('r-ref').textContent = d.ref_disp==null ? '—' : d.ref_disp;
    $('r-o').textContent = d.ratio_open==null ? '—' : d.ratio_open;
    $('r-b').textContent = d.ratio_block==null ? '—' : d.ratio_block;

    // carpet / on-rug bars
    const cpt = s.carpet || {};
    for (const z of ['left','center','right']){
      const val = cpt.scores ? cpt.scores[z] : null;
      const f = $('cf-'+z), lab = $('cv-'+z);
      if (val==null){ f.style.width='0%'; lab.textContent='—'; }
      else {
        f.style.width = Math.round(val*100)+'%';
        f.style.background = val<cpt.block ? 'var(--red)' : (val<cpt.edge ? 'var(--yel)' : 'var(--grn)');
        lab.textContent = val.toFixed(2);
      }
    }

    // vlm panel
    const kv = $('vlm-kv');
    if (s.vlm.mode === 'explore'){
      const ex = s.explore || {};
      const tgt = ex.target_close ? ' · TARGET REACHED' : (ex.target_visible ? ' · target ahead' : '');
      $('vlm-action').textContent = (ex.directive || 'explore').toUpperCase() + tgt;
      const dft = (ex.drift_trim!=null) ? ` · drift ${ex.drift_trim>=0?'+':''}${ex.drift_trim}` : '';
      kv.innerHTML = `<div class="k">map</div><div class="v">${ex.summary||'—'}</div>`
        + `<div class="k">state</div><div class="v">bias=${ex.bias} visited=${ex.visited}${ex.looping?' · LOOPING':''}${dft}</div>`;
      $('vlm-reason').textContent = ex.directive_reason || '';
    } else {
      const dec = v.decision;
      if (!dec){ $('vlm-action').textContent = v.enabled?'…':'off'; kv.innerHTML=''; }
      else {
        $('vlm-action').textContent = dec.target_visible
          ? `TARGET ${(''+(dec.target_bearing||dec.target_size||'')).toUpperCase()}` : 'SEARCHING';
        const order = ['target_visible','target_bearing','target_size','target_motion','turn_strength','search_strategy','safe_forward','arrived'];
        kv.innerHTML = order.filter(k=>k in dec).map(k=>`<div class="k">${k}</div><div class="v">${dec[k]}</div>`).join('');
        $('vlm-reason').textContent = dec.reason || '';
      }
    }

    // history
    $('vlm-hist').innerHTML = (v.history||[]).map(h =>
      `<div class="row"><span class="ago">${h.ago}s</span><span class="txt">${h.line} — <span style="color:var(--mut)">${h.reason}</span></span></div>`
    ).join('');

    // task / autonomy
    const tk = s.task || {};
    $('task-dot').className = 'dot' + (tk.running ? ' on' : '');
    $('btn-run').textContent = tk.running ? '▶ выполняется…' : '▶ Запустить';
    $('task-status').textContent = (tk.status || (tk.running?'running':'idle')) + (tk.dry_run ? ' [dry-run]' : '');
    $('task-action').textContent = tk.action ? ('· ' + tk.action) : '';
    if (!tk.vlm_available) $('task-status').textContent = 'VLM off — перезапусти с --vlm gemini/local/robotics';

    // target crosshair (robotics command point, or explore steering point)
    const rb = s.robotics || {};
    const exb0 = s.explore || {};
    const tpoint = (rb.active && rb.target_point) ? rb.target_point
                 : (exb0.active ? exb0.target_point : null);
    const xh = $('xhair');
    if (tpoint && tpoint.length === 2){
      xh.style.display = 'block';
      xh.style.top  = (tpoint[0] / 10) + '%';   // y 0-1000 -> %
      xh.style.left = (tpoint[1] / 10) + '%';   // x 0-1000 -> %
    } else { xh.style.display = 'none'; }
    const rbBox = $('rb');
    if (rb.active){
      rbBox.style.display = 'block';
      const c = rb.cmd, m = rb.measured, cal = rb.calib;
      $('rb-cmd').textContent  = c ? `${c.action} deg=${c.turn_degrees} ms=${c.duration_ms} spd=${c.speed} trim=${c.trim>=0?'+':''}${c.trim}` : '—';
      $('rb-meas').textContent = m ? `yaw=${m.yaw_deg}° img_change=${m.img_change==null?'—':m.img_change}${m.stall?' · STALL':''}` : '—';
      $('rb-calib').textContent= (cal && cal.ms_per_deg!=null) ? `${cal.ms_per_deg} ms/°` : '—';
    } else { rbBox.style.display = 'none'; }
  } catch(e){ /* transient */ }
}

async function drive(cmd){
  const speed = $('speed').value, ms = $('pulse').value;
  try { await fetch(`/drive?cmd=${cmd}&speed=${speed}&ms=${ms}`); } catch(e){}
}

async function startTask(){
  const prompt = $('task').value.trim();
  if(!prompt){ alert('Введите задачу'); return; }
  const dry = $('dry').checked ? 1 : 0;
  try {
    const r = await fetch(`/task/start?prompt=${encodeURIComponent(prompt)}&dry=${dry}`);
    const j = await r.json();
    if(!j.ok) alert('Не удалось запустить: ' + (j.error||''));
  } catch(e){ alert('Ошибка сети'); }
}
async function stopTask(){ try { await fetch('/task/stop'); } catch(e){} }

document.addEventListener('keydown', e=>{
  const el = document.activeElement;
  if (el && (el.tagName==='INPUT' || el.tagName==='TEXTAREA')) return;  // don't drive while typing the task
  const m = {ArrowUp:'forward',ArrowDown:'back',ArrowLeft:'left',ArrowRight:'right',' ':'stop'};
  if (m[e.key]){ e.preventDefault(); drive(m[e.key]); }
});

setInterval(refreshImgs, 150);
setInterval(poll, 300);
refreshImgs(); poll();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Live observability dashboard for the ESP32-CAM car")
    ap.add_argument("target", nargs="?", help="optional target -> richer seek-style VLM output")
    ap.add_argument("--target", dest="target_opt", help="same as positional target (overrides it)")
    ap.add_argument("--env", default=os.path.join(REPO_ROOT, ".env"))
    ap.add_argument("--car-ip", help="ESP32-CAM IP; defaults to CAR_IP from env")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)

    ap.add_argument("--vlm", choices=["gemini", "local", "robotics", "explore", "off"], default="gemini",
                    help="task backend: gemini/local seek, robotics (ER-1.6 closed-loop), "
                         "explore (local roaming + Gemini coarse directives), or off")
    ap.add_argument("--no-vlm", action="store_const", const="off", dest="vlm",
                    help="alias for --vlm off (camera + depth only)")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model (when --vlm gemini)")
    ap.add_argument("--robotics-model", default="gemini-robotics-er-1.6-preview",
                    help="Gemini Robotics-ER model for --vlm robotics")
    ap.add_argument("--local-vlm-model-dir",
                    default=os.path.join(REPO_ROOT, "models", "fastvlm-0.5b-bf16"),
                    help="mlx-vlm model dir for --vlm local (FastVLM / Qwen2.5-VL)")
    ap.add_argument("--vlm-sec", type=float, default=2.0, help="seconds between VLM calls")
    ap.add_argument("--vlm-frames", type=int, default=2,
                    help="frames per call (motion cue; clamped to 1 for FastVLM/LLaVA)")
    ap.add_argument("--vlm-frame-space", type=float, default=0.6)

    ap.add_argument("--no-depth", action="store_true", help="skip the local depth model")
    ap.add_argument("--depth-model", default=os.path.join(REPO_ROOT, "models", "da2-small"))
    ap.add_argument("--depth-size", type=int, default=252)
    ap.add_argument("--depth-hz", type=float, default=5.0)
    ap.add_argument("--capture-hz", type=float, default=8.0)
    ap.add_argument("--capture-timeout", type=float, default=4.0)

    # thresholds mirrored from reflex_drive.py so the bars read the same as the controller
    ap.add_argument("--block", type=float, default=0.18)
    ap.add_argument("--slow", type=float, default=0.36)
    ap.add_argument("--open", type=float, default=0.55)

    # autonomy / task execution (Start button) — defaults mirror seek.py
    ap.add_argument("--drive-speed", type=int, default=2, help="forward speed 0..8 while executing a task")
    ap.add_argument("--trim-slight", type=int, default=12)
    ap.add_argument("--trim-medium", type=int, default=24)
    ap.add_argument("--trim-hard", type=int, default=40)
    ap.add_argument("--bump", type=float, default=0.12,
                    help="depth center-openness below this = emergency bumper (something within ~25cm)")
    ap.add_argument("--turn-pulse", type=float, default=0.12, help="spin seconds per steering tick")
    ap.add_argument("--search-pulse", type=float, default=0.16, help="spin seconds per search tick")
    ap.add_argument("--scan-max", type=int, default=3, help="scan-turns before a forced forward relocate")
    ap.add_argument("--arrive-medium-frames", type=int, default=3,
                    help="stop after this many consecutive medium+centered readings (Gemini rarely says 'large')")
    ap.add_argument("--drive-hz", type=float, default=5.0, help="autonomy control-loop rate")
    ap.add_argument("--goal-stale-sec", type=float, default=8.0,
                    help="hold (stop) if the latest VLM decision is older than this")

    # closed-loop robotics mode (--vlm robotics); parametric pulses + measured feedback
    ap.add_argument("--ms-per-90", type=int, default=900, help="initial turn calibration (ms per 90deg)")
    ap.add_argument("--turnspeed", type=int, default=75, help="firmware in-place turn power 0..255")
    ap.add_argument("--settle-sec", type=float, default=0.25, help="pause after a pulse before measuring")
    ap.add_argument("--cam-hfov", type=float, default=60.0, help="camera horizontal FOV (deg) for yaw estimate")
    ap.add_argument("--stuck-change", type=float, default=0.015, help="image-change below this after forward = stall")
    ap.add_argument("--rb-max-speed", type=int, default=3)
    ap.add_argument("--rb-escape-speed", type=int, default=3)
    ap.add_argument("--rb-max-trim", type=int, default=18)
    ap.add_argument("--rb-near-trim", type=int, default=7)
    ap.add_argument("--rb-default-ms", type=int, default=350)
    ap.add_argument("--rb-min-ms", type=int, default=150)
    ap.add_argument("--rb-max-ms", type=int, default=650)
    ap.add_argument("--rb-max-back-ms", type=int, default=500)
    ap.add_argument("--rb-default-deg", type=int, default=18)
    ap.add_argument("--rb-max-deg", type=int, default=35)
    ap.add_argument("--rb-min-turn-ms", type=int, default=120)
    ap.add_argument("--rb-max-turn-ms", type=int, default=450)

    # local-AI explore mode (--vlm explore): roam, stay on rug, keep clear of objects
    ap.add_argument("--keep-clear", type=float, default=0.42,
                    help="center openness below this = object getting close -> slow + steer away early")
    ap.add_argument("--carpet-block", type=float, default=0.30,
                    help="on-rug confidence below this ahead = rug edge/off-rug -> treat as blocked")
    ap.add_argument("--carpet-edge", type=float, default=0.55,
                    help="on-rug confidence below this ahead = approaching edge -> slow + steer back")
    ap.add_argument("--side-open", type=float, default=0.30, help="side drivable needed to avoid an escape")
    ap.add_argument("--open-gain", type=float, default=1.2, help="steer gain from left/right drivable difference")
    ap.add_argument("--map-weight", type=float, default=0.6, help="how strongly the frontier map biases steering")
    ap.add_argument("--max-trim", type=int, default=24, help="max steering trim while exploring")
    ap.add_argument("--min-drive-speed", type=int, default=2)
    ap.add_argument("--escape-speed", type=int, default=3)
    ap.add_argument("--search-speed", type=int, default=1, help="speed cap under a 'search_here' directive")
    ap.add_argument("--target-approach-speed", type=int, default=1,
                    help="crawl speed once the supervisor confirms the target is ahead")
    ap.add_argument("--arrive-open", type=float, default=0.22,
                    help="when the target is confirmed ahead, stop once center openness drops below this")
    ap.add_argument("--target-memory-sec", type=float, default=6.0,
                    help="how long a 'target ahead' confirmation stays valid for the local stop")
    ap.add_argument("--approach-gain", type=float, default=0.9,
                    help="steering gain from the target's horizontal offset (ER point) during approach")
    ap.add_argument("--approach-vlm-sec", type=float, default=1.6,
                    help="faster supervisor interval while approaching a visible target (fresh steering point)")
    ap.add_argument("--drift-gain", type=float, default=1.0,
                    help="how aggressively to learn the constant drift-correction trim (0 disables)")
    ap.add_argument("--max-drift-trim", type=float, default=10.0,
                    help="cap on the learned drift-correction trim")
    ap.add_argument("--back-secs", type=float, default=0.35)
    ap.add_argument("--escape-turn-sec", type=float, default=0.38)
    ap.add_argument("--stale-view-sec", type=float, default=1.5)
    ap.add_argument("--turn-around-sec", type=float, default=1.0, help="spin time for a turn_around directive")
    ap.add_argument("--turn-around-cooldown", type=float, default=6.0)
    ap.add_argument("--frontier-bonus", type=float, default=1.2)
    ap.add_argument("--revisit-penalty", type=float, default=1.0)
    ap.add_argument("--blocked-penalty", type=float, default=2.5)
    ap.add_argument("--block-memory-sec", type=float, default=45.0)
    ap.add_argument("--commit-switch-margin", type=float, default=0.35)
    ap.add_argument("--explore-commit-sec", type=float, default=4.5)
    ap.add_argument("--loop-no-new-cell-sec", type=float, default=12.0)
    ap.add_argument("--loop-cell-entries", type=int, default=4)
    ap.add_argument("--map-cell-size", type=float, default=1.0)
    args = ap.parse_args()

    load_dotenv(args.env)
    args.target = args.target_opt if args.target_opt is not None else args.target
    args.car_ip = args.car_ip or os.environ.get("CAR_IP")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not args.car_ip:
        ap.error("set CAR_IP in .env or pass --car-ip")
    if args.vlm in ("gemini", "robotics", "explore") and not key:
        print(f"[warn] no GEMINI_API_KEY/GOOGLE_API_KEY -> VLM off (use --vlm local for FastVLM)")
        args.vlm = "off"
    if args.vlm == "explore" and args.no_depth:
        print("[warn] --vlm explore needs depth (obstacle distance + carpet); enabling depth")
        args.no_depth = False

    vlm_desc = {"local": f"local:{os.path.basename(args.local_vlm_model_dir)}",
                "gemini": f"{args.model} (seek)",
                "robotics": f"{args.robotics_model} (closed-loop)",
                "explore": f"{args.model} (explore: local roam + coarse directives)",
                "off": "off"}[args.vlm]
    print(f"dashboard | car={args.car_ip} | depth={'off' if args.no_depth else args.depth_model} | "
          f"vlm={vlm_desc} | tasks={'enabled' if args.vlm != 'off' else 'DISABLED (vlm off)'}")
    print(f"open http://{args.host}:{args.port}  —  Ctrl-C to stop")

    threading.Thread(target=capture_loop,
                     args=(args.car_ip, args.capture_hz, args.capture_timeout), daemon=True).start()
    if not args.no_depth:
        threading.Thread(target=depth_loop,
                         args=(args.depth_model, args.depth_size, args.depth_hz), daemon=True).start()
    if args.vlm == "robotics":
        threading.Thread(target=closedloop_loop, args=(args, key), daemon=True).start()
    elif args.vlm == "explore":
        threading.Thread(target=explore_loop, args=(args,), daemon=True).start()
        threading.Thread(target=directive_loop, args=(args, key), daemon=True).start()
    elif args.vlm != "off":
        threading.Thread(target=vlm_loop, args=(args, key), daemon=True).start()
        threading.Thread(target=autonomy_loop, args=(args,), daemon=True).start()

    app = make_app(args)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        S.running = False
        # leave the car stopped if we ever sent a manual command
        try:
            car_ctl(args.car_ip, "var=car&val=5")
        except Exception:
            pass
        print("\nstopped.")


if __name__ == "__main__":
    main()
