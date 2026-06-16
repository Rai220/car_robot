#!/usr/bin/env python3
"""
Gemini camera-only pulse controller for the ESP32-CAM car.

No local depth scanner is used. Gemini receives recent camera frames plus action
history and chooses one bounded motor pulse. The script sends STOP after every
pulse before capturing the next frame.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from io import BytesIO
from typing import Deque, Optional

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

from PIL import Image


ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["forward", "back", "turn_left", "turn_right", "stop"]},
        "speed": {"type": "integer"},
        "duration_ms": {"type": "integer"},
        "trim": {"type": "integer"},
        "turn_degrees": {"type": "integer"},
        "target_visible": {"type": "boolean"},
        "target_bearing": {
            "type": "string",
            "enum": ["far_left", "left", "center", "right", "far_right", "none"],
        },
        "target_size": {"type": "string", "enum": ["none", "small", "medium", "large"]},
        "arrived": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "expected_effect": {"type": "string"},
    },
    "required": [
        "action", "speed", "duration_ms", "trim", "turn_degrees",
        "target_visible", "target_bearing", "target_size", "arrived",
        "confidence", "reason", "expected_effect",
    ],
}


def load_dotenv(path: str) -> None:
    if not os.path.exists(path):
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


def parse_json_object(text: str) -> Optional[dict[str, object]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def capture_jpeg(car_ip: str, timeout: float = 5.0) -> bytes:
    jpg = urllib.request.urlopen(f"http://{car_ip}/capture", timeout=timeout).read()
    if not jpg.startswith(b"\xff\xd8"):
        raise ValueError("capture did not return JPEG")
    return jpg


def wait_camera(car_ip: str, wait_sec: float, timeout: float) -> bytes:
    deadline = time.time() + wait_sec
    attempt = 0
    last: Optional[BaseException] = None
    while time.time() < deadline:
        attempt += 1
        try:
            jpg = capture_jpeg(car_ip, timeout)
            print(f"[camera] ready after {attempt} attempt(s), {len(jpg)} bytes")
            return jpg
        except Exception as e:
            last = e
            print(f"[camera] waiting ({attempt}): {e}")
            time.sleep(2.0)
    raise RuntimeError(f"camera did not become ready: {last}")


def image_change(prev: Optional[bytes], cur: Optional[bytes]) -> Optional[float]:
    if not prev or not cur:
        return None
    try:
        a = Image.open(BytesIO(prev)).convert("L").resize((32, 24))
        b = Image.open(BytesIO(cur)).convert("L").resize((32, 24))
        pa = a.tobytes()
        pb = b.tobytes()
        return sum(abs(x - y) for x, y in zip(pa, pb)) / (255.0 * len(pa))
    except Exception:
        return None


class SimpleMap:
    def __init__(self, cell_size: float) -> None:
        self.cell_size = cell_size
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.visited: dict[tuple[int, int], int] = {}
        self.last_target: Optional[tuple[int, int, str, str]] = None

    def cell(self) -> tuple[int, int]:
        return (int(round(self.x / self.cell_size)), int(round(self.y / self.cell_size)))

    def update(self, dec: dict[str, object]) -> None:
        action = str(dec.get("action", "stop"))
        duration = int(dec.get("duration_ms", 0) or 0) / 1000.0
        speed = int(dec.get("speed", 0) or 0)
        trim = int(dec.get("trim", 0) or 0)
        degrees = int(dec.get("turn_degrees", 0) or 0)
        if action == "forward":
            dist = duration * (0.35 + 0.12 * max(1, speed))
            self.x += dist * math.cos(self.heading)
            self.y += dist * math.sin(self.heading)
            self.heading += duration * trim * 0.015
        elif action == "back":
            dist = duration * (0.30 + 0.10 * max(1, speed))
            self.x -= dist * math.cos(self.heading)
            self.y -= dist * math.sin(self.heading)
        elif action == "turn_left":
            self.heading -= math.radians(max(8, abs(degrees)))
        elif action == "turn_right":
            self.heading += math.radians(max(8, abs(degrees)))
        c = self.cell()
        self.visited[c] = self.visited.get(c, 0) + 1
        if dec.get("target_visible"):
            self.last_target = (c[0], c[1], str(dec.get("target_bearing")), str(dec.get("target_size")))

    def summary(self) -> str:
        c = self.cell()
        target = "none" if self.last_target is None else (
            f"cell=({self.last_target[0]},{self.last_target[1]}) "
            f"bearing={self.last_target[2]} size={self.last_target[3]}"
        )
        return f"pose_cell={c}; visited_cells={len(self.visited)}; last_target={target}"


class Car:
    def __init__(self, ip: str, dry_run: bool) -> None:
        self.base = f"http://{ip}"
        self.dry_run = dry_run

    def ctl(self, query: str, timeout: float = 3.0) -> bool:
        if self.dry_run:
            return True
        try:
            urllib.request.urlopen(f"{self.base}/control?{query}", timeout=timeout).read()
            return True
        except Exception as e:
            print(f"[ctl] {query}: {e}")
            return False

    def stop(self) -> None:
        self.ctl("var=car&val=5")

    def set_turnspeed(self, turnspeed: int) -> None:
        self.ctl(f"var=turnspeed&val={int(clamp(turnspeed, 0, 255))}")

    def pulse(self, action: str, speed: int, trim: int, duration_ms: int) -> None:
        secs = max(0.0, duration_ms / 1000.0)
        if action == "stop" or duration_ms <= 0:
            self.stop()
            return
        if action == "forward":
            self.ctl(f"var=speed&val={int(clamp(speed, 0, 8))}")
            self.ctl(f"var=trim&val={int(clamp(trim, -32, 32))}")
            self.ctl("var=car&val=1")
        elif action == "back":
            self.ctl(f"var=speed&val={int(clamp(speed, 0, 8))}")
            self.ctl("var=trim&val=0")
            self.ctl("var=car&val=2")
        elif action == "turn_left":
            self.ctl("var=car&val=3")
        elif action == "turn_right":
            self.ctl("var=car&val=4")
        else:
            self.stop()
            return
        time.sleep(secs)
        self.stop()


def same_action_streak(actions: Deque[str], action: str) -> int:
    n = 0
    for old in actions:
        if old != action:
            break
        n += 1
    return n


def steps_since_forward(actions: Deque[str]) -> int:
    n = 0
    for old in actions:
        if old == "forward":
            break
        n += 1
    return n


def history_line(step: int, dec: dict[str, object], delta: Optional[float]) -> str:
    changed = "unknown" if delta is None else f"{delta:.3f}"
    visible = "visible" if dec.get("target_visible") else "lost"
    return (
        f"- step {step}: executed {dec['action']} speed={dec['speed']} "
        f"trim={dec['trim']} turn_degrees={dec['turn_degrees']} duration_ms={dec['duration_ms']}; "
        f"target={visible} bearing={dec.get('target_bearing')} size={dec.get('target_size')}; "
        f"image_change={changed}; reason={str(dec.get('reason', ''))[:90]}"
    )


def turn_ms(degrees: int, args: argparse.Namespace) -> int:
    return int(clamp(degrees * args.ms_per_90 / 90.0, args.min_turn_ms, args.max_turn_ms))


def force_recovery(dec: dict[str, object], args: argparse.Namespace, action: str, reason: str,
                   *, degrees: Optional[int] = None, duration_ms: Optional[int] = None) -> dict[str, object]:
    guarded = dict(dec)
    guarded["action"] = action
    guarded["trim"] = 0
    if action == "back":
        guarded["speed"] = args.escape_speed
        guarded["turn_degrees"] = 0
        guarded["duration_ms"] = duration_ms if duration_ms is not None else min(args.default_ms, args.max_back_ms)
    elif action in {"turn_left", "turn_right"}:
        deg = degrees if degrees is not None else args.reacquire_turn_degrees
        guarded["speed"] = 0
        guarded["turn_degrees"] = deg
        guarded["duration_ms"] = duration_ms if duration_ms is not None else turn_ms(deg, args)
    elif action == "forward":
        guarded["speed"] = args.min_speed
        guarded["turn_degrees"] = 0
        guarded["duration_ms"] = duration_ms if duration_ms is not None else args.default_ms
    else:
        guarded["speed"] = 0
        guarded["turn_degrees"] = 0
        guarded["duration_ms"] = 0
    guarded["reason"] = f"{guarded['reason']} | local guard: {reason}"[:160]
    return guarded


def force_arrived(dec: dict[str, object], reason: str) -> dict[str, object]:
    guarded = dict(dec)
    guarded["action"] = "stop"
    guarded["speed"] = 0
    guarded["trim"] = 0
    guarded["turn_degrees"] = 0
    guarded["duration_ms"] = 0
    guarded["arrived"] = True
    guarded["confidence"] = max(float(guarded.get("confidence", 0.0) or 0.0), 0.75)
    guarded["reason"] = f"{guarded['reason']} | local arrival guard: {reason}"[:160]
    return guarded


def side_from_bearing(bearing: str, step: int) -> str:
    if bearing in {"far_left", "left"}:
        return "turn_left"
    if bearing in {"far_right", "right"}:
        return "turn_right"
    return "turn_left" if step % 2 else "turn_right"


def control_note(actions: Deque[str], last_change: Optional[float], lost_streak: int,
                 smap: SimpleMap, args: argparse.Namespace) -> str:
    repeated = actions[0] if actions else "none"
    streak = same_action_streak(actions, repeated) if actions else 0
    changed = "unknown" if last_change is None else f"{last_change:.3f}"
    warnings = []
    if last_change is not None and last_change < args.stuck_change:
        warnings.append("LOW_MOTION: last short pulse barely changed the image; do not infer a wall from this alone")
    if lost_streak >= 2 and repeated == "forward" and streak >= 2:
        warnings.append("LOST_TARGET_LOOP: stop repeating forward; scan gently")
    warning = "; ".join(warnings) if warnings else "none"
    return (
        f"- CONTROL STATE: {smap.summary()}; last_image_change={changed}; "
        f"lost_target_steps={lost_streak}; same_action={repeated}x{streak}; warning={warning}"
    )


def build_prompt(target: str, hist_lines: list[str], args: argparse.Namespace) -> str:
    mem = "\n".join(hist_lines) if hist_lines else "none yet"
    return (
        "You are directly driving a small ESP32-CAM robot car from its forward camera. "
        "You receive recent frames, oldest first; the final image is current. Choose "
        "exactly ONE short motor command. The program will execute it, send STOP, capture "
        "a new frame, and ask again.\n"
        f'TASK: explore the room, find this target, and stop close to it: "{target}".\n'
        "There is NO depth scanner. Use visual evidence, action history, and image-change "
        "feedback. Low image-change is weak traction/short-pulse feedback only: DO NOT infer "
        "a wall or obstacle from it unless the camera visibly shows contact or a blocked path. "
        "Dynamically adapt: if a command makes a medium/large target disappear after approach, "
        "treat that as likely arrived/too close; if you keep revisiting the same area, choose "
        "a different scan/relocation direction. If the target is not visible, scan only briefly, "
        "then drive forward to a new viewpoint; do not keep turning in place.\n\n"
        "Target matching is STRICT:\n"
        " - The target is specifically a small yellow rubber bath duck (toy duckie): a rounded yellow duck "
        "body with a small orange/yellow bill and painted eyes, the classic bathtub rubber duck.\n"
        " - Do NOT mark other yellow toys/objects as the target. A yellow ball, block, plush toy, or any "
        "yellow car / construction vehicle without a duck shape is a distractor.\n"
        " - If the duck is large and partly cropped at the left/right/lower edge, "
        "set target_visible=true and arrived=true; do not rotate just to center it.\n\n"
        "Movement model:\n"
        " - forward/back: speed + trim + duration_ms; trim < 0 arcs left, trim > 0 arcs right.\n"
        " - turn_left/turn_right: in-place turn; use turn_degrees for the intended angle.\n"
        " - stop: duration_ms=0. Use stop when arrived, or when a visible obstacle physically fills the path.\n\n"
        "Hard limits enforced by software:\n"
        f" - speed 0..{args.max_speed}; forward duration {args.min_pulse_ms}..{args.max_duration_ms} ms.\n"
        f" - turn_degrees {args.min_turn_degrees}..{args.max_turn_degrees}; "
        f"turn pulse <= {args.max_turn_ms} ms.\n"
        f" - trim -{args.max_trim}..{args.max_trim}; near target trim is capped tighter.\n\n"
        "Policy:\n"
        " - Open carpet/floor in the lower center is drivable; do not describe it as a wall.\n"
        " - Do not claim obstacle/wall unless it is visually obvious in the current image.\n"
        " - The target may be partly cropped near an image edge; if so, use a tiny scan/arc, not a hard spin.\n"
        " - If target is centered and small/medium: short forward, usually trim near 0; after sustained approach, stop.\n"
        " - If target is left/right: use a small turn or forward arc, not a hard spin.\n"
        " - If target is lost: use at most a few small scans, then move forward to a new viewpoint.\n"
        " - If history warns LOW_MOTION or LOST_TARGET_LOOP: prefer a new viewpoint, not reverse.\n"
        " - Avoid back/reverse unless there is visible contact, a visible immediate collision, or the local controller explicitly allows it.\n"
        " - arrived=true only if the target is large/close in the lower center, or contact is imminent.\n"
        " - Avoid furniture, walls, chair legs, cables, and dark narrow gaps.\n\n"
        f"Recent action history (newest first):\n{mem}\n\n"
        "Reply with ONLY JSON using the required schema. Keep reason and expected_effect short."
    )


def ask_gemini_action(model: str, key: str, target: str, frames: list[tuple[float, bytes]],
                      hist_lines: list[str], args: argparse.Namespace) -> dict[str, object]:
    now = frames[-1][0] if frames else time.time()
    parts: list[dict[str, object]] = [{"text": build_prompt(target, hist_lines, args)}]
    for i, (t, jpg) in enumerate(frames):
        tag = "CURRENT" if i == len(frames) - 1 else f"{now - t:.1f}s ago"
        parts.append({"text": f"Frame {i + 1}/{len(frames)} ({tag}):"})
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(jpg).decode("ascii"),
            }
        })
    gen: dict[str, object] = {
        "responseMimeType": "application/json",
        "responseSchema": ACTION_SCHEMA,
        "temperature": args.temperature,
    }
    if not model.startswith("gemini-2.0"):
        gen["thinkingConfig"] = {"thinkingBudget": args.thinking_budget}
    body = {"contents": [{"parts": parts}], "generationConfig": gen}
    data = json.dumps(body).encode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    last: Optional[BaseException] = None
    for _ in range(args.retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            doc = json.load(urllib.request.urlopen(req, timeout=args.timeout))
            text = doc["candidates"][0]["content"]["parts"][0]["text"]
            raw = parse_json_object(text)
            if raw is None:
                raise ValueError(f"Gemini did not return JSON: {text[:200]}")
            return raw
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            last = e
            time.sleep(1.0)
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise RuntimeError(f"Gemini action failed: {last}")


def enum(value: object, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def as_int(value: object, default: int) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def clamp_decision(raw: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    action = enum(raw.get("action"), {"forward", "back", "turn_left", "turn_right", "stop"}, "stop")
    visible = bool(raw.get("target_visible", False))
    bearing = enum(raw.get("target_bearing"), {"far_left", "left", "center", "right", "far_right", "none"}, "none")
    size = enum(raw.get("target_size"), {"none", "small", "medium", "large"}, "none")
    arrived = bool(raw.get("arrived", False))
    if visible and size == "large" and bearing in {"center", "left", "right"}:
        arrived = True
    if arrived:
        action = "stop"

    speed = int(clamp(as_int(raw.get("speed"), args.speed), 0, args.max_speed))
    trim = int(clamp(as_int(raw.get("trim"), 0), -args.max_trim, args.max_trim))
    degrees = int(clamp(abs(as_int(raw.get("turn_degrees"), args.default_turn_degrees)),
                        args.min_turn_degrees, args.max_turn_degrees))
    duration = as_int(raw.get("duration_ms"), args.default_ms)

    if action == "stop":
        speed = 0
        trim = 0
        degrees = 0
        duration = 0
    elif action in {"turn_left", "turn_right"}:
        speed = 0
        trim = 0
        if raw.get("turn_degrees") is not None:
            duration = turn_ms(degrees, args)
        duration = int(clamp(duration, args.min_turn_ms, args.max_turn_ms))
    elif action == "back":
        speed = max(args.min_speed, min(args.escape_speed, speed or args.escape_speed))
        trim = 0
        degrees = 0
        duration = int(clamp(duration, args.min_pulse_ms, args.max_back_ms))
    else:
        speed = max(args.min_speed, speed or args.speed)
        duration = int(clamp(duration, args.min_pulse_ms, args.max_duration_ms))
        degrees = 0
        if visible and size in {"medium", "large"}:
            trim = int(clamp(trim, -args.near_target_trim, args.near_target_trim))

    return {
        "action": action,
        "speed": speed,
        "duration_ms": duration,
        "trim": trim,
        "turn_degrees": degrees,
        "target_visible": visible,
        "target_bearing": bearing,
        "target_size": size,
        "arrived": arrived,
        "confidence": clamp(as_float(raw.get("confidence"), 0.0), 0.0, 1.0),
        "reason": str(raw.get("reason", ""))[:160],
        "expected_effect": str(raw.get("expected_effect", ""))[:160],
    }


def apply_history_guard(dec: dict[str, object], args: argparse.Namespace, actions: Deque[str],
                        last_change: Optional[float], step: int) -> dict[str, object]:
    if not args.repeat_guard or dec["target_visible"] or dec["arrived"]:
        return dec
    repeated_forward = dec["action"] == "forward" and same_action_streak(actions, "forward") >= args.max_forward_repeat
    if repeated_forward:
        side = "turn_left" if step % 2 else "turn_right"
        return force_recovery(dec, args, side, "forward-repeat guard: scan before another forward",
                              degrees=args.guard_turn_degrees)
    if dec["action"] in {"turn_left", "turn_right"} and same_action_streak(actions, str(dec["action"])) >= args.max_turn_repeat:
        side = "turn_right" if dec["action"] == "turn_left" else "turn_left"
        return force_recovery(dec, args, side, "alternate scan side after repeated turns",
                              degrees=args.guard_turn_degrees)
    if dec["action"] == "stop" and same_action_streak(actions, "stop") >= args.max_stop_repeat:
        side = "turn_left" if step % 2 else "turn_right"
        return force_recovery(dec, args, side, "small scan after repeated stop",
                              degrees=args.guard_turn_degrees)
    return dec


def apply_search_guard(dec: dict[str, object], args: argparse.Namespace, actions: Deque[str],
                       step: int, last_seen_step: int, relocate_remaining: int) -> tuple[dict[str, object], int]:
    if dec["target_visible"] or dec["arrived"]:
        return dec, 0
    if last_seen_step > 0 and step - last_seen_step <= args.target_memory_steps:
        return dec, 0
    if relocate_remaining > 0:
        return (
            force_recovery(
                dec,
                args,
                "forward",
                "continue relocation burst to reach a new viewpoint",
                duration_ms=args.explore_forward_ms,
            ),
            relocate_remaining - 1,
        )
    if dec["action"] in {"turn_left", "turn_right", "stop"} and steps_since_forward(actions) >= args.search_turns_before_forward:
        return (
            force_recovery(
                dec,
                args,
                "forward",
                "scan-only loop detected; relocate to a new viewpoint",
                duration_ms=args.explore_forward_ms,
            ),
            max(0, args.search_forward_burst - 1),
        )
    return dec, relocate_remaining


def apply_reverse_guard(dec: dict[str, object], args: argparse.Namespace, step: int,
                        last_seen_step: int, last_seen_size: str) -> dict[str, object]:
    if dec["action"] != "back" or args.allow_reverse:
        return dec
    if dec["target_visible"] and dec["target_size"] in {"medium", "large"}:
        return force_arrived(dec, "reverse requested while target is close; stop instead")
    if last_seen_step > 0 and step - last_seen_step <= args.target_memory_steps and last_seen_size in {"medium", "large"}:
        return force_arrived(dec, "reverse requested just after close target observation; stop instead")
    side = "turn_left" if step % 2 else "turn_right"
    return force_recovery(dec, args, side, "reverse disabled; scan instead",
                          degrees=args.guard_turn_degrees)


def apply_target_memory_guard(dec: dict[str, object], args: argparse.Namespace, actions: Deque[str],
                              step: int, last_seen_step: int, last_seen_bearing: str,
                              last_seen_size: str, approach_streak: int) -> dict[str, object]:
    if dec["target_visible"] or dec["arrived"] or last_seen_step <= 0:
        return dec
    age = step - last_seen_step
    if age <= 0 or age > args.target_memory_steps:
        return dec
    previous = actions[0] if actions else "none"
    if previous == "forward" and age <= 2 and last_seen_size in {"medium", "large"}:
        return force_arrived(
            dec,
            f"target was {last_seen_size}/{last_seen_bearing}, then disappeared after forward approach",
        )
    if previous == "forward" and age <= 2:
        side = side_from_bearing(last_seen_bearing, step)
        return force_recovery(
            dec,
            args,
            side,
            f"target was {last_seen_size}/{last_seen_bearing} {age} step(s) ago; rescan instead of reversing",
            degrees=args.reacquire_turn_degrees,
        )
    side = side_from_bearing(last_seen_bearing, step)
    if dec["action"] in {"forward", "stop"}:
        return force_recovery(
            dec,
            args,
            side,
            f"target was {last_seen_size}/{last_seen_bearing} {age} step(s) ago; rescan locally",
            degrees=args.reacquire_turn_degrees,
        )
    if dec["action"] in {"turn_left", "turn_right"}:
        guarded = dict(dec)
        if dec["action"] != side and last_seen_bearing != "center":
            guarded["action"] = side
        if int(guarded["turn_degrees"]) > args.reacquire_turn_degrees:
            guarded["turn_degrees"] = args.reacquire_turn_degrees
            guarded["duration_ms"] = turn_ms(args.reacquire_turn_degrees, args)
        guarded["reason"] = (
            f"{guarded['reason']} | local guard: small rescan near recent target"
        )[:160]
        return guarded
    return dec


def apply_arrival_guard(dec: dict[str, object], args: argparse.Namespace, approach_streak: int) -> dict[str, object]:
    if dec["arrived"] or not dec["target_visible"]:
        return dec
    centered = dec["target_bearing"] == "center"
    closeish = dec["target_size"] in {"medium", "large"}
    if centered and closeish and approach_streak >= args.arrival_medium_forward_steps:
        return force_arrived(dec, f"{approach_streak} centered medium/large approach steps")
    return dec


def main() -> int:
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    parser = argparse.ArgumentParser(description="Gemini camera-only pulse driver; no local depth scanner")
    parser.add_argument("--target", default="yellow rubber bath duck / toy duckie / жёлтая резиновая уточка для ванной")
    parser.add_argument("--car-ip", default=os.environ.get("CAR_IP"))
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--secs", type=float, default=150.0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no step limit")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--camera-wait-sec", type=float, default=90.0)
    parser.add_argument("--capture-timeout", type=float, default=5.0)
    parser.add_argument("--settle-sec", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=18.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--speed", type=int, default=2)
    parser.add_argument("--min-speed", type=int, default=2)
    parser.add_argument("--max-speed", type=int, default=3)
    parser.add_argument("--escape-speed", type=int, default=3)
    parser.add_argument("--turnspeed", type=int, default=75)
    parser.add_argument("--default-ms", type=int, default=350)
    parser.add_argument("--min-pulse-ms", type=int, default=150)
    parser.add_argument("--max-duration-ms", type=int, default=650)
    parser.add_argument("--max-back-ms", type=int, default=500)
    parser.add_argument("--default-turn-degrees", type=int, default=18)
    parser.add_argument("--min-turn-degrees", type=int, default=8)
    parser.add_argument("--max-turn-degrees", type=int, default=35)
    parser.add_argument("--ms-per-90", type=int, default=900)
    parser.add_argument("--min-turn-ms", type=int, default=120)
    parser.add_argument("--max-turn-ms", type=int, default=450)
    parser.add_argument("--max-trim", type=int, default=18)
    parser.add_argument("--near-target-trim", type=int, default=7)
    parser.add_argument("--repeat-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-forward-repeat", type=int, default=4)
    parser.add_argument("--max-turn-repeat", type=int, default=2)
    parser.add_argument("--max-stop-repeat", type=int, default=1)
    parser.add_argument("--stuck-change", type=float, default=0.015)
    parser.add_argument("--guard-turn-degrees", type=int, default=22)
    parser.add_argument("--target-memory-steps", type=int, default=8)
    parser.add_argument("--reacquire-turn-degrees", type=int, default=14)
    parser.add_argument("--arrival-medium-forward-steps", type=int, default=3)
    parser.add_argument("--search-turns-before-forward", type=int, default=4)
    parser.add_argument("--search-forward-burst", type=int, default=3)
    parser.add_argument("--explore-forward-ms", type=int, default=500)
    parser.add_argument("--allow-reverse", action="store_true")
    parser.add_argument("--map-cell-size", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.car_ip:
        parser.error("set CAR_IP in .env or pass --car-ip")
    if not args.api_key:
        parser.error("set GEMINI_API_KEY/GOOGLE_API_KEY in .env or pass --api-key")

    car = Car(args.car_ip, args.dry_run)
    frames: Deque[tuple[float, bytes]] = deque(maxlen=max(2, args.frames))
    history: Deque[str] = deque(maxlen=args.history)
    actions: Deque[str] = deque(maxlen=8)
    smap = SimpleMap(args.map_cell_size)
    last_change: Optional[float] = None
    last_seen_step = 0
    last_seen_bearing = "none"
    last_seen_size = "none"
    approach_streak = 0
    relocate_remaining = 0
    lost_streak = 0

    car.stop()
    frames.append((time.time(), wait_camera(args.car_ip, args.camera_wait_sec, args.capture_timeout)))
    car.set_turnspeed(args.turnspeed)
    print(
        f"gemini_drive | {'DRY-RUN' if args.dry_run else 'DRIVING'} | "
        f"model={args.model} target={args.target} | camera-only pulses | no depth scanner"
    )

    end = time.time() + args.secs
    step = 0
    try:
        while time.time() < end and (args.max_steps <= 0 or step < args.max_steps):
            step += 1
            prompt_history = [control_note(actions, last_change, lost_streak, smap, args), *list(history)]
            t0 = time.time()
            raw = ask_gemini_action(args.model, args.api_key, args.target, list(frames), prompt_history, args)
            latency = time.time() - t0
            dec = clamp_decision(raw, args)
            dec = apply_arrival_guard(dec, args, approach_streak)
            dec = apply_reverse_guard(dec, args, step, last_seen_step, last_seen_size)
            dec = apply_target_memory_guard(
                dec,
                args,
                actions,
                step,
                last_seen_step,
                last_seen_bearing,
                last_seen_size,
                approach_streak,
            )
            dec, relocate_remaining = apply_search_guard(
                dec,
                args,
                actions,
                step,
                last_seen_step,
                relocate_remaining,
            )
            dec = apply_history_guard(dec, args, actions, last_change, step)
            if dec["target_visible"]:
                last_seen_step = step
                last_seen_bearing = str(dec["target_bearing"])
                last_seen_size = str(dec["target_size"])
            print(
                f"[gemini {step:03d}] {latency:.1f}s action={dec['action']} "
                f"speed={dec['speed']} trim={dec['trim']:+d} deg={dec['turn_degrees']} "
                f"ms={dec['duration_ms']} vis={dec['target_visible']} "
                f"bearing={dec['target_bearing']} size={dec['target_size']} "
                f"conf={dec['confidence']:.2f} {dec['reason'][:100]}",
                flush=True,
            )

            if dec["arrived"]:
                car.stop()
                print(f"[arrived] {dec['reason']}")
                return 0

            before = frames[-1][1] if frames else None
            car.pulse(str(dec["action"]), int(dec["speed"]), int(dec["trim"]), int(dec["duration_ms"]))
            actions.appendleft(str(dec["action"]))
            smap.update(dec)
            time.sleep(args.settle_sec)
            try:
                after = capture_jpeg(args.car_ip, args.capture_timeout)
            except Exception as e:
                car.stop()
                print(f"[camera] lost after step {step}: {e}")
                return 2
            frames.append((time.time(), after))
            delta = image_change(before, after)
            last_change = delta
            lost_streak = 0 if dec["target_visible"] else lost_streak + 1
            if dec["target_visible"] and dec["target_bearing"] == "center" and dec["target_size"] in {"medium", "large"}:
                approach_streak = approach_streak + 1 if dec["action"] == "forward" else max(approach_streak, 1)
            elif not dec["target_visible"] and (step - last_seen_step) > args.target_memory_steps:
                approach_streak = 0
            history.appendleft(history_line(step, dec, delta))
            if args.debug:
                changed = "unknown" if delta is None else f"{delta:.3f}"
                print(f"[outcome {step:03d}] image_change={changed} map={smap.summary()}", flush=True)
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        car.stop()
        print("STOP sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
