#!/usr/bin/env python3
"""
Hybrid autonomy for the ESP32-CAM car.

Architecture:
  * fast local loop: reads frames, runs local perception on the Mac, and owns all
    motor commands. Safety and collision avoidance live here.
  * slow VLM loop: looks at occasional frames and updates goal bias only. It never
    emits direct wheel commands.

The important behavioral rule is that the car normally drives with forward arcs
(speed + trim). It only pivots in place for short, bounded recovery nudges when
the local view says forward is blocked.

Examples:
  .venv/bin/python tools/reflex_drive.py --dry-run --vlm off --secs 10 --debug
  .venv/bin/python tools/reflex_drive.py --target "yellow toy road roller" --dry-run
  .venv/bin/python tools/reflex_drive.py --target "yellow toy road roller"
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
from typing import Deque, Iterable, Optional

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

import numpy as np
from PIL import Image


ROOT = REPO_ROOT
TOOLS = SCRIPT_DIR


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


def as_bool(v: object, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    if v is None:
        return default
    return bool(v)


def enum(v: object, allowed: set[str], default: str) -> str:
    s = str(v or "").strip().lower()
    return s if s in allowed else default


@dataclass
class FrameSample:
    t: float
    seq: int
    jpg: bytes
    pil: Image.Image


@dataclass
class Perception:
    t: float
    seq: int
    source: str
    left: float
    center: float
    right: float
    latency: float

    @property
    def best_side(self) -> str:
        return "left" if self.left >= self.right else "right"

    @property
    def scores(self) -> dict[str, float]:
        return {"left": self.left, "center": self.center, "right": self.right}


@dataclass
class Goal:
    t: float = 0.0
    source: str = "default"
    mode: str = "explore"
    route_bias: str = "center"
    forward_intent: str = "advance"
    target_visible: bool = False
    target_bearing: str = "none"
    target_size: str = "none"
    arrived: bool = False
    caution: bool = False
    confidence: float = 0.0
    reason: str = "explore open floor"


@dataclass
class MapState:
    summary: str = "map: starting area; no explored cells yet"
    route_bias: str = "center"
    target_cell: Optional[tuple[int, int]] = None
    current_cell: tuple[int, int] = (0, 0)
    visited_cells: int = 0
    looping: bool = False
    loop_reason: str = ""


@dataclass
class Shared:
    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = True
    frame: Optional[FrameSample] = None
    ring: Deque[FrameSample] = field(default_factory=lambda: deque(maxlen=48))
    perception: Optional[Perception] = None
    goal: Goal = field(default_factory=Goal)
    capture_errors: int = 0
    perception_errors: int = 0
    vlm_errors: int = 0
    map_state: MapState = field(default_factory=MapState)


class ExplorerMemory:
    """Tiny egocentric topological map for exploration, not metric navigation."""

    def __init__(self, cell_size: float = 1.0) -> None:
        self.cell_size = cell_size
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.cell_ticks: dict[tuple[int, int], int] = {}
        self.cell_entries: dict[tuple[int, int], int] = {}
        self.blocked: dict[tuple[int, int], float] = {}
        self.blocked_events: Deque[float] = deque(maxlen=32)
        self.events: Deque[str] = deque(maxlen=8)
        self.target_observations: Deque[tuple[float, tuple[int, int], str, str]] = deque(maxlen=6)
        self.last_t: Optional[float] = None
        self.last_cell: Optional[tuple[int, int]] = None
        self.last_new_cell_t = time.time()
        self.last_action = "start"
        self.map_bias = "center"
        self.committed_bias = "center"
        self.committed_until = 0.0
        self.looping = False
        self.loop_reason = ""

    def cell(self, x: Optional[float] = None, y: Optional[float] = None) -> tuple[int, int]:
        xx = self.x if x is None else x
        yy = self.y if y is None else y
        return (int(round(xx / self.cell_size)), int(round(yy / self.cell_size)))

    def direction_cell(self, rel: str, distance: float = 1.0) -> tuple[int, int]:
        offset = {"left": -0.75, "center": 0.0, "right": 0.75}[rel]
        h = self.heading + offset
        return self.cell(self.x + math.cos(h) * distance, self.y + math.sin(h) * distance)

    def bearing_offset(self, bearing: str) -> float:
        return {
            "far_left": -0.9,
            "left": -0.45,
            "center": 0.0,
            "right": 0.45,
            "far_right": 0.9,
            "none": 0.0,
        }.get(bearing, 0.0)

    def update(self, action: str, speed: int, trim: int, per: Perception, goal: Goal,
               args: argparse.Namespace) -> None:
        now = time.time()
        dt = 0.0 if self.last_t is None else clamp(now - self.last_t, 0.0, 1.5)
        self.last_t = now

        if action.startswith("loop-escape"):
            dist = dt * (0.18 + 0.12 * max(1, speed))
            self.x -= math.cos(self.heading) * dist
            self.y -= math.sin(self.heading) * dist
            self.heading += (-1.0 if "left" in action else 1.0) * max(0.4, args.loop_escape_turn_sec) * 1.8
        elif action.startswith("arc"):
            # Arbitrary units: good enough to distinguish "already explored nearby".
            dist = dt * (0.12 + 0.16 * max(1, speed))
            self.heading += (trim / max(1, args.max_trim)) * dt * 0.9
            self.x += math.cos(self.heading) * dist
            self.y += math.sin(self.heading) * dist
        elif "back" in action or "retreat" in action:
            dist = dt * (0.18 + 0.12 * max(1, speed))
            self.x -= math.cos(self.heading) * dist
            self.y -= math.sin(self.heading) * dist
        elif "left" in action:
            self.heading -= dt * 1.8
        elif "right" in action:
            self.heading += dt * 1.8

        cur = self.cell()
        self.cell_ticks[cur] = self.cell_ticks.get(cur, 0) + 1
        if cur != self.last_cell:
            self.cell_entries[cur] = self.cell_entries.get(cur, 0) + 1
            if self.cell_entries[cur] == 1:
                self.last_new_cell_t = now
                self.events.appendleft(f"entered new cell {cur}")
            self.last_cell = cur
        self.last_action = action

        if per.center < args.block or action.startswith("dead-end"):
            front = self.direction_cell("center")
            self.blocked[front] = now
            self.blocked_events.append(now)
            self.events.appendleft(f"blocked front near cell {front}; recovered via {action}")

        if goal.target_visible:
            h = self.heading + self.bearing_offset(goal.target_bearing)
            dist = {"small": 3.0, "medium": 2.0, "large": 1.0, "none": 2.0}.get(goal.target_size, 2.0)
            tc = self.cell(self.x + math.cos(h) * dist, self.y + math.sin(h) * dist)
            self.target_observations.appendleft((now, tc, goal.target_bearing, goal.target_size))
            self.events.appendleft(f"target seen {goal.target_bearing}/{goal.target_size} near cell {tc}")

        self.update_loop_state(now, goal, args)
        self.map_bias = self.choose_bias(per, args, now)

    def update_loop_state(self, now: float, goal: Goal, args: argparse.Namespace) -> None:
        if goal.target_visible:
            self.looping = False
            self.loop_reason = ""
            return
        recent_blocks = sum(1 for t in self.blocked_events if now - t <= args.loop_window_sec)
        no_new_cell_for = now - self.last_new_cell_t
        cur_entries = self.cell_entries.get(self.cell(), 0)
        reasons = []
        if recent_blocks >= args.loop_block_events:
            reasons.append(f"{recent_blocks} blocked/dead-end recoveries in {args.loop_window_sec:.0f}s")
        if no_new_cell_for >= args.loop_no_new_cell_sec:
            reasons.append(f"no new cell for {no_new_cell_for:.1f}s")
        if cur_entries >= args.loop_cell_entries:
            reasons.append(f"cell revisited {cur_entries} times")
        self.looping = bool(reasons)
        self.loop_reason = "; ".join(reasons[:2])

    def direction_score(self, rel: str, per: Perception, args: argparse.Namespace, now: float) -> float:
        openness = {"left": per.left, "center": per.center, "right": per.right}
        cell = self.direction_cell(rel)
        entries = self.cell_entries.get(cell, 0)
        ticks = self.cell_ticks.get(cell, 0)
        blocked_recent = cell in self.blocked and now - self.blocked[cell] < args.block_memory_sec
        frontier_bonus = args.frontier_bonus if entries == 0 else 0.0
        return (
            openness[rel] * 2.0
            + frontier_bonus
            - entries * args.revisit_penalty
            - min(ticks, 30) * 0.015
            - (args.blocked_penalty if blocked_recent else 0.0)
        )

    def choose_bias(self, per: Perception, args: argparse.Namespace, now: float) -> str:
        choices = ("left", "right") if self.looping else ("left", "center", "right")
        scored = [(rel, self.direction_score(rel, per, args, now)) for rel in choices]
        best = max(scored, key=lambda item: item[1])

        if self.looping:
            self.committed_bias = best[0]
            self.committed_until = now + args.explore_commit_sec
            return best[0]

        committed_score = dict(scored).get(self.committed_bias)
        if (
            committed_score is not None
            and now < self.committed_until
            and committed_score >= best[1] - args.commit_switch_margin
        ):
            return self.committed_bias

        if best[0] != self.committed_bias or now >= self.committed_until:
            self.committed_bias = best[0]
            self.committed_until = now + args.explore_commit_sec
            self.events.appendleft(f"committed exploration bias {best[0]}")
        return best[0]

    def state(self) -> MapState:
        cur = self.cell()
        target_cell = self.target_observations[0][1] if self.target_observations else None
        heading_deg = int((math.degrees(self.heading) % 360 + 22.5) // 45 * 45) % 360
        recent = "; ".join(list(self.events)[:4]) if self.events else "no blocked or target events yet"
        blocked = sorted(self.blocked, key=self.blocked.get, reverse=True)[:5]
        summary = (
            f"map: current_cell={cur}, heading~{heading_deg}deg, visited_cells={len(self.cell_entries)}, "
            f"map_route_bias={self.map_bias}, committed_bias={self.committed_bias}, "
            f"looping={self.looping}, loop_reason={self.loop_reason or 'none'}, "
            f"recent_blocked={blocked or 'none'}, "
            f"last_target_cell={target_cell or 'unknown'}, recent_events={recent}"
        )
        return MapState(
            summary=summary,
            route_bias=self.map_bias,
            target_cell=target_cell,
            current_cell=cur,
            visited_cells=len(self.cell_entries),
            looping=self.looping,
            loop_reason=self.loop_reason,
        )


def heuristic_free_space(pil_rgb: Image.Image) -> dict[str, float]:
    gray = np.asarray(pil_rgb.convert("L"), dtype=np.float32)
    h, w = gray.shape
    mid = gray[int(h * 0.45):int(h * 0.92), :]
    ref = gray[int(h * 0.85):, int(w * 0.35):int(w * 0.65)]
    ref_mean = float(ref.mean()) + 1e-3
    ref_tex = float(np.abs(np.diff(ref, axis=1)).mean()) + 1e-3
    out: list[float] = []
    _, mw = mid.shape
    for a, b in ((0, mw // 3), (mw // 3, 2 * mw // 3), (2 * mw // 3, mw)):
        reg = mid[:, a:b]
        bright_diff = abs(float(reg.mean()) - ref_mean) / ref_mean
        clutter = float(np.abs(np.diff(reg, axis=1)).mean() +
                        np.abs(np.diff(reg, axis=0)).mean())
        openness = 1.0 / (1.0 + 3.0 * bright_diff + clutter / (ref_tex * 6.0))
        out.append(float(round(openness, 3)))
    return {"left": out[0], "center": out[1], "right": out[2]}


def capture_loop(shared: Shared, car_ip: str, hz: float, timeout: float) -> None:
    url = f"http://{car_ip}/capture"
    period = 1.0 / max(0.5, hz)
    seq = 0
    while shared.running:
        t0 = time.time()
        try:
            jpg = urllib.request.urlopen(url, timeout=timeout).read()
            if jpg[:2] != b"\xff\xd8":
                raise ValueError("capture did not start with JPEG SOI")
            pil = Image.open(BytesIO(jpg)).convert("RGB")
            sample = FrameSample(t=time.time(), seq=seq, jpg=jpg, pil=pil)
            seq += 1
            with shared.lock:
                shared.frame = sample
                shared.ring.append(sample)
                shared.capture_errors = 0
        except Exception as e:
            with shared.lock:
                shared.capture_errors += 1
                n = shared.capture_errors
            if n == 1 or n % 10 == 0:
                print(f"[capture] retry {n}: {e}")
            time.sleep(0.25)
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def perception_loop(shared: Shared, args: argparse.Namespace) -> None:
    DP = None
    if args.perception == "depth":
        sys.path.insert(0, TOOLS)
        import depth_perception as DP  # type: ignore

        print(f"loading local depth model from {args.depth_model} ...")
        device = DP.load(model_id=args.depth_model, size=args.depth_size)
        print(f"local depth ready on {device}")

    last_seq = -1
    period = 1.0 / max(0.5, args.local_hz)
    while shared.running:
        with shared.lock:
            sample = shared.frame
        if sample is None or sample.seq == last_seq:
            time.sleep(0.02)
            continue

        t0 = time.time()
        try:
            if DP is not None:
                scores = DP.free_space(DP.depth(sample.pil))
                source = "depth"
            else:
                scores = heuristic_free_space(sample.pil)
                source = "heuristic"
            per = Perception(
                t=time.time(),
                seq=sample.seq,
                source=source,
                left=float(scores["left"]),
                center=float(scores["center"]),
                right=float(scores["right"]),
                latency=time.time() - t0,
            )
            with shared.lock:
                shared.perception = per
                shared.perception_errors = 0
            last_seq = sample.seq
        except Exception as e:
            with shared.lock:
                shared.perception_errors += 1
                n = shared.perception_errors
            if n == 1 or n % 10 == 0:
                print(f"[perception] retry {n}: {e}")
            time.sleep(0.1)

        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


GOAL_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["explore", "seek", "approach", "hold"]},
        "route_bias": {"type": "string", "enum": ["left", "center", "right"]},
        "forward_intent": {"type": "string", "enum": ["advance", "hold", "retreat"]},
        "target_visible": {"type": "boolean"},
        "target_bearing": {
            "type": "string",
            "enum": ["far_left", "left", "center", "right", "far_right", "none"],
        },
        "target_size": {"type": "string", "enum": ["none", "small", "medium", "large"]},
        "arrived": {"type": "boolean"},
        "caution": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": [
        "mode",
        "route_bias",
        "forward_intent",
        "target_visible",
        "target_bearing",
        "target_size",
        "arrived",
        "caution",
        "confidence",
        "reason",
    ],
}


def build_goal_prompt(target: Optional[str], history: Iterable[str], map_state: MapState) -> str:
    mem = "\n".join(history) if history else "none"
    if target:
        task = (
            f'TASK: find and approach this target: "{target}". '
            "You may bias the route toward the target, but do not command wheels."
        )
    else:
        task = (
            "TASK: explore the indoor space. Prefer routes that look open and useful, "
            "but do not command wheels."
        )
    return (
        "You are the slow goal-planning VLM for a small ground robot with a forward "
        "camera. A separate fast local controller on the Mac handles collision "
        "avoidance and all motor commands.\n"
        f"{task}\n"
        "Your output is only a high-level goal update:\n"
        " - mode: explore, seek, approach, or hold.\n"
        " - route_bias: left, center, or right. This is only a preference.\n"
        " - forward_intent: advance if changing viewpoint helps, hold if the scene "
        "looks unsafe or ambiguous, retreat only if the robot appears boxed in.\n"
        " - target_* fields describe the current frame; use none/false without a target.\n"
        " - arrived is true only when the requested target is large, centered, and close.\n"
        "Use the map memory to reason about where the robot has already been. Prefer "
        "unvisited/open directions, avoid recent blocked cells, and if the target was "
        "seen before but is missing now, bias toward the last target area when safe.\n"
        "The local controller may ignore your bias for safety. Do not say spin, turn "
        "around, drive, stop, speed, trim, or any wheel command in reason.\n"
        f"Compact exploration map:\n{map_state.summary}\n"
        f"Recent goal memory, newest first:\n{mem}"
    )


def parse_json_object(text: str) -> Optional[dict[str, object]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    raw = match.group(0)
    for candidate in (raw, raw.replace("'", '"')):
        try:
            out = json.loads(candidate)
            return out if isinstance(out, dict) else None
        except Exception:
            pass
    return None


def clamp_goal(raw: Optional[dict[str, object]], source: str) -> Goal:
    allowed_mode = {"explore", "seek", "approach", "hold"}
    allowed_bias = {"left", "center", "right"}
    allowed_intent = {"advance", "hold", "retreat"}
    allowed_bearing = {"far_left", "left", "center", "right", "far_right", "none"}
    allowed_size = {"none", "small", "medium", "large"}
    if not isinstance(raw, dict):
        raw = {}
    bearing = enum(raw.get("target_bearing"), allowed_bearing, "none")
    bias = enum(raw.get("route_bias"), allowed_bias, "center")
    if bearing in {"far_left", "left"}:
        bias = "left"
    elif bearing in {"far_right", "right"}:
        bias = "right"
    conf = raw.get("confidence", 0.0)
    try:
        confidence = float(conf)
    except Exception:
        confidence = 0.0
    return Goal(
        t=time.time(),
        source=source,
        mode=enum(raw.get("mode"), allowed_mode, "explore"),
        route_bias=bias,
        forward_intent=enum(raw.get("forward_intent"), allowed_intent, "advance"),
        target_visible=as_bool(raw.get("target_visible"), False),
        target_bearing=bearing,
        target_size=enum(raw.get("target_size"), allowed_size, "none"),
        arrived=as_bool(raw.get("arrived"), False),
        caution=as_bool(raw.get("caution"), False),
        confidence=clamp(confidence, 0.0, 1.0),
        reason=str(raw.get("reason", ""))[:140],
    )


def ask_gemini_goal(
    model: str,
    key: str,
    prompt: str,
    frames: list[FrameSample],
    timeout: float,
    retries: int,
) -> Goal:
    parts: list[dict[str, object]] = [{"text": prompt}]
    for i, frame in enumerate(frames):
        tag = "current" if i == len(frames) - 1 else f"{len(frames) - i - 1} step(s) ago"
        parts.append({"text": f"Frame {i + 1}/{len(frames)} ({tag}):"})
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(frame.jpg).decode("ascii"),
            }
        })

    gen: dict[str, object] = {
        "responseMimeType": "application/json",
        "responseSchema": GOAL_SCHEMA,
        "temperature": 0.25,
    }
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
            text = doc["candidates"][0]["content"]["parts"][0]["text"]
            raw = parse_json_object(text)
            if raw is None:
                raw = json.loads(text)
            return clamp_goal(raw, "gemini")
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            last = e
            time.sleep(1.0)
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise RuntimeError(f"Gemini goal failed: {last}")


def goal_from_seek_decision(dec: dict[str, object]) -> Goal:
    visible = as_bool(dec.get("target_visible"), False)
    bearing = enum(
        dec.get("target_bearing"),
        {"far_left", "left", "center", "right", "far_right", "none"},
        "none",
    )
    strategy = enum(
        dec.get("search_strategy"),
        {"spin_left", "spin_right", "forward", "turn_around", "stop"},
        "forward",
    )
    if bearing in {"far_left", "left"} or strategy == "spin_left":
        bias = "left"
    elif bearing in {"far_right", "right"} or strategy in {"spin_right", "turn_around"}:
        bias = "right"
    else:
        bias = "center"
    safe_forward = as_bool(dec.get("safe_forward"), True)
    return Goal(
        t=time.time(),
        source="local-vlm",
        mode="approach" if visible else "seek",
        route_bias=bias,
        forward_intent="advance" if safe_forward and strategy != "stop" else "hold",
        target_visible=visible,
        target_bearing=bearing,
        target_size=enum(dec.get("target_size"), {"none", "small", "medium", "large"}, "none"),
        arrived=as_bool(dec.get("arrived"), False),
        caution=not safe_forward,
        confidence=0.6,
        reason=str(dec.get("reason", ""))[:140],
    )


def pick_frames(shared: Shared, n: int, spacing: float) -> list[FrameSample]:
    with shared.lock:
        ring = list(shared.ring)
    if not ring:
        return []
    latest = ring[-1].t
    picks: list[FrameSample] = []
    for k in range(n - 1, -1, -1):
        target_t = latest - k * spacing
        best = min(ring, key=lambda f: abs(f.t - target_t))
        if not picks or picks[-1].seq != best.seq:
            picks.append(best)
    return picks


def vlm_loop(shared: Shared, args: argparse.Namespace) -> None:
    if args.vlm == "off":
        return

    hist: Deque[str] = deque(maxlen=5)
    local_vlm = None
    if args.vlm == "local":
        if not args.target:
            print("[vlm] local VLM is target-oriented; using generic exploration target")
        sys.path.insert(0, TOOLS)
        import vlm_local as local_vlm  # type: ignore

        print(f"loading local VLM from {args.local_vlm_model_dir} ...")
        print(f"local VLM ready: {local_vlm.load_local(args.local_vlm_model_dir)}")

    while shared.running:
        frames = pick_frames(shared, args.vlm_frames, args.vlm_frame_space)
        if not frames:
            time.sleep(0.1)
            continue
        with shared.lock:
            map_state = shared.map_state

        try:
            prompt = build_goal_prompt(args.target, hist, map_state)
            if args.vlm == "gemini":
                goal = ask_gemini_goal(
                    args.model,
                    args.gemini_key,
                    prompt,
                    frames,
                    args.vlm_timeout,
                    args.vlm_retries,
                )
            else:
                target = args.target or "an open safe route through the room"
                dec = local_vlm.ask_seek_local(  # type: ignore[union-attr]
                    target,
                    [(f.t, f.jpg) for f in frames],
                    [map_state.summary, *list(hist)],
                    max_frames=args.vlm_frames,
                )
                goal = goal_from_seek_decision(dec)

            with shared.lock:
                shared.goal = goal
                shared.vlm_errors = 0
            hist.appendleft(
                f"- {goal.mode} bias={goal.route_bias} intent={goal.forward_intent} "
                f"visible={goal.target_visible} reason={goal.reason[:60]}"
            )
            if args.debug:
                print(
                    f"[vlm] {goal.mode} bias={goal.route_bias} intent={goal.forward_intent} "
                    f"vis={goal.target_visible} conf={goal.confidence:.2f} {goal.reason[:70]}"
                )
        except Exception as e:
            with shared.lock:
                shared.vlm_errors += 1
                n = shared.vlm_errors
            if n == 1 or n % 3 == 0:
                print(f"[vlm] keeping previous goal after error {n}: {e}")

        end = time.time() + args.vlm_sec
        while shared.running and time.time() < end:
            time.sleep(0.05)


class Car:
    def __init__(self, ip: str, dry_run: bool) -> None:
        self.base = f"http://{ip}"
        self.dry_run = dry_run
        self.last_drive: Optional[tuple[int, int]] = None
        self.last_motion = "stop"
        self.last_send = 0.0

    def ctl(self, query: str, timeout: float = 3.0) -> bool:
        if self.dry_run:
            return True
        try:
            urllib.request.urlopen(f"{self.base}/control?{query}", timeout=timeout).read()
            return True
        except Exception as e:
            print(f"[ctl] {query}: {e}")
            return False

    def set_turnspeed(self, turnspeed: int) -> None:
        self.ctl(f"var=turnspeed&val={int(clamp(turnspeed, 0, 255))}")

    def forward(self, speed: int, trim: int, force: bool = False) -> None:
        speed = int(clamp(speed, 0, 8))
        trim = int(clamp(trim, -32, 32))
        if speed <= 0:
            self.stop()
            return
        now = time.time()
        drive = (speed, trim)
        if not force and self.last_motion == "forward" and self.last_drive == drive and now - self.last_send < 0.9:
            return
        self.ctl(f"var=speed&val={speed}")
        self.ctl(f"var=trim&val={trim}")
        self.ctl("var=car&val=1")
        self.last_motion = "forward"
        self.last_drive = drive
        self.last_send = now

    def stop(self) -> None:
        if self.last_motion == "stop" and time.time() - self.last_send < 0.5:
            return
        self.ctl("var=car&val=5")
        self.last_motion = "stop"
        self.last_drive = None
        self.last_send = time.time()

    def reverse(self, speed: int, secs: float) -> None:
        self.ctl(f"var=speed&val={int(clamp(speed, 0, 8))}")
        self.ctl("var=trim&val=0")
        self.ctl("var=car&val=2")
        self.last_motion = "reverse"
        self.last_drive = None
        self.last_send = time.time()
        time.sleep(max(0.0, secs))
        self.stop()

    def pivot(self, side: str, secs: float) -> None:
        val = "3" if side == "left" else "4"
        self.ctl(f"var=car&val={val}")
        self.last_motion = f"pivot-{side}"
        self.last_drive = None
        self.last_send = time.time()
        time.sleep(max(0.0, secs))
        self.stop()


def goal_bias_value(goal: Goal) -> float:
    if goal.target_visible:
        if goal.target_bearing == "far_left":
            return -1.0
        if goal.target_bearing == "left":
            return -0.65
        if goal.target_bearing == "far_right":
            return 1.0
        if goal.target_bearing == "right":
            return 0.65
        return 0.0
    if goal.route_bias == "left":
        return -0.7
    if goal.route_bias == "right":
        return 0.7
    return 0.0


def route_bias_value(route_bias: str) -> float:
    if route_bias == "left":
        return -0.7
    if route_bias == "right":
        return 0.7
    return 0.0


def effective_goal(goal: Goal, stale_sec: float) -> Goal:
    if goal.source == "local-only":
        return goal
    if goal.t <= 0 or time.time() - goal.t > stale_sec:
        return Goal(reason="VLM stale; local exploration")
    return goal


def command_from_state(per: Perception, goal: Goal, map_state: MapState,
                       args: argparse.Namespace) -> tuple[str, int, int]:
    left, center, right = per.left, per.center, per.right
    if goal.target_visible:
        # Keep the target in view. Obstacle openness is only a small correction
        # here; the hard safety decisions happen before this function.
        open_bias = clamp(
            (right - left) * args.target_open_gain,
            -args.target_open_bias_limit,
            args.target_open_bias_limit,
        )
        steer = clamp(open_bias + args.target_weight * goal_bias_value(goal), -1.0, 1.0)
        trim_limit = args.target_max_trim
        if goal.target_bearing in {"far_left", "far_right"}:
            trim_limit = args.target_far_max_trim
        elif goal.target_bearing == "center":
            trim_limit = args.target_center_max_trim
        trim = int(clamp(round(steer * args.max_trim), -trim_limit, trim_limit))
    else:
        open_bias = clamp((right - left) * args.open_gain, -1.0, 1.0)
        steer = clamp(
            open_bias + args.goal_weight * goal_bias_value(goal)
            + args.map_weight * route_bias_value(map_state.route_bias),
            -1.0,
            1.0,
        )
        trim = int(round(steer * args.max_trim))

    speed = max(args.min_drive_speed, args.speed)
    close_to_target = goal.target_visible and goal.target_size == "large"
    drive_floor = args.approach_speed if close_to_target else args.min_drive_speed
    if center < args.open:
        speed = max(drive_floor, speed - 1)
    if abs(trim) >= args.max_trim * 0.75:
        speed = max(drive_floor, speed - 1)
    if goal.caution:
        speed = min(speed, args.caution_speed)
    if close_to_target:
        speed = min(speed, args.approach_speed)
    if goal.forward_intent == "hold":
        speed = 0
    elif goal.forward_intent == "retreat":
        return "retreat", 0, 0
    return "arc", int(speed), trim


def controller_loop(shared: Shared, args: argparse.Namespace) -> None:
    car = Car(args.car_ip, args.dry_run)
    car.set_turnspeed(args.turnspeed)
    memory = ExplorerMemory(args.map_cell_size)
    turns_without_forward = 0
    last_loop_escape = 0.0
    last_log = 0.0
    t_end = time.time() + args.secs if args.secs > 0 else None

    print(
        f"hybrid_drive | {'DRY-RUN' if args.dry_run else 'DRIVING'} | "
        f"vlm={args.vlm} perception={args.perception} speed={args.speed} "
        f"target={args.target or '-'}"
    )
    print("fast local loop owns motors; VLM only updates goal bias. Ctrl-C to stop.")

    wait_log = 0.0
    try:
        while shared.running:
            if t_end is not None and time.time() >= t_end:
                break

            with shared.lock:
                per = shared.perception
                raw_goal = shared.goal
                cap_err = shared.capture_errors
                per_err = shared.perception_errors
                map_state = shared.map_state
            now = time.time()
            goal = effective_goal(raw_goal, args.goal_stale_sec)

            if per is None:
                if now - wait_log > 1.5:
                    print(f"[waiting] perception pending capture_errors={cap_err} perception_errors={per_err}")
                    wait_log = now
                car.stop()
                time.sleep(0.05)
                continue

            age = now - per.t
            if age > args.stale_view_sec:
                if now - wait_log > 1.0:
                    print(f"[safe-stop] local view stale: {age:.2f}s")
                    wait_log = now
                car.stop()
                time.sleep(0.05)
                continue

            if args.target and goal.arrived and now - goal.t <= args.goal_stale_sec:
                car.stop()
                print(f"[arrived] {goal.reason}")
                break
            if (args.target and args.stop_on_large_target and goal.target_visible
                    and goal.target_size == "large" and now - goal.t <= args.goal_stale_sec):
                car.stop()
                print(f"[arrived-local] target is large/visible; stopping near it. {goal.reason}")
                break
            if (args.target and args.stop_on_medium_center and goal.target_visible
                    and goal.target_bearing == "center" and goal.target_size == "medium"
                    and goal.confidence >= args.arrival_confidence
                    and now - goal.t <= args.goal_stale_sec):
                car.stop()
                print(f"[arrived-local] target is centered/medium; stopping near it. {goal.reason}")
                break

            left, center, right = per.left, per.center, per.right
            best_side = per.best_side
            action = "none"
            action_speed = 0
            action_trim = 0

            target_push = (
                goal.target_visible
                and goal.target_size in {"small", "medium"}
                and center >= args.target_push_min_open
            )

            if (map_state.looping and not goal.target_visible
                    and now - last_loop_escape >= args.loop_escape_cooldown):
                side = map_state.route_bias if map_state.route_bias in {"left", "right"} else best_side
                action = f"loop-escape back+{side}"
                car.reverse(max(args.speed, args.escape_speed), args.back_secs * 1.25)
                car.pivot(side, args.loop_escape_turn_sec)
                action_speed = max(args.speed, args.escape_speed)
                turns_without_forward = 0
                last_loop_escape = now
            elif goal.forward_intent == "retreat":
                action = "vlm-retreat"
                car.reverse(max(args.speed, args.escape_speed), args.back_secs)
                car.pivot(best_side, args.escape_turn_sec)
                action_speed = max(args.speed, args.escape_speed)
                turns_without_forward = 0
            elif target_push and center < args.slow:
                _, speed, trim = command_from_state(per, goal, map_state, args)
                speed = max(speed, args.min_drive_speed)
                action = f"target-push speed={speed} trim={trim:+d}"
                action_speed = speed
                action_trim = trim
                car.forward(speed, trim, force=True)
                turns_without_forward = 0
            elif center < args.block and max(left, right) < args.side_open:
                action = f"dead-end back+{best_side}"
                car.reverse(max(args.speed, args.escape_speed), args.back_secs)
                car.pivot(best_side, args.escape_turn_sec)
                action_speed = max(args.speed, args.escape_speed)
                turns_without_forward = 0
            elif center < args.block:
                action = f"blocked nudge {best_side}"
                if turns_without_forward >= args.max_pivots_before_reverse:
                    car.reverse(max(args.speed, args.escape_speed), args.back_secs)
                    action_speed = max(args.speed, args.escape_speed)
                    turns_without_forward = 0
                car.pivot(best_side, args.turn_pulse)
                turns_without_forward += 1
            elif center < args.slow:
                action = f"slow nudge {best_side}"
                if turns_without_forward < args.max_pivots_before_forward:
                    car.pivot(best_side, args.nudge_pulse)
                    turns_without_forward += 1
                else:
                    trim = -args.max_trim if best_side == "left" else args.max_trim
                    action_speed = max(1, args.min_drive_speed)
                    action_trim = trim
                    car.forward(action_speed, trim, force=True)
                    turns_without_forward = 0
            else:
                kind, speed, trim = command_from_state(per, goal, map_state, args)
                if kind == "retreat":
                    action = "retreat"
                    car.reverse(max(args.speed, args.escape_speed), args.back_secs)
                    car.pivot(best_side, args.escape_turn_sec)
                    action_speed = max(args.speed, args.escape_speed)
                elif speed <= 0:
                    action = "hold"
                    car.stop()
                else:
                    action = f"arc speed={speed} trim={trim:+d}"
                    action_speed = speed
                    action_trim = trim
                    car.forward(speed, trim)
                    turns_without_forward = 0

            memory.update(action, action_speed, action_trim, per, goal, args)
            map_state = memory.state()
            with shared.lock:
                shared.map_state = map_state

            if args.debug or now - last_log >= args.log_sec:
                g_age = now - goal.t if goal.t else 999.0
                print(
                    f"[local] L={left:.2f} C={center:.2f} R={right:.2f} "
                    f"age={age:.2f}s act={action} | "
                    f"goal={goal.mode}/{goal.route_bias}/{goal.forward_intent} "
                    f"map={map_state.route_bias}/{map_state.current_cell}/{map_state.visited_cells}"
                    f"{'/loop' if map_state.looping else ''} "
                    f"g_age={g_age:.1f}s {goal.reason[:55]}"
                )
                last_log = now

            time.sleep(1.0 / max(1.0, args.hz))
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        shared.running = False
        car.stop()
        print("STOP sent. done.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Hybrid controller: local fast perception drives; VLM supplies slow goal bias."
    )
    ap.add_argument("target_arg", nargs="?", help="optional target to seek")
    ap.add_argument("--target", help="optional target to seek; overrides positional target")
    ap.add_argument("--env", default=".env", help="load missing env vars from this file")
    ap.add_argument("--car-ip", help="ESP32-CAM IP; defaults to CAR_IP")

    ap.add_argument("--vlm", choices=["gemini", "local", "off"], default="gemini")
    ap.add_argument("--no-vlm", "--no-gemini", action="store_const", const="off", dest="vlm")
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--local-vlm-model-dir", default=os.path.join(ROOT, "models", "qwen2.5-vl-7b-4bit"))
    ap.add_argument("--vlm-sec", type=float, default=4.0, help="seconds between VLM goal updates")
    ap.add_argument("--vlm-timeout", type=float, default=25.0)
    ap.add_argument("--vlm-retries", type=int, default=1)
    ap.add_argument("--vlm-frames", type=int, default=2)
    ap.add_argument("--vlm-frame-space", type=float, default=0.6)
    ap.add_argument("--goal-stale-sec", type=float, default=12.0)

    ap.add_argument("--perception", choices=["depth", "heuristic"], default="depth")
    ap.add_argument("--depth-model", default=os.path.join(ROOT, "models", "da2-small"))
    ap.add_argument("--depth-size", type=int, default=252)
    ap.add_argument("--capture-hz", type=float, default=8.0)
    ap.add_argument("--capture-timeout", type=float, default=4.0)
    ap.add_argument("--local-hz", type=float, default=5.0)
    ap.add_argument("--hz", type=float, default=8.0)

    ap.add_argument("--speed", type=int, default=2, help="normal forward speed 0..8")
    ap.add_argument("--min-drive-speed", type=int, default=2,
                    help="minimum non-caution forward speed; helps climb carpet edges")
    ap.add_argument("--approach-speed", type=int, default=1,
                    help="speed cap when the target is already medium/large")
    ap.add_argument("--caution-speed", type=int, default=2)
    ap.add_argument("--escape-speed", type=int, default=4)
    ap.add_argument("--turnspeed", type=int, default=70, help="firmware in-place turn power 0..255")
    ap.add_argument("--max-trim", type=int, default=32)
    ap.add_argument("--open-gain", type=float, default=1.4)
    ap.add_argument("--target-open-gain", type=float, default=0.35,
                    help="small obstacle-bias correction while the target is visible")
    ap.add_argument("--target-open-bias-limit", type=float, default=0.22)
    ap.add_argument("--goal-weight", type=float, default=0.35)
    ap.add_argument("--target-weight", type=float, default=0.55)
    ap.add_argument("--target-max-trim", type=int, default=12,
                    help="normal steering limit while the target is visible")
    ap.add_argument("--target-far-max-trim", type=int, default=16,
                    help="steering limit when target is at far left/right edge")
    ap.add_argument("--target-center-max-trim", type=int, default=4,
                    help="steering limit when target is centered")
    ap.add_argument("--target-push-min-open", type=float, default=0.08,
                    help="minimum center openness for pushing over carpet toward a visible small/medium target")
    ap.add_argument("--map-weight", type=float, default=0.65,
                    help="local exploration-memory bias when target is not visible")
    ap.add_argument("--map-cell-size", type=float, default=1.0)
    ap.add_argument("--explore-commit-sec", type=float, default=4.5,
                    help="hold a selected exploration frontier this long unless blocked")
    ap.add_argument("--commit-switch-margin", type=float, default=0.35)
    ap.add_argument("--frontier-bonus", type=float, default=1.2)
    ap.add_argument("--revisit-penalty", type=float, default=1.0)
    ap.add_argument("--blocked-penalty", type=float, default=2.5)
    ap.add_argument("--block-memory-sec", type=float, default=45.0)
    ap.add_argument("--loop-window-sec", type=float, default=18.0)
    ap.add_argument("--loop-block-events", type=int, default=6)
    ap.add_argument("--loop-no-new-cell-sec", type=float, default=10.0)
    ap.add_argument("--loop-cell-entries", type=int, default=4)
    ap.add_argument("--loop-escape-turn-sec", type=float, default=0.85)
    ap.add_argument("--loop-escape-cooldown", type=float, default=5.0)

    ap.add_argument("--block", type=float, default=0.18, help="center openness below this is blocked")
    ap.add_argument("--slow", type=float, default=0.36, help="center openness below this pivots/crawls")
    ap.add_argument("--open", type=float, default=0.55, help="center openness above this is comfortable")
    ap.add_argument("--side-open", type=float, default=0.30, help="side openness needed to avoid dead-end")
    ap.add_argument("--turn-pulse", type=float, default=0.15)
    ap.add_argument("--nudge-pulse", type=float, default=0.09)
    ap.add_argument("--back-secs", type=float, default=0.35)
    ap.add_argument("--escape-turn-sec", type=float, default=0.38)
    ap.add_argument("--max-pivots-before-reverse", type=int, default=4)
    ap.add_argument("--max-pivots-before-forward", type=int, default=3)
    ap.add_argument("--stale-view-sec", type=float, default=1.0)
    ap.add_argument("--stop-on-large-target", action=argparse.BooleanOptionalAction, default=True,
                    help="stop when VLM reports the target as large/close")
    ap.add_argument("--stop-on-medium-center", action=argparse.BooleanOptionalAction, default=True,
                    help="stop when VLM repeatedly frames the target as centered and medium-sized")
    ap.add_argument("--arrival-confidence", type=float, default=0.65)

    ap.add_argument("--secs", type=float, default=60.0, help="0 means run until Ctrl-C")
    ap.add_argument("--dry-run", action="store_true", help="read camera and decide, but do not move")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--log-sec", type=float, default=0.75)
    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv(args.env)

    args.target = args.target if args.target is not None else args.target_arg
    args.car_ip = args.car_ip or os.environ.get("CAR_IP")
    args.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not args.car_ip:
        parser.error("set CAR_IP, pass --car-ip, or put CAR_IP in .env")
    if args.vlm == "gemini" and not args.gemini_key:
        parser.error("set GEMINI_API_KEY/GOOGLE_API_KEY, or run with --vlm off")
    if args.speed < 0 or args.speed > 8:
        parser.error("--speed must be 0..8")
    if args.min_drive_speed < 0 or args.min_drive_speed > 8:
        parser.error("--min-drive-speed must be 0..8")
    if args.approach_speed < 0 or args.approach_speed > 8:
        parser.error("--approach-speed must be 0..8")
    args.min_drive_speed = max(args.min_drive_speed, args.speed)

    shared = Shared()
    if args.vlm == "off":
        shared.goal = Goal(t=time.time(), source="local-only", reason="local-only exploration")
    threads = [
        threading.Thread(
            target=capture_loop,
            args=(shared, args.car_ip, args.capture_hz, args.capture_timeout),
            daemon=True,
        ),
        threading.Thread(target=perception_loop, args=(shared, args), daemon=True),
        threading.Thread(target=vlm_loop, args=(shared, args), daemon=True),
    ]
    for t in threads:
        t.start()
    controller_loop(shared, args)


if __name__ == "__main__":
    main()
