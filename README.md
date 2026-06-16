# car_robot

Experiments toward a **universal robot driver** — an AI that can pilot almost any
robot regardless of how it's built — running on a ~$20 ESP32-CAM toy car.

![Live dashboard while the car searches for a target](docs/media/robocar.gif)

*`tools/dashboard.py` observability UI mid-task: the raw 320×240 camera (left),
the Depth-Anything-V2 free-space heatmap with the look-ahead band and floor
reference (center), and the live VLM decision log with per-zone openness and
on-rug confidence (right).*

<!-- The full-resolution screen capture (robocar.mp4, ~60 MB) is intentionally
     kept out of git; this GIF is a trimmed, downscaled excerpt of it. -->

Background & motivation (RU): [«Отпускной пост про DIY-роботов»](https://t.me/robofuture/155).

## Idea

One model isn't enough. Like a person — Kahneman's *Thinking, Fast and Slow* — the
robot needs a **fast** reflexive system and a **slow**, deliberate one, and here a
**third** system that watches both and improves them over time. The bet: just as
large language models reshaped NLP, a general "robot driver" will do the same for
robotics — and the path there is composing fast / slow / self-improving layers
rather than training one monolithic end-to-end policy.

## Three systems

1. **System 1 — fast & reflexive** — `tools/reflex_drive.py` + `tools/depth_perception.py`.
   Depth-Anything-V2 splits each frame into left / center / right, scores where it's
   most open, and steers in real time. Dumb and fast: it owns the wheels and
   close-range obstacle avoidance.

2. **System 2 — slow & smart** — Gemini 2.5 Flash / Gemini Robotics-ER, or a local
   Qwen2.5-VL / FastVLM. A vision-action-language model reads the recent photo
   history and the task prompt, recognizes complex situations, and emits JSON goals
   and route bias. It *guides* the fast loop; it never owns continuous motor control.

3. **System 3 — self-improvement** — `tools/self_improve_loop.sh` + `tools/evolve_loop.sh`.
   A code model (Codex) reads logs and annotated frames every few minutes and may
   rewrite the prompts and logic of Systems 1–2 — *a psychotherapist for the robot*,
   fixing deep mechanisms rather than momentary decisions. It is bounded by a file
   allowlist, a sandbox, before/after source-hash checks, and a unittest suite it is
   not allowed to edit.

See [docs/self_improvement.md](docs/self_improvement.md) for the layer-3 harness and run commands.

## Hardware

A cheap ESP32-CAM toy car (camera + WiFi + I²C motor driver), ~1800 ₽ / ~$20.
Firmware lives in `firmware/VideoCar/`; see [firmware/README.md](firmware/README.md)
for the build/flash toolchain and the HTTP control API.
