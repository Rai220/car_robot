# ESP32-CAM Video Car — dev setup

Toolchain and sketches for the Keyestudio KS5017 ESP32-CAM smart car.
Full vendor docs are mirrored under [`../docs`](../docs/docs/Tutorial.html).

## Layout

```
firmware/
  VideoCar/              main firmware (camera + WiFi + motor control)
    VideoCar.ino
    app_server.h         HTTP server + control API + web UI
    SetMotor.h           I2C motor helpers
    wifi_config.h        YOUR WiFi creds (git-ignored)
    wifi_config.example.h  template
    lib/                 dl_lib headers bundled by the vendor
  examples/              standalone learning sketches
    Blink/ breathing_light/ motor/
tools/                   build / flash / monitor / drive helpers
```

## Toolchain

Build backend is **PlatformIO** (in a local `.venv`), driven by `platformio.ini`.
Board: **AI-Thinker ESP32-CAM** (`board = esp32cam`), `framework = arduino`,
`platform = espressif32@^6.9.0` (arduino-esp32 2.0.x — the series this firmware
targets). The CH340 USB-serial driver is required for flashing.

> Why PlatformIO and not arduino-cli? In this environment `downloads.arduino.cc`
> is unreachable (HTTP 403), so arduino-cli can't fetch its `ctags`/discovery
> tools. PlatformIO uses its own registry, which works here.

Re-create the toolchain from scratch if needed:

```bash
python3 -m venv .venv && .venv/bin/pip install platformio
```

## WiFi

Credentials live in `firmware/VideoCar/wifi_config.h` (git-ignored).
Start from the template:

```bash
cp firmware/VideoCar/wifi_config.example.h firmware/VideoCar/wifi_config.h
# edit it: set ssid/password
```

- `ap = false` → **STA**: car joins your router; its IP is printed over serial @115200.
  ESP32 only sees **2.4 GHz** networks (not 5 GHz).
- `ap = true` → **AP**: car makes its own hotspot at `192.168.4.1` (only 1 client).

## Build / flash / monitor

```bash
tools/detect.sh                 # is the car's serial port visible?
tools/build.sh                  # compile firmware/VideoCar
tools/flash.sh                  # compile + upload (auto-detects port)
tools/monitor.sh                # serial @115200 — read the car's IP here
```

Flash a different sketch: `tools/flash.sh firmware/examples/Blink`.

## Interact over HTTP

After the car joins WiFi, grab its IP from the serial monitor, then:

```bash
export CAR_IP=192.168.1.42      # the IP from the serial log
tools/car.sh forward 800        # drive forward 0.8s then stop
tools/car.sh left 300
tools/car.sh speed 6            # 0..8
tools/car.sh light 120          # front LED 0..255
tools/car.sh photo shot.jpg     # still frame
tools/car.sh stream             # prints the MJPEG URL for browser/VLC
tools/car.sh status
```

## Autonomy

`tools/reflex_drive.py` is the current hybrid autonomy entrypoint. It uses two
separate loops:

- a fast local loop on the Mac captures frames, runs local perception, and owns
  every motor command;
- a slower VLM loop only updates high-level goal bias, so VLM latency cannot
  freeze collision avoidance or directly make the car spin.
- the local loop keeps a compact egocentric exploration map: approximate current
  cell, visited cells, recently blocked cells, and last target observations. The
  map biases exploration toward open unvisited directions, commits to a selected
  frontier for a few seconds, and triggers a larger loop-escape maneuver when the
  car keeps recovering in the same area.

Start with dry-run tuning; it reads the live camera but does not move motors:

```bash
tools/reflex_drive.py --dry-run --vlm off --secs 10 --debug
tools/reflex_drive.py --dry-run --target "yellow toy road roller" --secs 20 --debug
```

Drive with the same controller after the dry-run decisions look sane:

```bash
tools/reflex_drive.py --target "toy road roller / steamroller / asphalt compactor"
```

For a camera-only controller without local depth estimation, use the Gemini
pulse driver. Gemini receives the current camera view plus recent action history
and returns one bounded command at a time, for example "forward 450 ms" or
"turn left 15 degrees"; the script stops the car after every pulse before asking
Gemini again.

```bash
tools/gemini_drive.py --target "toy road roller / steamroller / asphalt compactor" --debug
```

For the Gemini pulse driver, target matching is intentionally strict: a generic
construction toy is not enough. It should only stop for a road roller/steamroller
with a visible smooth roller drum.

Both autonomy scripts auto-reexec through `.venv/bin/python` when the venv
exists, so direct `tools/reflex_drive.py ...` and `tools/gemini_drive.py ...`
launches use the installed local ML stack.

Useful safety knobs:

| Option | Effect |
|---|---|
| `--speed 2..3` | normal forward speed; default is 2 for carpet edges |
| `--min-drive-speed 2..3` | minimum ordinary arc-drive power during exploration |
| `--approach-speed 1..2` | lower speed cap only once the target is large/very close |
| `--escape-speed 3..5` | stronger reverse/recovery power |
| `--turnspeed 60..90` | firmware in-place turn power for recovery nudges |
| `--target-max-trim`, `--target-far-max-trim` | cap steering while the target is visible so it stays in camera view |
| `--target-push-min-open` | keep pushing toward a visible target over carpet edges unless the center is truly blocked |
| `--map-weight 0..1` | how strongly the local exploration map biases route choice; default is 0.65 |
| `--explore-commit-sec` | how long to keep a chosen exploration direction before re-planning |
| `--loop-*` | loop detector / larger escape maneuver when the car is stuck in one area |
| `--vlm off` | pure local exploration / obstacle avoidance |
| `--perception heuristic` | fallback when the depth model is unavailable |
| `--no-stop-on-large-target` | keep moving even if VLM says the target is close |
| `--block`, `--slow`, `--open` | local openness thresholds for recovery vs arc driving |
| `--frame-log`/`--no-frame-log` | write a JSONL decision stream + annotated frames to `logs/` (on by default) |
| `--frame-log-sec`, `--frame-log-keep`, `--no-annotate-frames` | heartbeat / ring size / raw-frame toggle for the run log |

While driving, `reflex_drive.py` records `logs/reflex_<ts>.jsonl` (one decision
per tick) and annotated frames in `logs/frames/` (L/C/R openness + action + goal +
map overlay). Both are gitignored and are what the layer-3 self-improvement loop
reads back — see [docs/self_improvement.md](../docs/self_improvement.md). A fast
`unittest` harness guards the controller logic: `CAR_ROBOT_NO_VENV_REEXEC=1
.venv/bin/python -m unittest discover -s tests`.

Useful Gemini pulse-driver knobs:

| Option | Effect |
|---|---|
| `--speed 2 --max-speed 3` | modest carpet-capable forward power |
| `--max-duration-ms 650` | longest ordinary forward pulse before a fresh camera check |
| `--ms-per-90 900` | turn calibration used to convert degrees into milliseconds |
| `--max-turn-degrees 35` | prevents hard spins that throw the target out of view |
| `--frames 2 --history 10` | visual/action memory sent to Gemini each step |
| `--repeat-guard` | local guard against repeated forward commands when the image barely changes |
| `--target-memory-steps 8` | local recovery window after the target was recently visible |
| `--reacquire-turn-degrees 14` | small local scan angle after the target disappears |
| `--arrival-medium-forward-steps 3` | stop after sustained centered approach even if Gemini never calls the target `large` |
| `--search-turns-before-forward 4` | after several lost-target scans, force relocation instead of spinning in place |
| `--search-forward-burst 3` | number of forced forward pulses used to reach a new viewpoint |
| `--explore-forward-ms 500` | duration of each forced relocation pulse |
| `--allow-reverse` | opt in to reverse pulses; disabled by default to avoid forward/back oscillation |

### Live dashboard (watch what the car sees)

`tools/dashboard.py` is an observe-only browser dashboard for understanding the
stack live. It runs its own capture + perception + VLM loops (it does **not**
drive autonomously) and shows three panels side by side:

- **Camera** — the raw forward-facing frame.
- **Depth** — the Depth Anything V2 heatmap (near = warm/red) plus the
  left/center/right free-space openness scores, drawn against the same
  `block`/`slow`/`open` thresholds `reflex_drive.py` uses to choose recover-vs-arc.
  It also exposes the raw nearness `ratio` (near/floor) each zone's openness comes
  from and overlays the exact pixels used — the cyan look-ahead band and the
  magenta floor-reference patch — so you can see *why* a zone reads open or blocked.
- **VLM** — the current Gemini decision (action/goal + reason + latency + a short
  rolling history), so you can see *why* it would steer.

```bash
set -a; source .env; set +a
tools/dashboard.py                                    # camera + depth + Gemini explore-VLM
tools/dashboard.py --target "yellow toy road roller"  # richer seek-style VLM output
tools/dashboard.py --vlm local                         # local FastVLM instead of Gemini (no cloud)
tools/dashboard.py --no-depth                          # skip the local depth model
tools/dashboard.py --no-vlm                            # camera + depth only (no VLM)
tools/dashboard.py --port 8080 --vlm-sec 1.5
```

Open `http://127.0.0.1:8000`. Optional manual-drive buttons (or arrow keys /
space) send short pulse commands to the car so you can move it around and watch
depth and the VLM react; the dashboard never moves the wheels on its own. Like
the other tools it auto-reexecs through `.venv/bin/python`.

| Option | Effect |
|---|---|
| `--vlm gemini\|local\|off` | VLM panel backend: cloud Gemini, local mlx-vlm model, or off |
| `--local-vlm-model-dir` | mlx-vlm model dir for `--vlm local` (default `models/fastvlm-0.5b-bf16`) |
| `--target "..."` | switch the VLM panel from explore (`action`) to seek (`bearing/size/motion/safe_forward/...`) |
| `--model gemini-2.5-flash` | Gemini model (when `--vlm gemini`) |
| `--vlm-sec 2.0` | seconds between VLM calls (lower = fresher, more API/compute) |
| `--no-vlm` / `--no-depth` | run without the VLM / without the local depth model |
| `--block`/`--slow`/`--open` | openness thresholds shown on the depth bars (mirror `reflex_drive.py`) |
| `--port`/`--host` | where the dashboard is served |

The **local VLM** backend reuses `tools/vlm_local.py`, which loads any mlx-vlm
model (Apple FastVLM, Qwen2.5-VL, ...). FastVLM is near-real-time on Apple Silicon
(~1 s/decision, fully local) but the 0.5B variant is weak at the strict navigation
JSON — use it to gauge latency and swap to a larger model for quality. One-time setup:

```bash
.venv/bin/pip install timm                            # FastVLM's FastViTHD encoder needs timm
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('mlx-community/FastVLM-0.5B-bf16', local_dir='models/fastvlm-0.5b-bf16')"
# quick self-test (latency + JSON sanity) on a still frame:
.venv/bin/python tools/vlm_local.py start_view.jpg models/fastvlm-0.5b-bf16
```

### Closed-loop control with Gemini Robotics-ER (`--vlm robotics`)

`tools/dashboard.py --vlm robotics` runs a **closed-loop** task mode where
**Gemini Robotics-ER 1.6** (`gemini-robotics-er-1.6-preview`, via the same API key)
drives the car directly:

- ER points at the target (`target_point [y,x]`, shown as a crosshair on the camera)
  and emits a **parametric command** itself — `forward/back` with `duration_ms`, or
  `turn_left/right` with `turn_degrees`, plus `speed`/`trim`.
- The command is executed as a timed pulse, then the **actual motion is measured from
  the camera** (no encoders): yaw degrees from the horizontal scene shift, forward
  progress / stall from the frame change. That feedback is sent back to ER on the next
  step, and the `ms-per-degree` turn calibration **self-tunes** live (shown in the UI).
- The VLM panel shows the command vs the measured result and the live calibration; the
  depth emergency bumper, arrival→stop→silence, Stop button, and dry-run all still apply.

```bash
set -a; source .env; set +a
tools/dashboard.py --vlm robotics                      # ER-1.6 closed-loop (tasks burn API only while running)
tools/dashboard.py --vlm robotics --ms-per-90 800 --rb-max-deg 30
```

Useful knobs: `--robotics-model`, `--ms-per-90` (initial turn calibration),
`--cam-hfov` (yaw estimate), `--turnspeed`, `--rb-max-ms`/`--rb-default-ms` (forward
pulse), `--rb-max-deg`, `--bump` (depth bumper), `--settle-sec`. ER calls happen only
while a task is running (Start→Stop).

### Local-AI room exploration (`--vlm explore`)

`tools/dashboard.py --vlm explore` lets the car roam a room **on its own** with a
**local** brain; cloud Gemini only issues occasional high-level directives.

- **Local controller owns motors + safety** (fast loop): obstacle distance from the
  Depth Anything model (`--keep-clear` makes it slow/steer away *early*, not at the
  last 25 cm), an on-rug detector (`carpet_scores`: HSV hue/sat of the look-ahead vs
  the rug under the wheels — keeps it **on the carpet**, `--carpet-block`/`--carpet-edge`),
  and a frontier exploration map (visited cells, frontier bonus, loop escape) so it
  covers the room instead of circling. Drivable = clear **and** on-rug per L/C/R zone.
- **Gemini = slow supervisor**: every `--vlm-sec` it returns ONE directive —
  `explore` / `search_here` (slow + scan) / `move_on` (go to a new area) /
  `turn_around` / `stop` — which only *biases* the local policy; it never drives.
- The dashboard shows the **on-rug bars** (next to the depth bars), the current
  **directive**, and the exploration state (visited cells / bias / looping).

```bash
set -a; source .env; set +a
tools/dashboard.py --vlm explore                       # roam; needs depth (auto-enabled)
tools/dashboard.py --vlm explore --keep-clear 0.5 --carpet-block 0.35 --drive-speed 2
```

Press **▶ Запустить** to start roaming. If the task text names a target, the supervisor
(**Gemini Robotics-ER**) flags `target_visible`/`target_close` and **points at the target**
(`target_point`, shown as a crosshair). The local loop then **actively steers to keep the
target centered** (visual servo on the point, polled faster via `--approach-vlm-sec`),
**crawls** up to it without steering away, and **stops the instant depth shows it is
close** (`--arrive-open`) — the stop is decided locally, not by the slow supervisor, so it
can't coast past. The controller also **learns a constant drift trim** from the measured
forward yaw (`--drift-gain`, capped by `--max-drift-trim`) so a car that doesn't roll
straight is corrected automatically.
**■ Стоп** / manual / a `stop` directive also halt. Use **dry-run** first to watch the
on-rug bars, chosen direction, and directive without moving. Tune `--keep-clear`, `--carpet-block/edge`, `--drive-speed`,
and the frontier knobs (`--frontier-bonus`, `--revisit-penalty`, `--explore-commit-sec`,
`--loop-*`).

### Raw HTTP API

| Request | Effect |
|---|---|
| `GET /` | control web UI |
| `GET /control?var=car&val=1..5` | 1=fwd 2=back 3=left 4=right 5=stop |
| `GET /control?var=speed&val=0..8` | drive speed |
| `GET /control?var=trim&val=-32..32` | steering trim |
| `GET /control?var=flash&val=0..255` | front LED brightness |
| `GET /control?var=framesize&val=0..6` | resolution |
| `GET /control?var=quality&val=10..63` | JPEG quality |
| `GET /capture` | single JPEG |
| `GET /status` | camera status JSON |
| `GET :81/stream` | MJPEG video stream |

## Troubleshooting

- **No serial port / `detect.sh` finds nothing:** use a USB **data** cable (not
  charge-only), set the power **switch to ON**, plug **directly** into the Mac
  (not through a hub), and confirm the CH340 driver is installed.
- **Upload fails:** the board has an auto-download circuit, so no boot button is
  normally needed; retry, and lower the upload speed if it stalls.
- **Can't reach the car over HTTP:** confirm the IP from serial, and make sure the
  Mac is on the **same 2.4 GHz network** (STA mode).
