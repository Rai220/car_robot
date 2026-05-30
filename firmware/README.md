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
tools/reflex_drive.py --target "toy road roller / asphalt paver / steamroller"
```

The script auto-reexecs itself through `.venv/bin/python` when the venv exists,
so direct `tools/reflex_drive.py ...` launches use the installed local ML stack.

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
