#!/usr/bin/env bash
# Drive / interact with the ESP32-CAM car over its HTTP API.
#
# The car's IP (printed over serial after it joins WiFi) must be supplied via
# the CAR_IP environment variable:
#     export CAR_IP=192.168.1.42
#     tools/car.sh forward 800
#
# Commands:
#   forward [ms]   backward [ms]   left [ms]   right [ms]    move, then auto-stop after ms (default 500)
#   stop                                                      stop motors
#   speed <0-8>                                               drive speed
#   trim  <-32..32>                                           steering trim
#   light <0-255>                                             front LED brightness
#   vflip <0|1>   hmirror <0|1>                               flip image vertically / mirror horizontally
#   photo [file.jpg]                                          grab a still (default car_photo.jpg)
#   status                                                    camera status JSON
#   stream                                                    print the MJPEG stream URL (open in browser/vlc)
#   raw var=<x>&val=<y>                                        send a raw /control query
set -euo pipefail

: "${CAR_IP:?set CAR_IP first, e.g. export CAR_IP=192.168.1.42}"
BASE="http://${CAR_IP}"
ctl() { curl -s -m 5 "${BASE}/control?$1" >/dev/null && echo "ok: $1"; }
move() { local dir="$1" ms="${2:-500}"; ctl "var=car&val=$dir"; sleep "$(awk "BEGIN{print $ms/1000}")"; ctl "var=car&val=5"; }

cmd="${1:-}"; shift || true
case "$cmd" in
  forward)  move 1 "${1:-500}" ;;
  backward|back) move 2 "${1:-500}" ;;
  left)     move 3 "${1:-500}" ;;
  right)    move 4 "${1:-500}" ;;
  stop)     ctl "var=car&val=5" ;;
  speed)    ctl "var=speed&val=${1:?0-8}" ;;
  trim)     ctl "var=trim&val=${1:?-32..32}" ;;
  turnspeed) ctl "var=turnspeed&val=${1:?0-255}" ;;
  light|flash) ctl "var=flash&val=${1:?0-255}" ;;
  vflip)    ctl "var=vflip&val=${1:?0|1}" ;;
  hmirror)  ctl "var=hmirror&val=${1:?0|1}" ;;
  photo)    out="${1:-car_photo.jpg}"; curl -s -m 10 "${BASE}/capture" -o "$out" && echo "saved $out" ;;
  status)   curl -s -m 5 "${BASE}/status"; echo ;;
  stream)   echo "MJPEG stream: ${BASE}:81/stream"; echo "control page: ${BASE}/" ;;
  raw)      ctl "${1:?var=...&val=...}" ;;
  *) sed -n '2,30p' "$0"; exit 1 ;;
esac
