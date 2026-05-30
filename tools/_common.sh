#!/usr/bin/env bash
# Shared helpers for car_robot firmware tools (PlatformIO backend).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIO="$PROJECT_ROOT/.venv/bin/pio"
ENV_NAME="esp32cam"
SKETCH_DEFAULT="$PROJECT_ROOT/firmware/VideoCar"
BAUD=115200

require_pio() {
  if [ ! -x "$PIO" ]; then
    echo "ERROR: PlatformIO not found at $PIO" >&2
    echo "  Install it:  python3 -m venv .venv && .venv/bin/pip install platformio" >&2
    exit 1
  fi
}

# Find the car's USB-serial port. CH340 shows up as cu.usbserial-* or cu.wchusbserial*.
detect_port() {
  ls /dev/cu.usbserial-* /dev/cu.wchusbserial* /dev/cu.SLAB_USBtoUART* 2>/dev/null | head -1 || true
}

require_port() {
  local p; p="$(detect_port)"
  if [ -z "$p" ]; then
    echo "ERROR: no ESP32 serial port found." >&2
    echo "  Check: (1) USB data cable (not charge-only), (2) power switch ON," >&2
    echo "         (3) plug directly into the Mac (not through a USB hub)," >&2
    echo "         (4) CH340 driver installed." >&2
    exit 1
  fi
  echo "$p"
}
