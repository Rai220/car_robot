#!/usr/bin/env bash
# Check whether the car's USB-serial port is visible.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PORT="$(detect_port)"
if [ -n "$PORT" ]; then
  echo "OK  found serial port: $PORT"
  [ -x "$PIO" ] && "$PIO" device list 2>/dev/null || true
else
  echo "NOT FOUND  no ESP32 serial port."
  echo "  Check: data cable (not charge-only), power switch ON, plug direct (not via hub), CH340 driver."
fi
