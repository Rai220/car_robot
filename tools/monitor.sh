#!/usr/bin/env bash
# Open the serial monitor (115200 baud). Usage: tools/monitor.sh [port]
# This is where the car prints its IP address after connecting to WiFi.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_pio

PORT="${1:-$(require_port)}"
echo ">> serial monitor on $PORT @ ${BAUD} (Ctrl-C to quit)"
cd "$PROJECT_ROOT"
"$PIO" device monitor -p "$PORT" -b "$BAUD"
