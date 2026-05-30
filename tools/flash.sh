#!/usr/bin/env bash
# Compile + upload a sketch to the car.
# Usage: tools/flash.sh [sketch_dir] [port]   (defaults: firmware/VideoCar, auto-detected port)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_pio

SKETCH="${1:-$SKETCH_DEFAULT}"
PORT="${2:-$(require_port)}"

echo ">> compiling + uploading $SKETCH to $PORT"
cd "$PROJECT_ROOT"
PLATFORMIO_SRC_DIR="$SKETCH" "$PIO" run -e "$ENV_NAME" -t upload --upload-port "$PORT"
echo ">> done. See the car's IP:  tools/monitor.sh"
