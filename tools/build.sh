#!/usr/bin/env bash
# Compile a sketch. Usage: tools/build.sh [sketch_dir]   (default: firmware/VideoCar)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_pio

SKETCH="${1:-$SKETCH_DEFAULT}"
echo ">> compiling $SKETCH"
cd "$PROJECT_ROOT"
PLATFORMIO_SRC_DIR="$SKETCH" "$PIO" run -e "$ENV_NAME"
