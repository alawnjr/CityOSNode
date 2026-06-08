#!/bin/sh
set -eu

DURATION="${1:-60}"
OUTPUT_DIR="${SMARTROOM_OUTPUT_DIR:-$HOME/CityOS/data}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON="${SMARTROOM_PYTHON:-python3}"

if [ -x "$HOME/CityOS/.venv/bin/python" ]; then
  PYTHON="$HOME/CityOS/.venv/bin/python"
fi

exec "$PYTHON" "$SCRIPT_DIR/run_smartroom_capture.py" \
  --duration "$DURATION" \
  --output-dir "$OUTPUT_DIR"
