#!/bin/sh
# Launch the smartroom capture using the project venv.
# Usage: ./run_smartroom_capture.sh [duration_seconds]   (default: 30)
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

PYTHON="python3"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
fi

exec "$PYTHON" "$SCRIPT_DIR/run_smartroom_capture.py" --duration "${1:-30}"
