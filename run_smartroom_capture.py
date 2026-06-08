#!/usr/bin/env python3
"""
Named launcher for the smartroom capture — delegates to capture.py.

All capture logic lives in capture.py; this entry point exists only for the
run_smartroom_capture.sh wrapper and the desktop/service shortcuts. The
--duration flag is forwarded through to capture.py.

Launch via run_smartroom_capture.sh so the project venv is used.
"""

import capture

if __name__ == "__main__":
    capture.main()
