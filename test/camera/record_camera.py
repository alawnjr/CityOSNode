#!/usr/bin/env python3
"""Take a single snapshot from the smartroom USB camera."""

import subprocess
from datetime import datetime

CAMERA = "/dev/v4l/by-id/usb-lihappe8_Corp._USB_2.0_Camera-video-index0"

output = f"snapshot_{datetime.now():%Y-%m-%d_%H-%M-%S}.jpg"
subprocess.run([
    "ffmpeg", "-y",
    "-f", "v4l2",
    "-input_format", "mjpeg",
    "-video_size", "640x480",
    "-i", CAMERA,
    "-frames:v", "1",
    output,
], check=True)
print(f"saved {output}")
