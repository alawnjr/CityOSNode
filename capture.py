#!/usr/bin/env python3
"""
Capture video from the smartroom node camera into one recording.

Video-only for now (audio and the custom I2C/PCB sensors were removed). Records
for DURATION_SECONDS and writes a recording folder under data/ in the same
layout as sample_dataset/:

    data/day_NN_YYYY-MM-DD/rec_YYYYMMDD_NNN/
        metadata.json
        streams/
            camera_main.mp4              (USB camera, h264)
            camera_main_timestamps.csv   (frame_index, timestamp_seconds)

The same code runs on every node (smartroom1, smartroom2, ...), but each node
has a different camera, so the device/format are env-configurable and otherwise
auto-detected:

    SMARTROOM_CAMERA       v4l2 device (default: first USB camera by-id, else /dev/video0)
    SMARTROOM_CAMERA_SIZE  WxH (default: 1280x720)
    SMARTROOM_CAMERA_FPS   frames/sec (default: 30)

Run on the Pi:  python capture.py            # 30s (default)
                python capture.py -d 60      # 60s
"""

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DEFAULT_DURATION = 30
DURATION_SECONDS = DEFAULT_DURATION  # overridden by --duration in main()
DATA_DIR = Path(__file__).resolve().parent / "data"

TIMESTAMP_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def detect_camera():
    """Pick the camera v4l2 device.

    SMARTROOM_CAMERA wins if set. Otherwise auto-detect the first USB camera via
    its stable /dev/v4l/by-id/ symlink (so each node — different cameras — just
    works off the one shared codebase), falling back to /dev/video0.
    """
    override = os.environ.get("SMARTROOM_CAMERA")
    if override:
        return override
    by_id = Path("/dev/v4l/by-id")
    if by_id.is_dir():
        cams = sorted(p for p in by_id.iterdir() if p.name.endswith("-video-index0"))
        if cams:
            return str(cams[0])
    return "/dev/video0"


CAMERA = detect_camera()
_size = os.environ.get("SMARTROOM_CAMERA_SIZE", "1280x720")
CAMERA_WIDTH, CAMERA_HEIGHT = (int(v) for v in _size.lower().split("x"))
CAMERA_FPS = int(os.environ.get("SMARTROOM_CAMERA_FPS", "30"))


def make_recording_dir():
    """Pick the day_NN_DATE / rec_DATE_NNN folder, matching sample_dataset."""
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    compact = now.strftime("%Y%m%d")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing_days = sorted(DATA_DIR.glob("day_*"))
    day_dir = next((d for d in existing_days if d.name.endswith(date)), None)
    if day_dir is None:
        day_dir = DATA_DIR / f"day_{len(existing_days) + 1:02d}_{date}"

    rec_num = len(list(day_dir.glob("rec_*"))) + 1
    rec_dir = day_dir / f"rec_{compact}_{rec_num:03d}"
    (rec_dir / "streams").mkdir(parents=True, exist_ok=True)
    return rec_dir


def record_camera(path):
    timestamp_filter = (
        f"drawtext=fontfile={TIMESTAMP_FONT}:"
        "text=%{localtime\\\\:%Y-%m-%d %H-%M-%S}:"
        "x=12:y=12:fontsize=22:fontcolor=white:"
        "box=1:boxcolor=black@0.55:boxborderw=8"
    )
    source = [
        "ffmpeg", "-y",
        "-f", "v4l2", "-framerate", str(CAMERA_FPS),
        "-input_format", "mjpeg", "-video_size", f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
    ]
    preview_path = os.environ.get("SMARTROOM_PREVIEW")
    if preview_path:
        # Record to file (with overlay) AND write the latest frame (~5 fps) to a
        # single jpg so the web page can show the camera while recording. The
        # duration limit goes on the INPUT so BOTH outputs stop together (an
        # unbounded second output would keep ffmpeg running forever).
        command = source + [
            "-t", str(DURATION_SECONDS), "-i", CAMERA,
            "-map", "0:v:0", "-vf", timestamp_filter,
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(path),
            "-map", "0:v:0", "-r", "5", "-update", "1", "-y", preview_path,
        ]
    else:
        command = source + [
            "-i", CAMERA, "-t", str(DURATION_SECONDS),
            "-vf", timestamp_filter,
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(path),
        ]
    subprocess.run(command, check=True)


def write_camera_timestamps(path, fps):
    frame_count = DURATION_SECONDS * fps
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_seconds"])
        for i in range(frame_count):
            writer.writerow([i, f"{i / fps:.6f}"])
    return frame_count


def write_metadata(rec_dir, start_time, end_time, frame_count):
    metadata = {
        "recording_id": rec_dir.name,
        "node": socket.gethostname(),  # which Pi recorded this (smartroom1/smartroom2/...)
        "space": "smart_room_1",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": DURATION_SECONDS,
        "schema_version": "0.1",
        "streams": {
            "camera_main": {
                "modality": "video",
                "path": "streams/camera_main.mp4",
                "codec": "h264",
                "device": CAMERA,
                "resolution": [CAMERA_WIDTH, CAMERA_HEIGHT],
                "fps": CAMERA_FPS,
                "frame_count": frame_count,
                "timestamps_path": "streams/camera_main_timestamps.csv",
            },
        },
    }
    (rec_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main():
    global DURATION_SECONDS
    parser = argparse.ArgumentParser(description="Capture video from the smartroom node camera.")
    parser.add_argument("-d", "--duration", type=int, default=DEFAULT_DURATION,
                        help=f"Recording length in seconds (default: {DEFAULT_DURATION}).")
    DURATION_SECONDS = parser.parse_args().duration

    rec_dir = make_recording_dir()
    streams = rec_dir / "streams"
    print(f"Recording {DURATION_SECONDS}s from {CAMERA} -> {rec_dir}", file=sys.stderr)

    start_time = datetime.now().astimezone()
    record_camera(streams / "camera_main.mp4")  # blocks for DURATION_SECONDS
    end_time = datetime.now().astimezone()

    frame_count = write_camera_timestamps(streams / "camera_main_timestamps.csv", CAMERA_FPS)
    write_metadata(rec_dir, start_time, end_time, frame_count)
    print(f"Done -> {rec_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
