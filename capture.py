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
    SMARTROOM_CAMERA_SIZE  WxH (default: largest MJPG mode <=1280 wide, e.g. C920->1280x720, lihappe8->640x480)
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


def detect_camera_size(device, cap_width=1280):
    """Pick the capture resolution.

    SMARTROOM_CAMERA_SIZE (WxH) wins if set. Otherwise query the device's MJPG
    modes and pick the largest with width <= cap_width — so a wide-FOV camera
    records at 1280x720 while a cheaper 640x480-only camera records at 640x480,
    off the one shared codebase. The cap keeps software H.264 encode load sane on
    a Pi 3/4 (no 1080p). Falls back to 1280x720 if detection fails.
    """
    override = os.environ.get("SMARTROOM_CAMERA_SIZE")
    if override:
        w, h = (int(v) for v in override.lower().split("x"))
        return w, h
    try:
        out = subprocess.run(
            ["v4l2-ctl", "-d", device, "--list-formats-ext"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        sizes, in_mjpg = [], False
        for line in out.splitlines():
            s = line.strip()
            if "]:" in s and "'" in s:                  # format header, e.g. [2]: 'MJPG' ...
                in_mjpg = "MJPG" in s
            elif in_mjpg and s.startswith("Size: Discrete"):
                w, h = (int(v) for v in s.split()[-1].split("x"))
                sizes.append((w, h))
        usable = [(w, h) for w, h in sizes if w <= cap_width]
        if usable:
            return max(usable, key=lambda wh: wh[0] * wh[1])
    except Exception as error:  # noqa: BLE001
        print(f"camera: size auto-detect failed ({error}); using 1280x720", file=sys.stderr)
    return 1280, 720


CAMERA = detect_camera()
CAMERA_WIDTH, CAMERA_HEIGHT = detect_camera_size(CAMERA)
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


def probe_frame_times(mp4_path):
    """Per-frame presentation times (seconds from the recording's start), ascending.

    Read from the *finished* mp4 with ffprobe, so we get one entry per frame the
    camera actually delivered — the USB cameras run variable-rate and deliver far
    fewer frames than the nominal fps, so this is the only accurate source. Returns
    [] on any failure (missing ffprobe, unreadable file) so the caller can fall
    back to the nominal grid.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "frame=best_effort_timestamp_time",
             "-of", "csv=p=0", str(mp4_path)],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    times = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line == "N/A":
            continue
        try:
            times.append(float(line))
        except ValueError:
            continue
    times.sort()  # ffprobe emits decode order; sort to presentation order
    if times and times[0] != 0:  # some containers start PTS != 0 — rebase to t=0
        base = times[0]
        times = [t - base for t in times]
    return times


def write_camera_timestamps(path, mp4_path, fps):
    """Write one row (frame_index, timestamp_seconds) per ACTUAL encoded frame.

    Real per-frame timestamps come from the recorded mp4 (see probe_frame_times);
    only if that yields nothing do we fall back to the old synthetic i/fps grid.
    Returns the real frame count so metadata.json matches the video.
    """
    times = probe_frame_times(mp4_path)
    if not times:
        times = [i / fps for i in range(DURATION_SECONDS * fps)]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_seconds"])
        for i, t in enumerate(times):
            writer.writerow([i, f"{t:.6f}"])
    return len(times)


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

    mp4_path = streams / "camera_main.mp4"
    start_time = datetime.now().astimezone()
    record_camera(mp4_path)  # blocks for DURATION_SECONDS
    end_time = datetime.now().astimezone()

    frame_count = write_camera_timestamps(streams / "camera_main_timestamps.csv", mp4_path, CAMERA_FPS)
    write_metadata(rec_dir, start_time, end_time, frame_count)
    print(f"Done -> {rec_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
