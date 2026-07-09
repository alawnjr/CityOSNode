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


def load_node_env():
    """Apply per-node overrides from <repo>/node.env (KEY=VALUE lines, # comments).

    node.env is gitignored — it's machine-local config, like calibration/ — so a
    node can pin e.g. SMARTROOM_CAMERA_SIZE=800x600 (a camera that only sustains
    30fps below full resolution) without diverging the shared code. The real
    environment always wins over the file.
    """
    path = Path(__file__).resolve().parent / "node.env"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


load_node_env()  # before the module-level auto-detection below reads the env


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
CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"


def camera_id(device=None):
    """Stable per-physical-camera identity: the /dev/v4l/by-id/ symlink stem
    without the -video-indexN suffix (e.g. usb-046d_0809_8633D7D7). Calibration
    files are keyed by this, so a swapped camera never inherits stale intrinsics.
    None for devices with no by-id identity (bare /dev/videoN)."""
    device = device or CAMERA
    name = Path(device).name
    if "-video-index" in name:
        return name.split("-video-index")[0]
    by_id = Path("/dev/v4l/by-id")
    if by_id.is_dir():
        try:
            target = Path(device).resolve()
            for link in by_id.iterdir():
                if "-video-index0" in link.name and link.resolve() == target:
                    return link.name.split("-video-index")[0]
        except OSError:
            pass
    return None


def load_calibration():
    """This camera's intrinsics from calibration/<camera_id>.json (written by
    calibrate_camera.py), ready to embed in metadata.json. If the calibration
    was done at a different resolution than we record at, fx/fy/cx/cy are scaled
    proportionally (distortion coefficients are resolution-invariant) and the
    original size is noted under 'scaled_from'. None when uncalibrated."""
    cam = camera_id()
    if cam is None:
        return None
    path = CALIBRATION_DIR / f"{cam}.json"
    try:
        cal = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    blob = {
        "camera_id": cal.get("camera_id", cam),
        "camera_matrix": cal["camera_matrix"],
        "dist_coeffs": cal["dist_coeffs"],
        "image_size": cal["image_size"],
        "rms": cal.get("rms"),
        "calibrated_at": cal.get("calibrated_at"),
        "file": path.name,
    }
    cal_w, cal_h = cal["image_size"]
    if (cal_w, cal_h) != (CAMERA_WIDTH, CAMERA_HEIGHT) and cal_w and cal_h:
        sx, sy = CAMERA_WIDTH / cal_w, CAMERA_HEIGHT / cal_h
        m = [row[:] for row in cal["camera_matrix"]]
        m[0][0] *= sx  # fx
        m[0][2] *= sx  # cx
        m[1][1] *= sy  # fy
        m[1][2] *= sy  # cy
        blob["camera_matrix"] = m
        blob["image_size"] = [CAMERA_WIDTH, CAMERA_HEIGHT]
        blob["scaled_from"] = [cal_w, cal_h]
    return blob


def load_extrinsics():
    """This camera's pose in the room/tag frame (calibrate_extrinsics.py), for
    embedding in metadata.json. None when not extrinsically calibrated."""
    cam = camera_id()
    if cam is None:
        return None
    path = CALIBRATION_DIR / f"{cam}.extrinsics.json"
    try:
        ext = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    keys = ("camera_id", "frame", "tag", "rvec", "tvec_mm", "rotation_cam_to_room",
            "camera_position_mm", "reprojection_error_px", "calibrated_at")
    return {k: ext[k] for k in keys if k in ext}


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
    # The camera can take a few seconds to actually be released (the web page's
    # preview hands it over just before this runs, and a glitchy USB link makes
    # the v4l2 close slow) — retry briefly on "busy" instead of failing the
    # whole recording. Real errors still raise, with ffmpeg's stderr shown.
    for attempt in range(5):
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        # busy = preview handoff race; "no such" = the camera dropped off the
        # USB bus and is re-enumerating (takes a few seconds on a flaky link).
        err = (proc.stderr or "").lower()
        if attempt < 4 and ("busy" in err or "no such" in err or "input/output error" in err):
            print(f"camera not ready — retrying ({attempt + 1}/4)…", file=sys.stderr)
            time.sleep(3)
            continue
        sys.stderr.write(proc.stderr or "")
        raise subprocess.CalledProcessError(proc.returncode, command)


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
    # Intrinsics from calibrate_camera.py, when this camera has been calibrated.
    # Videos stay raw — downstream analysis uses these to undistort as needed.
    calibration = load_calibration()
    if calibration is not None:
        metadata["streams"]["camera_main"]["calibration"] = calibration
    extrinsics = load_extrinsics()
    if extrinsics is not None:
        metadata["streams"]["camera_main"]["extrinsics"] = extrinsics
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
