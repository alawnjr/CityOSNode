#!/usr/bin/env python3
"""
Intrinsic camera calibration for a smartroom node, using a printed checkerboard.

Run ON the Pi (headless — no display needed) with the venv python:

    ~/CityOS/.venv/bin/python ~/CityOS/calibrate_camera.py

Close the node's live-view browser tab first (the camera is single-access; the
web page releases it a few seconds after the last viewer leaves). Then hold a
printed checkerboard in front of the camera and move it slowly around the frame
— center, all four corners, near/far, tilted — while the script collects views.
Progress is printed; each accepted view also saves a corner-overlay JPG under
calibration/debug/<camera-id>/ so detection quality can be checked afterwards.

The result is written to calibration/<camera-id>.json, keyed by the camera's
stable /dev/v4l/by-id/ serial name — so swapping in a different camera never
reuses stale intrinsics. capture.py embeds this file's values into each
recording's metadata.json (videos themselves stay raw; undistortion happens
downstream).

Board default: 9x6 INNER corners (a 10x7-square board), 25 mm squares —
override with --cols/--rows/--square-mm if using a different print.
"""

import argparse
import datetime as dt
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# Same device/size detection (and SMARTROOM_CAMERA* env overrides) as recording,
# so calibration runs at exactly the resolution recordings use.
import capture

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_ROOT / "calibration"


def open_camera(device: str, width: int, height: int):
    """VideoCapture on the v4l2 device at the recording resolution (MJPG).
    Friendly retry loop for 'device busy' — the live preview releases the camera
    a few seconds after its last viewer disconnects."""
    # OpenCV 5's V4L2 backend can't open by device *name* (the by-id symlink) —
    # resolve to the real /dev/videoN and open by integer index instead.
    real = Path(device).resolve()
    index = int(str(real).removeprefix("/dev/video")) if str(real).startswith("/dev/video") else device
    for attempt in range(6):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
        if attempt == 0:
            print("camera busy — close the live-view browser tab; retrying for ~30s…",
                  file=sys.stderr)
        time.sleep(5)
    print(f"ERROR: could not open {device}. Is the live view or a recording running?",
          file=sys.stderr)
    return None


def _atomic_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def collect_views(cap, pattern, n_views, min_move_px, debug_dir: Path):
    """Grab frames until n_views diverse checkerboard detections are accepted.
    Returns (object_points_list, image_points_list, image_size)."""
    cols, rows = pattern
    # One canonical board model in board coordinates (Z=0 plane); square size is
    # applied by the caller (calibrateCamera only needs a consistent scale).
    board = np.zeros((cols * rows, 3), np.float32)
    board[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints, imgpoints, centroids = [], [], []
    image_size = None
    last_status = 0.0
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"collecting {n_views} views — move the board around the frame "
          "(center, corners, near, far, tilted)…", file=sys.stderr)
    while len(imgpoints) < n_views:
        ok, frame = cap.read()
        if not ok:
            print("frame grab failed; retrying…", file=sys.stderr)
            time.sleep(0.2)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        # Detect on a half-scale copy (fast enough for the Pi 3), then refine the
        # corner positions at full resolution for calibration accuracy.
        small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        found, corners = cv2.findChessboardCorners(
            small, (cols, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK)
        now = time.monotonic()
        if not found:
            if now - last_status > 3:
                print(f"  … board not visible ({len(imgpoints)}/{n_views})", file=sys.stderr)
                last_status = now
            continue

        corners = corners * 2.0  # back to full-res coordinates
        corners = cv2.cornerSubPix(
            gray, corners.astype(np.float32), (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Only accept views meaningfully different from the ones we have, so the
        # calibration sees diverse board poses instead of 15 copies of one pose.
        centroid = corners.reshape(-1, 2).mean(axis=0)
        if centroids and min(np.linalg.norm(centroid - c) for c in centroids) < min_move_px:
            if now - last_status > 3:
                print(f"  … seen — move the board to a new position ({len(imgpoints)}/{n_views})",
                      file=sys.stderr)
                last_status = now
            continue

        centroids.append(centroid)
        objpoints.append(board)
        imgpoints.append(corners)
        overlay = frame.copy()
        cv2.drawChessboardCorners(overlay, (cols, rows), corners, True)
        cv2.imwrite(str(debug_dir / f"view_{len(imgpoints):02d}.jpg"), overlay)
        print(f"  captured {len(imgpoints)}/{n_views}", file=sys.stderr)
    return objpoints, imgpoints, image_size


def main():
    ap = argparse.ArgumentParser(description="Checkerboard intrinsic calibration for this node's camera.")
    ap.add_argument("--frames", type=int, default=15, help="diverse views to collect (default 15)")
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row (default 9)")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column (default 6)")
    ap.add_argument("--square-mm", type=float, default=25.0, help="square edge length in mm (default 25)")
    ap.add_argument("--min-move", type=float, default=40.0,
                    help="min corner-centroid movement (px) between accepted views (default 40)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output dir (default calibration/)")
    args = ap.parse_args()

    device = capture.CAMERA
    width, height = capture.CAMERA_WIDTH, capture.CAMERA_HEIGHT
    cam_id = capture.camera_id(device)  # same identity keying capture.py reads back
    if cam_id is None:
        print(f"ERROR: {device} has no /dev/v4l/by-id identity; cannot key the calibration "
              "to a physical camera. Plug the camera into USB (not a CSI/virtual device).",
              file=sys.stderr)
        return 1
    print(f"camera: {cam_id} ({device}) at {width}x{height}", file=sys.stderr)

    cap = open_camera(device, width, height)
    if cap is None:
        return 1
    try:
        objpoints, imgpoints, image_size = collect_views(
            cap, (args.cols, args.rows), args.frames, args.min_move,
            args.out / "debug" / cam_id)
    finally:
        cap.release()

    # Scale the board model to real millimetres so tvecs (unused here, but stored
    # rms is unaffected) are metric; intrinsics don't depend on the scale.
    objpoints = [o * args.square_mm for o in objpoints]
    print("calibrating…", file=sys.stderr)
    rms, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)

    out_path = args.out / f"{cam_id}.json"
    _atomic_write_json(out_path, {
        "schema_version": "1",
        "camera_id": cam_id,
        "device": device,
        "node": socket.gethostname(),
        "image_size": list(image_size),
        "pattern": {"cols": args.cols, "rows": args.rows, "square_mm": args.square_mm},
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "rms": round(float(rms), 4),
        "frames_used": len(imgpoints),
        "calibrated_at": dt.datetime.now().astimezone().isoformat(),
    })

    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    print(f"RMS reprojection error: {rms:.3f} px "
          f"{'(good)' if rms < 1.0 else '(HIGH — consider redoing with more varied views)'}",
          file=sys.stderr)
    print(f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}  "
          f"(image center is [{image_size[0]/2:.0f}, {image_size[1]/2:.0f}])", file=sys.stderr)
    print(f"saved -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
