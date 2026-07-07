#!/usr/bin/env python3
"""
Intrinsic camera calibration for a smartroom node, using a printed checkerboard.

Run ON the Pi (headless — no display needed) with the venv python. Two modes:

    # from photos snapped on the web page (recommended — you can see framing):
    ~/CityOS/.venv/bin/python ~/CityOS/calibrate_camera.py --photos

    # live capture (camera must be free — close the live-view browser tab):
    ~/CityOS/.venv/bin/python ~/CityOS/calibrate_camera.py

Photos mode reads data/photos/*.jpg (the Snap photo button's files — snap the
board at 10+ positions: center, corners, near, far, tilted) and never opens the
camera. Live mode holds a printed checkerboard in front of the camera and moves
it slowly around the frame while the script collects views.
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


def board_model(pattern):
    # One canonical board model in board coordinates (Z=0 plane); square size is
    # applied by the caller (calibrateCamera only needs a consistent scale).
    cols, rows = pattern
    board = np.zeros((cols * rows, 3), np.float32)
    board[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return board


def detect_corners(gray, pattern):
    """Checkerboard corners in one grayscale image, or None. Detects on a
    half-scale copy (fast enough for the Pi 3), then refines the corner
    positions at full resolution for calibration accuracy."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    found, corners = cv2.findChessboardCorners(
        small, pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK)
    if not found:
        return None
    corners = corners * 2.0  # back to full-res coordinates
    return cv2.cornerSubPix(
        gray, corners.astype(np.float32), (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))


def collect_from_photos(paths, pattern, debug_dir: Path):
    """Detect the board in pre-snapped photos (the web page's Snap photo files).
    Returns (objpoints, imgpoints, image_size, sample_bgr) — sample is the last
    detected photo, for the before/after comparison."""
    board = board_model(pattern)
    objpoints, imgpoints = [], []
    image_size, sample = None, None
    debug_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is None:
            print(f"  {p.name}: unreadable — skipped", file=sys.stderr)
            continue
        size = (frame.shape[1], frame.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            print(f"  {p.name}: {size[0]}x{size[1]} differs from {image_size[0]}x{image_size[1]} — skipped "
                  "(all photos must share one resolution)", file=sys.stderr)
            continue
        corners = detect_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), pattern)
        if corners is None:
            print(f"  {p.name}: no board found — skipped", file=sys.stderr)
            continue
        objpoints.append(board)
        imgpoints.append(corners)
        sample = frame
        overlay = frame.copy()
        cv2.drawChessboardCorners(overlay, pattern, corners, True)
        cv2.imwrite(str(debug_dir / f"view_{len(imgpoints):02d}.jpg"), overlay)
        print(f"  {p.name}: board detected ({len(imgpoints)} good so far)", file=sys.stderr)
    return objpoints, imgpoints, image_size, sample


def collect_views(cap, pattern, n_views, min_move_px, debug_dir: Path):
    """Grab frames until n_views diverse checkerboard detections are accepted.
    Returns (object_points_list, image_points_list, image_size)."""
    cols, rows = pattern
    board = board_model(pattern)

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

        corners = detect_corners(gray, (cols, rows))
        now = time.monotonic()
        if corners is None:
            if now - last_status > 3:
                print(f"  … board not visible ({len(imgpoints)}/{n_views})", file=sys.stderr)
                last_status = now
            continue

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
    ap.add_argument("--photos", nargs="?", type=Path, const=capture.DATA_DIR / "photos",
                    metavar="DIR",
                    help="calibrate from the web page's snapped photos instead of live capture "
                         "(default dir: data/photos/); the camera is not opened at all")
    args = ap.parse_args()

    device = capture.CAMERA
    width, height = capture.CAMERA_WIDTH, capture.CAMERA_HEIGHT
    cam_id = capture.camera_id(device)  # same identity keying capture.py reads back
    if cam_id is None:
        print(f"ERROR: {device} has no /dev/v4l/by-id identity; cannot key the calibration "
              "to a physical camera. Plug the camera into USB (not a CSI/virtual device).",
              file=sys.stderr)
        return 1

    if args.photos is not None:
        photos = sorted(args.photos.glob("*.jpg"))
        if not photos:
            print(f"ERROR: no .jpg photos in {args.photos} — snap some with the web page first.",
                  file=sys.stderr)
            return 1
        print(f"camera: {cam_id} — calibrating from {len(photos)} photo(s) in {args.photos}",
              file=sys.stderr)
        objpoints, imgpoints, image_size, sample = collect_from_photos(
            photos, (args.cols, args.rows), args.out / "debug" / cam_id)
        if len(imgpoints) < 5:
            print(f"ERROR: only {len(imgpoints)} photo(s) with a detectable board — need at "
                  "least 5 (10+ diverse views recommended). Snap more and rerun.", file=sys.stderr)
            return 1
        if len(imgpoints) < 10:
            print(f"warning: only {len(imgpoints)} usable views — calibration will work but "
                  "10+ diverse views give better results.", file=sys.stderr)
    else:
        print(f"camera: {cam_id} ({device}) at {width}x{height}", file=sys.stderr)
        cap = open_camera(device, width, height)
        if cap is None:
            return 1
        try:
            objpoints, imgpoints, image_size = collect_views(
                cap, (args.cols, args.rows), args.frames, args.min_move,
                args.out / "debug" / cam_id)
            # One extra still for the before/after comparison written below.
            ok, sample = cap.read()
            if not ok:
                sample = None
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

    # Before/after comparison: the SAME single frame, raw vs undistorted with the
    # just-computed intrinsics. The correction is most visible at the frame edges
    # (straight lines that bow in `before` come out straight in `after`).
    if sample is not None:
        debug_dir = args.out / "debug" / cam_id
        debug_dir.mkdir(parents=True, exist_ok=True)
        after = cv2.undistort(sample, mtx, dist)
        cv2.imwrite(str(debug_dir / "before.jpg"), sample)
        cv2.imwrite(str(debug_dir / "after.jpg"), after)
        side = np.hstack([sample, after])
        for text, x in (("before", 12), ("after", sample.shape[1] + 12)):
            cv2.putText(side, text, (x, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(side, text, (x, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imwrite(str(debug_dir / "compare.jpg"), side)
        print(f"before/after -> {debug_dir}/before.jpg, after.jpg, compare.jpg", file=sys.stderr)

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
