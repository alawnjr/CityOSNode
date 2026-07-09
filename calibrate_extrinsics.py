#!/usr/bin/env python3
"""
Extrinsic calibration for a smartroom node camera, from an AprilTag.

Where the INTRINSIC calibration (calibrate_camera.py) measures the lens, this
measures the camera's POSE — where it sits and where it points — relative to a
single AprilTag (36h11 family) taped somewhere static and visible. The tag
defines the ROOM coordinate frame:

    origin = tag center, X = tag's right, Y = tag's up, Z = out of the tag
    (units: millimetres)

Run ON the Pi with the venv python, from photos snapped on the web page
(same flow as intrinsics — snap 3+ photos of the tag, then):

    ~/CityOS/.venv/bin/python ~/CityOS/calibrate_extrinsics.py --photos

Requires the camera to be intrinsically calibrated first (needs K + distortion
from calibration/<serial>.json). Output: calibration/<serial>.extrinsics.json
with the camera's position and orientation in the tag/room frame, plus debug
overlays (detected tag + axes) under calibration/debug/<serial>/.

Tag: 36h11, id 1, black-square edge 173mm by default — override with
--tag-id / --tag-size-mm or the SMARTROOM_TAG_SIZE_MM env (node.env works).
Note: the tag's printed size sets the METRIC SCALE of everything downstream;
measure the black square, not the paper.
"""

import argparse
import datetime as dt
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

import capture  # same device identity + calibration lookup as recording

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_ROOT / "calibration"


def _atomic_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_intrinsics(cam_id: str, width: int, height: int):
    """K + dist from the intrinsic calibration, scaled to (width, height)."""
    path = DEFAULT_OUT / f"{cam_id}.json"
    try:
        cal = json.loads(path.read_text())
    except (OSError, ValueError):
        return None, None
    mtx = np.array(cal["camera_matrix"], dtype=np.float64)
    dist = np.array(cal["dist_coeffs"], dtype=np.float64)
    cw, ch = cal["image_size"]
    if (cw, ch) != (width, height) and cw and ch:
        sx, sy = width / cw, height / ch
        mtx = mtx.copy()
        mtx[0, 0] *= sx
        mtx[0, 2] *= sx
        mtx[1, 1] *= sy
        mtx[1, 2] *= sy
    return mtx, dist


def main():
    ap = argparse.ArgumentParser(description="AprilTag extrinsic calibration for this node's camera.")
    ap.add_argument("--photos", nargs="?", type=Path, const=capture.DATA_DIR / "photos",
                    default=capture.DATA_DIR / "photos", metavar="DIR",
                    help="photos dir (default: data/photos — snap the tag with the web page)")
    ap.add_argument("--tag-id", type=int, default=int(os.environ.get("SMARTROOM_TAG_ID", "1")))
    ap.add_argument("--tag-size-mm", type=float,
                    default=float(os.environ.get("SMARTROOM_TAG_SIZE_MM", "173")),
                    help="edge of the BLACK square in mm (default 173)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    cam_id = capture.camera_id(capture.CAMERA)
    if cam_id is None:
        print("ERROR: camera has no /dev/v4l/by-id identity.", file=sys.stderr)
        return 1

    photos = sorted(args.photos.glob("*.jpg"))
    if not photos:
        print(f"ERROR: no photos in {args.photos} — snap the AprilTag with the web page first.",
              file=sys.stderr)
        return 1

    # Tag corner model for SOLVEPNP_IPPE_SQUARE (order must match aruco's
    # detected corner order: TL, TR, BR, BL), tag-centered, Z out of the tag.
    s = args.tag_size_mm / 2.0
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11),
        cv2.aruco.DetectorParameters())

    debug_dir = args.out / "debug" / cam_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    results = []  # (reproj_err, rvec, tvec, image_size, photo_name)
    mtx = dist = None
    for p in photos:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        mtx, dist = load_intrinsics(cam_id, w, h)
        if mtx is None:
            print(f"ERROR: no intrinsic calibration for {cam_id} — run calibrate_camera.py first.",
                  file=sys.stderr)
            return 1
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        idx = None
        if ids is not None:
            hits = np.where(ids.flatten() == args.tag_id)[0]
            idx = int(hits[0]) if len(hits) else None
        if idx is None:
            print(f"  {p.name}: tag {args.tag_id} not found — skipped", file=sys.stderr)
            continue
        img_pts = corners[idx].reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts, mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            print(f"  {p.name}: solvePnP failed — skipped", file=sys.stderr)
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, mtx, dist)
        err = float(np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1).mean())
        results.append((err, rvec, tvec, (w, h), p.name))
        overlay = frame.copy()
        cv2.aruco.drawDetectedMarkers(overlay, [corners[idx]])
        cv2.drawFrameAxes(overlay, mtx, dist, rvec, tvec, args.tag_size_mm * 0.75)
        cv2.imwrite(str(debug_dir / f"extrinsic_{p.stem}.jpg"), overlay)
        print(f"  {p.name}: tag found, reprojection {err:.2f} px", file=sys.stderr)

    if not results:
        print(f"ERROR: tag {args.tag_id} (36h11) not detected in any photo. Make sure it's "
              "flat, well lit, unobstructed, and large enough in frame.", file=sys.stderr)
        return 1

    # Best view (lowest reprojection error) is the answer; the spread of camera
    # positions across views is the consistency check.
    positions = []
    for err, rvec, tvec, _size, _name in results:
        R, _ = cv2.Rodrigues(rvec)
        positions.append((-R.T @ tvec).flatten())
    spread = float(np.linalg.norm(np.std(np.array(positions), axis=0))) if len(positions) > 1 else 0.0

    err, rvec, tvec, image_size, best_name = min(results, key=lambda r: r[0])
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = (-R.T @ tvec).flatten()

    out_path = args.out / f"{cam_id}.extrinsics.json"
    _atomic_write_json(out_path, {
        "schema_version": "1",
        "camera_id": cam_id,
        "node": socket.gethostname(),
        "frame": "tag: origin=center, X=right, Y=up, Z=out of tag; units mm",
        "tag": {"family": "36h11", "id": args.tag_id, "size_mm": args.tag_size_mm},
        "image_size": list(image_size),
        "rvec": rvec.flatten().tolist(),          # tag -> camera rotation (Rodrigues)
        "tvec_mm": tvec.flatten().tolist(),       # tag origin in camera frame
        "rotation_cam_to_room": R.T.tolist(),
        "camera_position_mm": [round(float(v), 1) for v in cam_pos],
        "reprojection_error_px": round(err, 3),
        "photos_used": len(results),
        "best_photo": best_name,
        "position_spread_mm": round(spread, 1),
        "calibrated_at": dt.datetime.now().astimezone().isoformat(),
    })

    dist_m = float(np.linalg.norm(cam_pos)) / 1000.0
    print(f"camera position in tag frame: [{cam_pos[0]:.0f}, {cam_pos[1]:.0f}, {cam_pos[2]:.0f}] mm "
          f"({dist_m:.2f} m from the tag)", file=sys.stderr)
    print(f"reprojection {err:.2f} px over {len(results)} view(s), "
          f"position spread {spread:.0f} mm {'(consistent)' if spread < 50 else '(HIGH — tag moved or blurry photos?)'}",
          file=sys.stderr)
    print(f"saved -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
