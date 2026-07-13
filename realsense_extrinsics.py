#!/usr/bin/env python3
"""AprilTag extrinsic calibration for a RealSense depth camera — native SDK data.

Where calibrate_extrinsics.py needs a prior checkerboard INTRINSIC calibration,
this uses the RealSense's FACTORY intrinsics read straight off the device (every
unit is individually calibrated by Intel), and cross-checks the resulting pose
against the camera's own depth measurements at the tag corners — the two are
independent, so agreement is a built-in sanity check. Same tag, same room frame,
same output schema as the webcam flow:

    origin = tag center, X = tag's right, Y = tag's up, Z = out of the tag
    (units: millimetres)
    output: calibration/<usb-serial>.extrinsics.json

Used in-process by realsense_depth_page.py (the "Calibrate extrinsic" button on
the web page grabs live frames); can also run standalone on the Pi with the
venv python (the camera must be free — stop the depth page first):

    ~/CityOS/.venv/bin/python ~/CityOS/realsense_extrinsics.py [--serial SN]

Tag: 36h11, id 1, black-square edge 173mm by default — override with
--tag-id / --tag-size-mm or SMARTROOM_TAG_ID / SMARTROOM_TAG_SIZE_MM (node.env).

Tag chaining: any OTHER 36h11 tag (tag 2, ...) visible in the same frame as the
reference tag gets its pose computed in the room frame and merged into
calibration/tags.json (all tags assumed printed at the same size). capture.py
embeds that map into every recording's metadata.json, so a camera that can only
see tag 2 can still be placed in the one room frame downstream.
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

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_ROOT / "calibration"

TAG_ID = int(os.environ.get("SMARTROOM_TAG_ID", "1"))
TAG_SIZE_MM = float(os.environ.get("SMARTROOM_TAG_SIZE_MM", "173"))


def _atomic_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def intrinsics_to_cv(intr):
    """rs.intrinsics -> (K, dist) for OpenCV. The color sensor usually reports
    (inverse_)brown_conrady with tiny coefficients; if the model isn't a plain
    forward Brown-Conrady, zeros are closer to correct than misusing them."""
    K = np.array([[intr.fx, 0.0, intr.ppx],
                  [0.0, intr.fy, intr.ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    model = getattr(intr.model, "name", str(intr.model))
    if "brown" in model and "inverse" not in model:
        dist = np.array(intr.coeffs, dtype=np.float64)
    else:
        dist = np.zeros(5, dtype=np.float64)
    return K, dist


def _depth_at(depth_m, px, py):
    """Median of valid depth (meters) in a 5x5 window around a pixel."""
    h, w = depth_m.shape
    x, y = int(round(px)), int(round(py))
    if not (0 <= x < w and 0 <= y < h):
        return None
    window = depth_m[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
    valid = window[window > 0]
    return float(np.median(valid)) if valid.size else None


def calibrate_from_samples(samples, intr, serial, camera_name="RealSense",
                           tag_id=TAG_ID, tag_size_mm=TAG_SIZE_MM,
                           out_dir=DEFAULT_OUT):
    """samples: list of (color_bgr, depth_m) with depth ALIGNED to color.
    Returns (ok, message); writes calibration/<serial>.extrinsics.json and a
    debug overlay on success."""
    if not samples:
        return False, "no frames captured"
    K, dist = intrinsics_to_cv(intr)

    # Tag corner model for SOLVEPNP_IPPE_SQUARE (aruco corner order TL,TR,BR,BL),
    # tag-centered, Z out of the tag — identical to calibrate_extrinsics.py.
    s = tag_size_mm / 2.0
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11),
        cv2.aruco.DetectorParameters())

    def tag_pose(corners_1x4):
        """(reproj_err, rvec, tvec) for one detected tag, or None."""
        img_pts = corners_1x4.reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1).mean())
        return err, rvec, tvec, img_pts

    results = []      # reference tag: (reproj_err, rvec, tvec, img_pts, color_bgr, depth_m)
    other_tags = {}   # other tag id -> list of (err, R_room, pos_room_mm) per frame
    for color_bgr, depth_m in samples:
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY))
        if ids is None:
            continue
        flat = ids.flatten()
        hits = np.where(flat == tag_id)[0]
        if not len(hits):
            continue
        ref = tag_pose(corners[int(hits[0])])
        if ref is None:
            continue
        err, rvec, tvec, img_pts = ref
        results.append((err, rvec, tvec, img_pts, color_bgr, depth_m))
        # Tag chaining: every OTHER tag visible in the same frame gets its pose
        # expressed in the reference tag's (room) frame, so it can anchor
        # cameras that can't see the reference tag. Assumes the same printed
        # tag size for all tags.
        R1, _ = cv2.Rodrigues(rvec)
        for j, other_id in enumerate(flat):
            if int(other_id) == tag_id:
                continue
            other = tag_pose(corners[j])
            if other is None:
                continue
            err2, rvec2, tvec2, _pts2 = other
            R2, _ = cv2.Rodrigues(rvec2)
            pos_room = (R1.T @ (tvec2 - tvec)).flatten()        # tag center, mm
            R_room = R1.T @ R2                                   # tag axes in room frame
            other_tags.setdefault(int(other_id), []).append((max(err, err2), R_room, pos_room))

    if not results:
        return False, (f"tag {tag_id} (36h11) not seen by the color camera — make sure "
                       "it's flat, well lit and large enough in frame")

    positions = []
    for err, rvec, tvec, *_ in results:
        R, _ = cv2.Rodrigues(rvec)
        positions.append((-R.T @ tvec).flatten())
    spread = float(np.linalg.norm(np.std(np.array(positions), axis=0))) if len(positions) > 1 else 0.0

    err, rvec, tvec, img_pts, color_bgr, depth_m = min(results, key=lambda r: r[0])
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = (-R.T @ tvec).flatten()

    # Native cross-check: the camera's own depth at each tag corner vs the
    # distance the PnP pose predicts for that corner. Independent measurements —
    # small disagreement means both the pose and the printed tag size are right.
    pnp_corners_cam = (R @ obj.T + tvec).T  # 4x3, mm, camera frame
    diffs = []
    for (px, py), pnp_pt in zip(img_pts, pnp_corners_cam):
        z = _depth_at(depth_m, px, py)
        if z is None:
            continue
        diffs.append(abs(z * 1000.0 - float(np.linalg.norm(pnp_pt))))
    depth_agreement_mm = round(float(np.mean(diffs)), 1) if diffs else None

    h, w = color_bgr.shape[:2]
    debug_dir = out_dir / "debug" / serial
    debug_dir.mkdir(parents=True, exist_ok=True)
    overlay = color_bgr.copy()
    cv2.aruco.drawDetectedMarkers(overlay, [img_pts.reshape(1, 4, 2).astype(np.float32)])
    cv2.drawFrameAxes(overlay, K, dist, rvec, tvec, tag_size_mm * 0.75)
    cv2.imwrite(str(debug_dir / "extrinsic_live.jpg"), overlay)

    out_path = out_dir / f"{serial}.extrinsics.json"
    _atomic_write_json(out_path, {
        "schema_version": "1",
        "camera_id": serial,
        "camera": camera_name,
        "source": "realsense_native",  # factory intrinsics + depth cross-check
        "node": socket.gethostname(),
        "frame": "tag: origin=center, X=right, Y=up, Z=out of tag; units mm",
        "tag": {"family": "36h11", "id": tag_id, "size_mm": tag_size_mm},
        "image_size": [w, h],
        "rvec": rvec.flatten().tolist(),          # tag -> camera rotation (Rodrigues)
        "tvec_mm": tvec.flatten().tolist(),       # tag origin in camera frame
        "rotation_cam_to_room": R.T.tolist(),
        "camera_position_mm": [round(float(v), 1) for v in cam_pos],
        "reprojection_error_px": round(err, 3),
        "depth_agreement_mm": depth_agreement_mm,
        "frames_used": len(results),
        "position_spread_mm": round(spread, 1),
        "calibrated_at": dt.datetime.now().astimezone().isoformat(),
    })

    tag_notes = _save_room_tags(other_tags, serial, tag_id, tag_size_mm, out_dir)

    dist_m = float(np.linalg.norm(cam_pos)) / 1000.0
    agree = (f", depth agrees within {depth_agreement_mm} mm" if depth_agreement_mm is not None
             else ", no depth at corners")
    return True, (f"camera at [{cam_pos[0]:.0f}, {cam_pos[1]:.0f}, {cam_pos[2]:.0f}] mm in tag frame "
                  f"({dist_m:.2f} m from tag), reproj {err:.2f} px over {len(results)} frame(s)"
                  f"{agree}{tag_notes} — saved {out_path.name}")


def _save_room_tags(other_tags, serial, ref_tag_id, tag_size_mm, out_dir):
    """Merge every chained tag's room-frame pose into calibration/tags.json —
    the shared room tag map that capture.py embeds into each recording's
    metadata. Returns a short note for the status message ('' if no other
    tags were seen)."""
    if not other_tags:
        return ""
    path = out_dir / "tags.json"
    try:
        tag_map = json.loads(path.read_text())
    except (OSError, ValueError):
        tag_map = {
            "schema_version": "1",
            "frame": f"tag {ref_tag_id}: origin=center, X=right, Y=up, Z=out of tag; units mm",
            "reference_tag": {"family": "36h11", "id": ref_tag_id, "size_mm": tag_size_mm},
            "tags": {},
        }
    notes = []
    for other_id, observations in sorted(other_tags.items()):
        # best (lowest joint reprojection error) observation gives the rotation;
        # the position is the mean, with the spread as the consistency check
        observations.sort(key=lambda o: o[0])
        best_err, R_room, _ = observations[0]
        positions = np.array([o[2] for o in observations])
        pos = positions.mean(axis=0)
        spread = float(np.linalg.norm(positions.std(axis=0))) if len(positions) > 1 else 0.0
        tag_map["tags"][str(other_id)] = {
            "position_mm": [round(float(v), 1) for v in pos],
            "rotation_tag_to_room": R_room.tolist(),
            "size_mm": tag_size_mm,
            "reprojection_error_px": round(best_err, 3),
            "position_spread_mm": round(spread, 1),
            "frames_used": len(observations),
            "observed_by": serial,
            "measured_at": dt.datetime.now().astimezone().isoformat(),
        }
        notes.append(f"tag {other_id} at [{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}] mm")
    _atomic_write_json(path, tag_map)
    return "; " + ", ".join(notes) + " (room frame, saved to tags.json)"


def main():
    import pyrealsense2 as rs  # standalone mode only; the page imports us without it

    ap = argparse.ArgumentParser(description="AprilTag extrinsic calibration for a RealSense camera.")
    ap.add_argument("--serial", default=None, help="camera USB serial (default: first device)")
    ap.add_argument("--tag-id", type=int, default=TAG_ID)
    ap.add_argument("--tag-size-mm", type=float, default=TAG_SIZE_MM)
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    try:
        profile = pipeline.start(config)
    except RuntimeError:
        config = rs.config()
        if args.serial:
            config.enable_device(args.serial)
        profile = pipeline.start(config)  # let the SDK pick (USB 2 etc.)

    device = profile.get_device()
    serial = device.get_info(rs.camera_info.serial_number)
    name = device.get_info(rs.camera_info.name)
    depth_scale = device.first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    samples = []
    try:
        for _ in range(10):  # warm-up for auto-exposure
            pipeline.wait_for_frames(5000)
        while len(samples) < args.frames:
            frames = align.process(pipeline.wait_for_frames(5000))
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if not depth or not color:
                continue
            samples.append((np.asanyarray(color.get_data()).copy(),
                            np.asanyarray(depth.get_data()).astype(np.float32) * depth_scale))
            time.sleep(0.2)
    finally:
        pipeline.stop()

    ok, message = calibrate_from_samples(samples, intr, serial, camera_name=name,
                                         tag_id=args.tag_id, tag_size_mm=args.tag_size_mm)
    print(message, file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
