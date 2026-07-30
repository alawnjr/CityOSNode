#!/usr/bin/env python3
"""
Calibration for the Reolink NVR cameras (intrinsics + gravity-levelled extrinsics).

The Pi nodes own a USB camera each and calibrate it live (calibrate_camera.py /
calibrate_extrinsics.py, keyed by USB serial). The Reolink cameras are different:
they hang off an NVR, this host never opens them as a device, and the only input
is a folder of stills exported from the NVR. So this script is photos-only and
keyed by a caller-supplied camera id.

It writes the SAME two files the Pi path writes, so everything downstream
(capture.py's metadata embedding, the mirror's lib/project.ts) consumes them
without a special case:

    calibration/<cam-id>.json              intrinsics (camera_matrix, dist_coeffs)
    calibration/<cam-id>.extrinsics.json   pose in the room frame

Three things here differ from the Pi scripts, each for a measured reason:

1. **The board needs the classic corner finder.** `findChessboardCornersSB`
   returns nothing on the dense 17x17 DFvision plate at 4K — it found 0 of 25
   frames. `findChessboardCorners` + `cornerSubPix` finds 25 of 25.

2. **The tag needs a wider threshold sweep.** The room-view stills are dim and
   the tag is far away; the default `DetectorParameters` missed tag 4 entirely
   in a frame where it is plainly visible. A 3..53 adaptive-threshold sweep with
   subpixel refinement finds it.

3. **The reference tag lies FLAT ON THE FLOOR, so the raw tag frame is not the
   room frame.** For a wall tag the aruco frame is already roughly levelled; for
   a floor tag its +Z (out of the tag) points at the CEILING. The room frame the
   archive uses is gravity-levelled with up = -Y (realsense_extrinsics.py rotates
   measured gravity-up onto [0,-1,0]; the mirror's ROLL_FIX then takes it Y-up).
   Writing the raw frame here would hand the mirror a horizontal axis where it
   expects height, and every camera would land in the wrong place. LEVEL_FLOOR_TAG
   below is that rotation, and it is exact rather than measured: the tag is ON the
   floor, so the tag's own normal IS the floor normal — no depth pass needed.

Verified against the archive's own tag map: reconstructing tag 3's position in
the room frame from three different Reolink views reproduces room_tags' stored
[969.2, -1361.3, 1322.7] mm to within ~50 mm on a 2131 mm baseline, signs and
axes included.

Tag sizes and which tags lie flat come from node.env via calibration_config
(SMARTROOM_TAG_SIZES / SMARTROOM_TAG_FLOOR), or from the flags below. Nothing is
defaulted: PnP range is linear in the assumed edge length, so a guessed size does
not fail loudly, it just puts the camera at the wrong distance.

Usage (laptop or server — needs only opencv + the photos):

    python reolink_calibrate.py --root "/path/to/reolink" \
        --tag-id 4 --tag-sizes 4:335,3:235 --floor-tag 4
    python reolink_calibrate.py --root "/path/to/reolink" --only "Camera 1"

Layout it expects under --root (as exported from the NVR):

    Camera 1/*.jpg                          checkerboard stills for intrinsics
    smartroomNVRcams-Camera1-*.jpg          room view showing the reference tag

Tag sizes are the BLACK SQUARE edge, measured, not the paper.
"""

import argparse
import datetime as dt
import glob
import json
import os
import re
import socket
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

import calibration_config as cfg

PROJECT_ROOT = cfg.PROJECT_ROOT
DEFAULT_OUT = cfg.DEFAULT_OUT

# Board geometry and tag facts live in calibration_config, not here — a second
# copy of a tag size is exactly the stale default that module exists to kill.
BOARD_COLS, BOARD_ROWS, SQUARE_MM = cfg.BOARD_COLS, cfg.BOARD_ROWS, cfg.BOARD_SQUARE_MM

TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
# Filled by main() from node.env + CLI. No module-level sizes: PnP translation is
# linear in the assumed edge length, so a wrong default does not fail, it silently
# reports the camera at a fraction of its true range.
TAG_SIZE_MM: dict = {}
DEFAULT_TAG_SIZE_MM: float = 0.0
FLOOR_TAGS: set = set()

# Raw aruco tag frame -> gravity-levelled room frame, for a tag lying flat on the
# floor. The tag's +Z (out of its face) points up, and the levelled frame wants
# up = -Y, so this is the minimal rotation taking [0,0,1] -> [0,-1,0]: R_x(90deg).
#   X_room = X_tag        Y_room = -Z_tag (down)      Z_room = Y_tag
LEVEL_FLOOR_TAG = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0]])

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Below this the tag is too small for a trustworthy pose; above 3x between the two
# IPPE_SQUARE branches the winning one is a genuine minimum rather than a coin flip.
MIN_TAG_SIDE_PX = 150.0
MIN_AMBIGUITY_RATIO = 3.0


def level_rotation(tag_id: int) -> np.ndarray:
    """Raw aruco tag frame -> the archive's gravity-levelled room frame.

    Only a tag lying flat needs it. An upright tag's -Y is already the vertical,
    which is what calibrate_extrinsics.py writes and what the identity preserves.
    """
    return LEVEL_FLOOR_TAG if int(tag_id) in FLOOR_TAGS else np.eye(3)


def _atomic_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def tag_object_points(tag_id: int) -> np.ndarray:
    """Tag corners in its own frame, in the order cv2.aruco returns: TL,TR,BR,BL."""
    h = TAG_SIZE_MM.get(tag_id, DEFAULT_TAG_SIZE_MM) / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def make_detector() -> "cv2.aruco.ArucoDetector":
    params = cv2.aruco.DetectorParameters()
    # See (2) in the module docstring — the defaults miss a dim, distant tag.
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(TAG_FAMILY), params)


def detect_tags(gray: np.ndarray):
    """{tag_id: 4x2 corners}, keeping the largest detection of any repeated id."""
    corners, ids, _ = make_detector().detectMarkers(gray)
    if ids is None:
        return {}
    best = {}
    for c, i in zip(corners, ids.flatten()):
        tid = int(i)
        area = cv2.contourArea(c[0])
        if tid not in best or area > best[tid][0]:
            best[tid] = (area, c[0].astype(np.float64))
    return {tid: v[1] for tid, v in best.items()}


def solve_tag_pose(img_pts: np.ndarray, tag_id: int, K: np.ndarray, dist: np.ndarray):
    """(R_tag_to_cam, t_cam_mm, reproj_px, ambiguity_ratio, side_px) for one tag.

    Uses solvePnPGeneric so BOTH IPPE_SQUARE branches are visible: a square seen
    from far away has a near-mirror second solution, and silently taking the
    lower-error one is how a camera ends up in the wrong half of the room.
    """
    obj = tag_object_points(tag_id)
    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or not len(rvecs):
        return None
    # solvePnPGeneric returns each error as a 1x1 array, not a scalar.
    err_list = [float(np.ravel(e)[0]) for e in errs]
    order = np.argsort(err_list)
    best, alt = int(order[0]), (int(order[1]) if len(order) > 1 else None)
    ratio = (err_list[alt] / max(err_list[best], 1e-9)) if alt is not None else float("inf")
    side = float(np.mean([np.linalg.norm(img_pts[k] - img_pts[(k + 1) % 4]) for k in range(4)]))
    R, _ = cv2.Rodrigues(rvecs[best])
    return R, tvecs[best].reshape(3), err_list[best], ratio, side


def collect_board_corners(photos, cols=BOARD_COLS, rows=BOARD_ROWS):
    """(objpoints, imgpoints, image_size, used, skipped) from checkerboard stills."""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_MM
    objpoints, imgpoints, used, skipped = [], [], [], []
    image_size = None
    # See (1) in the module docstring — SB finds nothing on this board.
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    for p in photos:
        img = cv2.imread(str(p))
        if img is None:
            skipped.append((p.name, "unreadable"))
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            skipped.append((p.name, f"{size[0]}x{size[1]} differs from {image_size[0]}x{image_size[1]}"))
            continue
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
        if not found:
            skipped.append((p.name, "board not found"))
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        objpoints.append(objp)
        imgpoints.append(corners)
        used.append(p.name)
    return objpoints, imgpoints, image_size, used, skipped


def calibrate_one(cam_dir: Path, room_photo: Path, cam_id: str, out_dir: Path, tag_id: int):
    photos = sorted(p for p in cam_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file())
    if not photos:
        print(f"{cam_id}: no checkerboard photos in {cam_dir}", file=sys.stderr)
        return False

    objpoints, imgpoints, image_size, used, skipped = collect_board_corners(photos)
    for name, why in skipped:
        print(f"  {name}: {why} — skipped", file=sys.stderr)
    if len(used) < 8:
        print(f"{cam_id}: only {len(used)} usable board views (need 8+)", file=sys.stderr)
        return False

    # Rational model: these are wide-angle lenses and the plain 5-coefficient
    # model leaves visible bow at the frame edges.
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=cv2.CALIB_RATIONAL_MODEL)

    per_view = []
    for objp, imgp, rv, tv in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rv, tv, K, dist)
        d = imgp.reshape(-1, 2).astype(np.float64) - proj.reshape(-1, 2).astype(np.float64)
        per_view.append(float(np.sqrt((d ** 2).sum(axis=1)).mean()))

    _atomic_write_json(out_dir / f"{cam_id}.json", {
        "schema_version": "1",
        "camera_id": cam_id,
        "device": f"reolink-nvr:{cam_dir.name}",
        "node": socket.gethostname(),
        "image_size": list(image_size),
        "pattern": {"cols": BOARD_COLS, "rows": BOARD_ROWS, "square_mm": SQUARE_MM},
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "rms": round(float(rms), 4),
        "frames_used": len(used),
        "worst_view_px": round(max(per_view), 3),
        "calibrated_at": dt.datetime.now().astimezone().isoformat(),
    })
    print(f"{cam_id}: intrinsics rms {rms:.3f} px over {len(used)}/{len(photos)} views", file=sys.stderr)

    # ---- extrinsics, from the room-view still --------------------------------
    img = cv2.imread(str(room_photo))
    if img is None:
        print(f"{cam_id}: cannot read room photo {room_photo}", file=sys.stderr)
        return False
    tags = detect_tags(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if tag_id not in tags:
        print(f"{cam_id}: reference tag {tag_id} not visible in {room_photo.name} "
              f"(saw {sorted(tags) or 'nothing'}) — extrinsics skipped", file=sys.stderr)
        return False

    solved = solve_tag_pose(tags[tag_id], tag_id, K, dist)
    if solved is None:
        print(f"{cam_id}: solvePnP failed on tag {tag_id}", file=sys.stderr)
        return False
    R_tag_to_cam, tvec_mm, reproj, ratio, side_px = solved

    if side_px < MIN_TAG_SIDE_PX:
        print(f"{cam_id}: WARNING tag {tag_id} is only ~{side_px:.0f}px per side — "
              f"pose may be noisy; prefer a view where it reads larger", file=sys.stderr)
    if ratio < MIN_AMBIGUITY_RATIO:
        print(f"{cam_id}: WARNING pose ambiguity — the two square solutions are within "
              f"{ratio:.1f}x ({reproj:.2f}px vs {reproj * ratio:.2f}px). Check the result "
              f"against the real room before trusting it.", file=sys.stderr)

    # Camera pose in the RAW tag frame, then levelled (see (3) in the docstring).
    L = level_rotation(tag_id)
    R_cam_to_tag = R_tag_to_cam.T
    cam_pos_tag = (-R_cam_to_tag @ tvec_mm).reshape(3)
    R_cam_to_room = L @ R_cam_to_tag
    cam_pos_room = L @ cam_pos_tag

    # Every other tag in the same frame, mapped into the room frame — the same
    # chaining realsense_extrinsics.py does, so tags.json/room_tags can grow.
    also = {}
    for tid, pts in sorted(tags.items()):
        if tid == tag_id:
            continue
        s = solve_tag_pose(pts, tid, K, dist)
        if s is None:
            continue
        R_t, t_t, e_t, _r, _sp = s
        pos_room = L @ (R_cam_to_tag @ (t_t - tvec_mm)).reshape(3)
        also[str(tid)] = {
            "position_mm": [round(float(v), 1) for v in pos_room],
            "size_mm": TAG_SIZE_MM.get(tid, DEFAULT_TAG_SIZE_MM),
            "reprojection_error_px": round(e_t, 3),
        }

    _atomic_write_json(out_dir / f"{cam_id}.extrinsics.json", {
        "schema_version": "1",
        "camera_id": cam_id,
        "node": socket.gethostname(),
        "frame": "tag: origin=center, X=right, Y=DOWN (gravity-levelled: up is -Y), Z=out of tag; units mm",
        "tag": {"family": cfg.TAG_FAMILY, "id": tag_id,
                "size_mm": TAG_SIZE_MM.get(tag_id, DEFAULT_TAG_SIZE_MM)},
        "image_size": [img.shape[1], img.shape[0]],
        "rotation_cam_to_room": R_cam_to_room.tolist(),
        "camera_position_mm": [round(float(v), 1) for v in cam_pos_room],
        "reprojection_error_px": round(reproj, 3),
        "ambiguity_ratio": (None if not np.isfinite(ratio) else round(ratio, 2)),
        "tag_side_px": round(side_px, 1),
        "levelled": ({"source": "floor-tag", "note": "reference tag lies flat on the floor; its own "
                      "normal is the floor normal, so no depth levelling is needed"}
                     if int(tag_id) in FLOOR_TAGS else
                     {"source": "none", "note": "upright reference tag; raw tag frame kept, as in "
                      "calibrate_extrinsics.py"}),
        "tags_seen": sorted(int(t) for t in tags),
        "other_tags_room_mm": also,
        "room_photo": room_photo.name,
        "calibrated_at": dt.datetime.now().astimezone().isoformat(),
    })

    x, y, z = cam_pos_room
    print(f"{cam_id}: camera at [{x:.0f}, {y:.0f}, {z:.0f}] mm in the room frame "
          f"({np.linalg.norm(cam_pos_room) / 1000:.2f} m from tag {tag_id}, "
          f"{-y / 1000:.2f} m above the floor), reprojection {reproj:.2f} px", file=sys.stderr)
    return True


def find_room_photo(root: Path, cam_dir_name: str):
    """Newest smartroomNVRcams-<Camera N>-*.jpg for this camera, if any."""
    slug = cam_dir_name.replace(" ", "")
    hits = sorted(glob.glob(str(root / f"smartroomNVRcams-{slug}-*.jpg")))
    return Path(hits[-1]) if hits else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="folder holding 'Camera N/' dirs + room-view stills")
    ap.add_argument("--only", action="append", help="limit to these camera folders (repeatable)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output dir (default {DEFAULT_OUT})")
    ap.add_argument("--tag-id", type=int, default=None, help="reference tag id (default: node.env)")
    ap.add_argument("--tag-sizes", default=None,
                    help="black-square edges, 'id:mm,id:mm' (default: SMARTROOM_TAG_SIZES)")
    ap.add_argument("--floor-tag", action="append", type=int, default=None,
                    help="tag id lying flat on the floor, repeatable (default: SMARTROOM_TAG_FLOOR)")
    ap.add_argument("--id-prefix", default="reolink-", help="camera id prefix (default 'reolink-')")
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"ERROR: --root {args.root} is not a directory", file=sys.stderr)
        return 1

    # node.env is the base; CLI overrides it. Nothing is invented here — see the
    # TAG_SIZE_MM note: a guessed edge length misplaces the camera silently.
    global TAG_SIZE_MM, DEFAULT_TAG_SIZE_MM, FLOOR_TAGS
    cfg.load_node_env()
    tag_id = args.tag_id if args.tag_id is not None else cfg.tag_id()
    TAG_SIZE_MM = dict(cfg.tag_sizes())
    DEFAULT_TAG_SIZE_MM = cfg.tag_size_mm()
    FLOOR_TAGS = set(cfg.floor_tags())
    if args.tag_sizes:
        try:
            for pair in args.tag_sizes.split(","):
                k, v = pair.split(":")
                TAG_SIZE_MM[int(k)] = float(v)
        except ValueError:
            print(f"ERROR: --tag-sizes wants 'id:mm,id:mm', got {args.tag_sizes!r}", file=sys.stderr)
            return 1
    if args.floor_tag:
        FLOOR_TAGS |= set(args.floor_tag)

    if tag_id not in TAG_SIZE_MM and not DEFAULT_TAG_SIZE_MM:
        print(f"ERROR: no size for reference tag {tag_id}. Measure its BLACK SQUARE and pass "
              f"--tag-sizes {tag_id}:<mm> (or set SMARTROOM_TAG_SIZES in node.env). Refusing to "
              f"guess: PnP range is linear in this number.", file=sys.stderr)
        return 1
    if tag_id not in FLOOR_TAGS:
        print(f"NOTE: tag {tag_id} is not marked as lying flat, so the raw tag frame is kept. "
              f"If it is on the floor, pass --floor-tag {tag_id} (or SMARTROOM_TAG_FLOOR) or the "
              f"mirror will read a horizontal axis as height.", file=sys.stderr)

    cam_dirs = sorted((d for d in args.root.iterdir() if d.is_dir() and re.match(r"(?i)^camera\s*\d+$", d.name)),
                      key=lambda d: int(re.sub(r"\D", "", d.name) or 0))
    if args.only:
        wanted = {w.lower() for w in args.only}
        cam_dirs = [d for d in cam_dirs if d.name.lower() in wanted]
    if not cam_dirs:
        print(f"ERROR: no 'Camera N' folders under {args.root}", file=sys.stderr)
        return 1

    ok_count = 0
    for d in cam_dirs:
        cam_id = args.id_prefix + d.name.lower().replace(" ", "")
        room = find_room_photo(args.root, d.name)
        if room is None:
            print(f"{cam_id}: no room-view still (smartroomNVRcams-{d.name.replace(' ', '')}-*.jpg) — skipped",
                  file=sys.stderr)
            continue
        if calibrate_one(d, room, cam_id, args.out, tag_id):
            ok_count += 1
        print(file=sys.stderr)

    print(f"calibrated {ok_count}/{len(cam_dirs)} camera(s) -> {args.out}", file=sys.stderr)
    return 0 if ok_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
