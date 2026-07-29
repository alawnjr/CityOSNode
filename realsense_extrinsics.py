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

MIXED TAG SIZES. Tags need not all be the same size — a camera that needs more
pixels on a tag gets a bigger print (a 90-degree-FOV D455 needs ~310mm at 2m
where a D435 manages on ~145mm). Sizes that differ from the default go in
node.env:

    SMARTROOM_TAG_SIZES=3:235,4:335        # id:mm, black square

This is not optional bookkeeping. PnP translation is linear in the assumed edge
length, so solving a 235mm tag with a 138.4mm model maps it at 59% of its true
range — and the only outward symptom is the depth cross-check disagreeing with
the pose (measured: 20.7%). A tag's size, once solved, is recorded in its
tags.json entry and reused from there, so a mapped anchor carries its own size.

Tag chaining: any OTHER 36h11 tag (tag 2, ...) visible in the same frame as the
reference tag gets its pose computed in the room frame and merged into
calibration/tags.json, each with the size it was solved at. capture.py embeds
that map into every recording's metadata.json, so a camera that can only see
tag 2 can still be placed in the one room frame downstream.

Joint multi-tag solve: when a frame shows several MAPPED tags, the pose is fitted
to all of their corners at once. One 138mm tag only gives the solver 138mm of
baseline to read perspective from, which is why a camera facing it square can
barely tell its two pose branches apart; two tags a metre apart give it a metre.
Measured on the D435, which sees tags 1 and 2 together: at 0.3px of corner noise
the right branch goes from 48% of the time (a coin flip) to 100%. The catch is
that the joint solve is only as ACCURATE as calibration/tags.json — it kills the
scatter, not a bias baked into the map. A tape measure beats PnP on a 35px tag,
so an entry may be written by hand with "source": "measured" and no solve will
overwrite it.

Gravity levelling: the tag defines the room frame's ORIGIN and YAW, but nothing
else about it deserves to be trusted with the vertical. It is small in the image
(tens of pixels), so its corners pin the pose's TILT down only weakly, and it is
itself only as plumb as whoever hung it — either way, every camera then looks
tilted in the room frame even though each is physically level. So the vertical is
measured rather than inferred from the tag: find_room_vertical pools the surface
normals of every horizontal thing the depth camera can see (floor, ceiling,
desks — they all share one normal) and the saved pose is rotated onto it, in two
steps that fix two different errors:

  1. the TAG hangs crooked — the room frame itself is relabelled, so every camera
     rotates about the tag origin. Measured once, by a camera whose view of the
     tag is oblique enough for its solve to be trusted, and shared through
     calibration/room_level.json (which capture.py embeds and which cameras that
     see no horizontal surface fall back on).
  2. THIS camera's PnP tilt is off — its orientation alone is corrected, keeping
     the position PnP put it at and its yaw.

What levelling cannot fix is YAW, which still rests entirely on the tag's corners
and degrades in proportion to how few pixels across the tag is; calibration warns
when the tag is too small or too near the frame edge to pin it down.
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

# How obliquely a camera must see the tag before its solve is trusted to define
# the room's vertical. Below this the planar-PnP tilt is not observable enough.
MIN_TAG_OBLIQUITY_DEG = 20.0
# How much horizontal surface the depth must show before its measured vertical
# is trusted over the tag's. A camera staring at a wall sees a few thousand
# normals off one desk — not enough to overrule anything.
MIN_LEVEL_NORMALS = 15000
MAX_LEVEL_SCATTER_DEG = 6.0
# Every normal-COUNT threshold here was tuned on 640x480 frames, and
# _surface_normals samples a fixed 1-in-`step` grid — so the raw counts scale
# with the frame's pixel area, not with how much room is actually in view. Once
# calibration moved to the cameras' full colour modes that made the gates
# meaningless: at 1920x1080 the same starved sliver of desk yields 6.75x more
# normals, which let the D435 level off ~6400 640x480-equivalent normals having
# correctly refused 5537 the run before. Scale the thresholds by area so a gate
# means the same thing at any resolution.
NORMALS_REF_PIXELS = 640 * 480
# Below this the tag is too few pixels across for its corners to pin down yaw.
MIN_TAG_PIXELS = 60.0
# Colour modes to try for a CALIBRATION capture, highest first — extrinsics are
# solved at the best resolution the camera offers, not at the recording profile.
#
# Corner-localisation noise is fixed in PIXELS (~0.3 px), so the pose's angular
# error scales as 1/tag_px, and tag_px = f*size/range with f proportional to the
# capture width. The 138.4 mm tag at ~2.5 m is ~26 px across at 640x480 and
# ~77 px at 1920x1080 — a 3x cut in yaw error for no hardware change.
#
# This is safe because a camera's POSE is a property of its body, not of the
# stream it happens to be delivering: extrinsics solved at high resolution apply
# unchanged to a 640x480 recording, since each profile carries its own
# intrinsics and both share one optical centre. 15 fps because calibration wants
# six still frames, and 1920x1080 BGR8 at 30 fps is 186 MB/s of USB on a bus
# that a second camera is also using.
#
# Not every mode exists on every model (the D435's colour goes to 1920x1080, the
# D455's sensor is 1MP), so these are ATTEMPTS: the first that starts wins.
#
# Ordered by WIDTH first, then height. Width is what matters twice over here: it
# sets the tag's pixel size, and it sets how much of the room is in frame — and a
# frame holding several mapped tags is what lets the joint multi-tag solve fix
# yaw off a metre of baseline instead of one 138mm square. The 4:3 entries sit
# last because they are horizontal crops at unchanged focal length: they cost
# field of view without buying any tag pixels.
CALIB_COLOR_ATTEMPTS = ((1920, 1080, 15), (1280, 800, 15), (1280, 720, 15),
                        (960, 540, 15), (848, 480, 15), (640, 480, 30))
# Depth sizes to try alongside. Depth is enabled SEPARATELY from colour because
# no D4xx does depth above 1280x720, so it cannot simply match a 1920x1080 colour
# stream.
#
# 640x480 DELIBERATELY, not the sensor's best: measured on both cameras, raising
# depth to 1280x720 made every depth-derived quantity worse, because a downsampled
# depth frame is a DENOISED one. The vertical lost valid normals (the D455 went
# from 28661 to 15940 640x480-equivalents — the discontinuity test throws out more
# of a noisier frame), and depth_agreement_mm rose on both cameras (D455 50.6 ->
# 84.2, D435 151.7 -> 219.2) because _depth_at's fixed 5x5 median window spans
# less physical area at higher resolution, so it averages less. Stereo depth on a
# flat low-texture target like a printed tag is exactly where per-pixel noise
# hurts most. Left as a list so a camera lacking 640x480 still negotiates.
CALIB_DEPTH_ATTEMPTS = ((640, 480), (848, 480), (1280, 720))
# Depth only breaks the planar-PnP tie when it beats the other branch by this
# much; a near-tie is noise, not a decision.
DEPTH_TIEBREAK_MARGIN = 0.75
# How much closer to the measured vertical one pose branch must be before it is
# taken as the right one. Measured separation on real frames: 1.3 vs 12.2 deg.
BRANCH_MARGIN_DEG = 3.0
# --- yaw from the wall (find_room_wall) --------------------------------------
# The tag's yaw is its worst axis (a 30 px tag pins heading to a few degrees).
# The wall the tag hangs on fills thousands of depth pixels, so its normal fixes
# the room's FORWARD far better — the same trade the vertical already makes for
# pitch/roll. A surface counts as a wall when its normal sits within this of
# horizontal (i.e. the surface is within this of vertical).
WALL_MAX_TILT_DEG = 25.0
# Enough wall normals, and tight enough, before the depth heading is trusted
# over the tag's. Walls show fewer pixels than a floor across the whole frame,
# so this sits below MIN_LEVEL_NORMALS.
MIN_YAW_NORMALS = 4000
MAX_YAW_SCATTER_DEG = 7.0
# The tag heading is only a few degrees off, so the wall the tag is on is near
# the tag's forward. If the mean-shift walked further than this it locked onto a
# DIFFERENT wall (a side wall ~90 deg away) — reject rather than yaw the room
# onto the wrong wall.
MAX_YAW_CORRECTION_DEG = 20.0

# Resolved at CALL time, not import time — the web page imports this module
# before it loads node.env, so import-time reads would freeze the defaults.
def _env_tag_id():
    return int(os.environ.get("SMARTROOM_TAG_ID", "1"))


def _env_tag_size_mm():
    return float(os.environ.get("SMARTROOM_TAG_SIZE_MM", "173"))


def _env_tag_sizes():
    """{tag id: size_mm} for tags that are NOT the default size.

    The room does not have to be one size of tag, and after this was assumed it
    silently wrecked a calibration: a 235mm tag solved with a 138.4mm corner
    model was mapped at 59% of its true range (PnP translation is linear in the
    assumed size), and the only outward sign was depth disagreeing with the pose
    by 20.7%. Bigger tags are worth having where a camera needs the pixels, so
    the size has to be per-tag.

        SMARTROOM_TAG_SIZES=3:235,4:335        # in node.env, mm, black square

    Anything unlisted falls back to SMARTROOM_TAG_SIZE_MM."""
    sizes = {}
    for part in os.environ.get("SMARTROOM_TAG_SIZES", "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition(":")
        try:
            sizes[int(key.strip())] = float(value.strip())
        except ValueError:
            print(f"ignoring unparseable SMARTROOM_TAG_SIZES entry {part!r} "
                  f"(want id:mm, e.g. 3:235)", file=sys.stderr)
    return sizes


def _env_tag_min_pixels():
    """Ignore tag detections smaller than this many pixels across (0 = off).

    Small tags do not just contribute little to a solve, they pull it: the D455
    anchoring on an 18px tag at 5m mapped its two good tags with 200-330mm of
    position scatter, while a 74px tag sat 2m away in the same frame. Default 0
    so nothing changes for a node whose tags are all small; raise it in node.env
    once there are big tags to prefer."""
    return float(os.environ.get("SMARTROOM_TAG_MIN_PIXELS", "0"))


def corner_model(size_mm):
    """Tag corner model for SOLVEPNP_IPPE_SQUARE (aruco order TL,TR,BR,BL),
    tag-centred, Z out of the tag — identical to calibrate_extrinsics.py."""
    s = size_mm / 2.0
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)


def _env_tag_height_mm():
    """Height of the REFERENCE tag's centre above the floor, mm.

    Named for tag 3 because that is the tag it was measured on, and the old
    SMARTROOM_TAG_HEIGHT_MM said nothing about which tag it meant — which
    mattered once tag 2 was taken down and its 1590 was left behind describing
    a tag that no longer existed. The old name is still honoured so a node that
    has not been updated keeps working.

    This is what fixes the FLOOR: the room frame's origin is the reference tag's
    centre and its vertical is gravity-levelled, so the floor is the plane
    y = +height (the levelled frame's up is -Y). It is only the origin's height
    when the reference tag IS tag 3 — otherwise the origin sits wherever
    SMARTROOM_TAG_ID is, and this number has to be propagated there through
    tags.json."""
    for key in ("SMARTROOM_TAG3_HEIGHT_MM", "SMARTROOM_TAG_HEIGHT_MM"):
        if key in os.environ:
            return float(os.environ[key])
    return 1110.0


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


def rotate180_intrinsics(intr):
    """Intrinsics for a frame that has been rotated 180 degrees.

    The flipped camera's frames are rotated AFTER capture (SMARTROOM_DEPTH_FLIP —
    the rotation is off the capture hot path), but the SDK still reports the
    SENSOR's intrinsics. Feeding those to solvePnP on a rotated image biases the
    pose by the doubled principal-point offset (~1 degree on the D435), so mirror
    the principal point and flip the tangential terms to match the pixels."""
    coeffs = list(getattr(intr, "coeffs", []) or [])
    if len(coeffs) >= 4:
        coeffs[2], coeffs[3] = -coeffs[2], -coeffs[3]  # p1, p2 point the other way
    return type("RotatedIntrinsics", (), {
        "fx": intr.fx, "fy": intr.fy,
        "ppx": (intr.width - 1) - intr.ppx,
        "ppy": (intr.height - 1) - intr.ppy,
        "width": intr.width, "height": intr.height,
        "model": intr.model, "coeffs": coeffs,
    })()


def _average_pose(poses):
    """Chordal L2 mean of several (rvec, tvec) tag poses -> (rvec, tvec).

    One frame's pose is never the pose to keep: a tag seen nearly head-on makes
    planar PnP ill-conditioned, so its per-frame tilt jitters by degrees while the
    reprojection error stays near zero — picking the single lowest-error frame
    reliably picks a degenerate one. Averaging over the burst does not fix the
    conditioning (the depth levelling does) but it does drop the jitter."""
    rotations = np.stack([cv2.Rodrigues(rv)[0] for rv, _ in poses])
    U, _, Vt = np.linalg.svd(rotations.mean(axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:                       # keep it a rotation
        U[:, -1] *= -1
        R = U @ Vt
    tvec = np.median(np.stack([tv.reshape(3) for _, tv in poses]), axis=0)
    return cv2.Rodrigues(R)[0], tvec.reshape(3, 1)


def _minimal_rotation(a, b):
    """Smallest rotation matrix taking unit vector a onto unit vector b."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    s, c = float(np.linalg.norm(v)), float(a @ b)
    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def normals_scale(intr):
    """Multiplier taking a 640x480-tuned normal count to this frame's size.

    _surface_normals walks a fixed 1-in-`step` grid, so its output count is
    proportional to the frame's pixel area. See NORMALS_REF_PIXELS."""
    px = float(getattr(intr, "width", 0) or 0) * float(getattr(intr, "height", 0) or 0)
    return px / NORMALS_REF_PIXELS if px > 0 else 1.0


def _surface_normals(depth_mm, intr, step=4):
    """Per-pixel surface normals (unit, camera frame) from one depth image.

    Returns (points, normals) for the pixels where a normal could be estimated:
    both neighbour pairs must have depth and neither may straddle a depth
    discontinuity (an edge's cross product points anywhere)."""
    h, w = depth_mm.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_mm
    points = np.stack([(us - intr.ppx) / intr.fx * z, (vs - intr.ppy) / intr.fy * z, z], axis=2)
    centre = points[step:-step, step:-step]
    dx = points[step:-step, 2 * step:] - points[step:-step, :-2 * step]
    dy = points[2 * step:, step:-step] - points[:-2 * step, step:-step]
    zc = z[step:-step, step:-step]
    valid = ((zc > 300.0) & (zc < 8000.0)
             & (np.abs(dx[:, :, 2]) < 0.08 * zc) & (np.abs(dy[:, :, 2]) < 0.08 * zc)
             & (z[step:-step, 2 * step:] > 0) & (z[step:-step, :-2 * step] > 0)
             & (z[2 * step:, step:-step] > 0) & (z[:-2 * step, step:-step] > 0))
    normals = np.cross(dx[valid], dy[valid])
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 1e-6
    return centre[valid][keep], normals[keep] / lengths[keep, None]


def find_room_vertical(samples_mm, intr, up_hint, notes=None):
    """The room's true vertical, in CAMERA coordinates, from the depth alone.

    Not "find the floor": the floor is often out of frame (a camera aimed across
    the room sees neither floor nor ceiling), and any single plane can be a desk.
    But EVERY horizontal surface in the room — floor, ceiling, desks, tables —
    shares one normal, so the vertical is the dominant direction among the
    surface normals that point roughly the way the tag pose says is up.
    `up_hint` only seeds that search; mean-shift then re-centres and narrows the
    window so the hint's own error (degrees, for a tag seen head-on) doesn't
    bias the answer. Returns None when too little horizontal surface is in view.

    Being expressed in camera coordinates is the point: it is measured
    independently of the tag pose it is about to correct."""
    up = np.asarray(up_hint, dtype=np.float64)
    up /= np.linalg.norm(up)
    scale = normals_scale(intr)      # counts below are 640x480-equivalents
    floor_min = 2000 * scale
    clouds, normal_sets = [], []
    for depth in samples_mm[:4]:
        points, normals = _surface_normals(depth, intr)
        if len(points):
            clouds.append(points)
            normal_sets.append(normals)
    if not clouds:
        return None
    points, normals = np.concatenate(clouds), np.concatenate(normal_sets)
    normals = np.where((normals @ up)[:, None] < 0, -normals, normals)  # point them up

    used = None
    for window_deg in (30.0, 15.0, 8.0, 5.0):
        near = normals[normals @ up > np.cos(np.radians(window_deg))]
        if len(near) < floor_min:
            break
        # mean of a tight cluster of unit vectors — the dominant surface normal
        up = near.mean(axis=0)
        up /= np.linalg.norm(up)
        used = near
    if used is None or len(used) < floor_min:
        if notes is not None:
            notes.append("too few horizontal-surface normals")
        return None
    scatter_deg = float(np.degrees(np.arccos(np.clip(used @ up, -1.0, 1.0))).mean())

    # Heights of everything along the measured vertical, so the floor (the
    # lowest broad horizontal surface) can be reported as well.
    heights = points @ up
    horizontal = points[(normals @ up) > np.cos(np.radians(8.0))]
    floor_mm = None
    if len(horizontal) > floor_min:
        levels = horizontal @ up
        counts, edges = np.histogram(levels, bins=np.arange(-4000, 4000, 50))
        big = [i for i, c in enumerate(counts) if c > max(1500 * scale, 0.05 * len(horizontal))]
        if big:
            band = levels[np.abs(levels - (edges[big[0]] + 25)) < 100]
            floor_mm = float(np.median(band))       # signed height above the camera
    if notes is not None:
        notes.append(f"normals={len(used)} scatter={scatter_deg:.2f}deg "
                     f"hint_moved={np.degrees(np.arccos(np.clip(np.asarray(up_hint) @ up / np.linalg.norm(up_hint), -1, 1))):.2f}deg "
                     f"span={np.ptp(heights):.0f}mm "
                     f"floor={'%.0f' % floor_mm if floor_mm is not None else 'n/a'}")
    return {"up_cam": up, "normals_used": int(len(used)),
            # the same count at 640x480, which is what the thresholds are quoted
            # in — the raw figure is meaningless without the frame size
            "normals_used_ref": int(len(used) / scale) if scale > 0 else int(len(used)),
            "scatter_deg": round(scatter_deg, 2),
            "floor_below_camera_mm": round(floor_mm, 1) if floor_mm is not None else None}


def find_room_wall(samples_mm, intr, fwd_hint_cam, up_cam, notes=None):
    """The reference wall's outward normal, in CAMERA coordinates, from depth.

    The room's FORWARD is defined by the wall the tag hangs on. That wall fills
    thousands of depth pixels, so its normal is far better conditioned than the
    tag's yaw. This mirrors find_room_vertical, with two differences: a room has
    several walls (unlike its one vertical), so the search is SEEDED with the
    tag's own forward (`fwd_hint_cam`, a few degrees off at worst) and kept in a
    window around it — it locks onto the tag's wall, never a side wall — and only
    normals roughly PERPENDICULAR to the room vertical count, so the floor and
    ceiling can't win. Returns None when too little wall is in view or the search
    wandered onto a different wall.

    Measured in camera coordinates on purpose: independent of the tag yaw it is
    about to replace."""
    fwd = np.asarray(fwd_hint_cam, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(up_cam, dtype=np.float64)
    up /= np.linalg.norm(up)
    min_normals = MIN_YAW_NORMALS * normals_scale(intr)   # 640x480-equivalent
    clouds, normal_sets = [], []
    for depth in samples_mm[:4]:
        points, normals = _surface_normals(depth, intr)
        if len(points):
            clouds.append(points)
            normal_sets.append(normals)
    if not clouds:
        return None
    normals = np.concatenate(normal_sets)
    # vertical surfaces only: normal within WALL_MAX_TILT_DEG of horizontal
    wall_like = np.abs(normals @ up) < np.sin(np.radians(WALL_MAX_TILT_DEG))
    normals = normals[wall_like]
    if len(normals) < min_normals:
        if notes is not None:
            notes.append("too little wall surface for a depth heading")
        return None
    normals = np.where((normals @ fwd)[:, None] < 0, -normals, normals)  # toward the tag's forward

    used = None
    direction = fwd
    for window_deg in (30.0, 15.0, 8.0, 5.0):
        near = normals[normals @ direction > np.cos(np.radians(window_deg))]
        if len(near) < min_normals:
            break
        direction = near.mean(axis=0)
        direction /= np.linalg.norm(direction)
        used = near
    if used is None or len(used) < min_normals:
        if notes is not None:
            notes.append("too few wall normals near the tag's forward")
        return None
    # flatten to pure horizontal (the yaw axis is the room vertical) and reject a
    # search that walked onto a different wall
    horiz = direction - (direction @ up) * up
    if np.linalg.norm(horiz) < 1e-6:
        return None
    horiz /= np.linalg.norm(horiz)
    moved_deg = float(np.degrees(np.arccos(np.clip(horiz @ fwd, -1.0, 1.0))))
    scatter_deg = float(np.degrees(np.arccos(np.clip(used @ direction, -1.0, 1.0))).mean())
    if moved_deg > MAX_YAW_CORRECTION_DEG or scatter_deg > MAX_YAW_SCATTER_DEG:
        if notes is not None:
            notes.append(f"wall heading rejected (moved {moved_deg:.1f}deg, "
                         f"scatter {scatter_deg:.1f}deg, n={len(used)})")
        return None
    if notes is not None:
        notes.append(f"wall normals={len(used)} scatter={scatter_deg:.2f}deg "
                     f"heading_moved={moved_deg:.2f}deg")
    return {"normal_cam": horiz, "normals_used": int(len(used)),
            "scatter_deg": round(scatter_deg, 2), "moved_deg": round(moved_deg, 2)}


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
                           tag_id=None, tag_size_mm=None,
                           out_dir=DEFAULT_OUT, min_tag_px=None):
    """samples: list of (color_bgr, depth_m) with depth ALIGNED to color.
    Returns (ok, message); writes calibration/<serial>.extrinsics.json and a
    debug overlay on success."""
    if not samples:
        return False, "no frames captured"
    if tag_id is None:
        tag_id = _env_tag_id()
    if tag_size_mm is None:
        tag_size_mm = _env_tag_size_mm()
    if min_tag_px is None:
        min_tag_px = _env_tag_min_pixels()
    K, dist = intrinsics_to_cv(intr)

    # Per-tag edge lengths. The room mixes sizes on purpose — a camera that needs
    # more pixels on a tag gets a bigger print — so every corner model below is
    # built for the tag it belongs to, never from one global size. An unlisted tag
    # falls back to tag_size_mm; a MAPPED tag's size comes from tags.json, which
    # records what it was measured with.
    tag_sizes = _env_tag_sizes()

    def size_of(tid):
        return float(tag_sizes.get(int(tid), tag_size_mm))

    def obj_of(tid):
        return corner_model(size_of(tid))

    obj = obj_of(tag_id)          # the reference tag's model

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11),
        cv2.aruco.DetectorParameters())

    # The room's vertical, measured from the depth BEFORE the tag is solved —
    # it both picks the pose branch below and levels the result afterwards, so
    # it must not be seeded from the pose it is about to judge. Plain image-up
    # is seed enough: the mean-shift in find_room_vertical walks it the rest of
    # the way (measured: 5.7-7.2 degrees) for any roughly upright camera.
    samples_mm = [depth * 1000.0 for _, depth in samples]
    level_notes = []
    vertical = find_room_vertical(samples_mm, intr, [0.0, -1.0, 0.0], level_notes)
    # Compared in 640x480-equivalents: the raw count scales with the frame area,
    # so at full colour resolution it would clear a 640x480 threshold off the
    # same sliver of surface that was rightly rejected at 640x480.
    if vertical is not None and (vertical["normals_used_ref"] < MIN_LEVEL_NORMALS
                                 or vertical["scatter_deg"] > MAX_LEVEL_SCATTER_DEG):
        level_notes.append(f"rejected: {vertical['normals_used']} normals "
                           f"({vertical['normals_used_ref']} at 640x480, want "
                           f"{MIN_LEVEL_NORMALS}) at {vertical['scatter_deg']}° scatter is too "
                           f"little horizontal surface to level from")
        vertical = None

    branch_picks = []       # how each detection's branch got chosen (diagnostic)

    def branch_up_error(rvec):
        """How far this pose branch's idea of 'up' is from the measured vertical.

        Assumes tags hang upright, so the tag frame's -Y is up (aruco returns it
        rolled 180 about the tag normal). None when nothing was measured."""
        if vertical is None:
            return None
        up_cam = -cv2.Rodrigues(rvec)[0][:, 1]
        return float(np.degrees(np.arccos(np.clip(up_cam @ vertical["up_cam"], -1.0, 1.0))))

    def corner_depth_error(rvec, tvec, img_pts, depth_m, model=None):
        """Mean |measured depth - PnP-predicted range| over the tag's corners, mm.

        `model` is the corner model of the tag this pose belongs to — with the
        wrong edge length the predicted corner ranges are wrong and this stops
        being a check on anything."""
        if depth_m is None:
            return None
        predicted = (cv2.Rodrigues(rvec)[0] @ (obj if model is None else model).T + tvec).T
        diffs = [abs(z * 1000.0 - float(np.linalg.norm(point)))
                 for (px, py), point in zip(img_pts, predicted)
                 if (z := _depth_at(depth_m, px, py)) is not None]
        return float(np.mean(diffs)) if diffs else None

    def tag_pose(corners_1x4, depth_m=None, tid=None):
        """(reproj_err, rvec, tvec, img_pts) for one detected tag, or None.

        IPPE returns BOTH solutions of the planar-pose ambiguity, and at this
        tag size their reprojection errors are effectively tied — measured, the
        LOWER one is the wrong branch about half the time, and the two sit
        13-15 degrees apart in tilt. So do not choose on reprojection. The
        branches disagree about which way is up, and the depth has already
        measured that from tens of thousands of surface normals, so choose on
        agreement with it; only fall back to the corner depths (and then to
        reprojection) when the vertical isn't available or doesn't separate
        them."""
        img_pts = corners_1x4.reshape(4, 2).astype(np.float64)
        model = obj if tid is None else obj_of(tid)
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(model, img_pts, K, dist,
                                                     flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not count:
            return None
        solutions = []
        for rvec, tvec in zip(rvecs, tvecs):
            proj, _ = cv2.projectPoints(model, rvec, tvec, K, dist)
            err = float(np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1).mean())
            solutions.append((err, corner_depth_error(rvec, tvec, img_pts, depth_m, model),
                              branch_up_error(rvec), rvec, tvec))

        by_up = sorted((s for s in solutions if s[2] is not None), key=lambda s: s[2])
        by_depth = sorted((s for s in solutions if s[1] is not None), key=lambda s: s[1])
        if len(by_up) > 1 and by_up[1][2] - by_up[0][2] > BRANCH_MARGIN_DEG:
            best, how = by_up[0], "vertical"
        elif len(by_depth) > 1 and by_depth[0][1] < DEPTH_TIEBREAK_MARGIN * by_depth[1][1]:
            best, how = by_depth[0], "corner depth"
        else:
            # nothing separated them — either branch is as good as the other
            best, how = min(solutions, key=lambda s: s[0]), "reprojection"
        return best[0], best[3], best[4], img_pts

    # Every tag whose room-frame pose is known: the reference tag by definition,
    # plus any other tag previously measured against it (tags.json). Each entry
    # is that tag's four corners in ROOM coordinates, which is all a joint solve
    # needs — and note a tag's own ORIENTATION barely matters here: getting it
    # wrong by 5 degrees moves its corners by ~6 mm, nothing against a
    # metre-scale spread between tags. It is the tag CENTRES that must be right,
    # and centres come from PnP's translation, the well-conditioned half.
    anchors = _load_room_tags(out_dir)       # {id: (R tag->room, centre mm, size mm)}
    # The reference tag sits at the origin BY DEFINITION, so a tags.json entry for
    # it must never win: that file records where each tag is relative to whatever
    # the reference WAS, so after re-pointing SMARTROOM_TAG_ID at a tag already in
    # the map, the update would have displaced the new origin to its old
    # coordinates and shifted every pose by that offset, silently.
    stale_ref = anchors.pop(tag_id, None)
    if stale_ref is not None:
        level_notes.append(
            f"tags.json still placed the reference tag {tag_id} at "
            f"[{stale_ref[1][0]:.0f}, {stale_ref[1][1]:.0f}, {stale_ref[1][2]:.0f}] mm — "
            f"ignored, it is the origin now; clear tags.json to re-map the room around it")
    known = {tag_id: (np.eye(3), np.zeros(3), size_of(tag_id))}
    known.update(anchors)
    known_corners = {tid: (Q @ corner_model(sz).T).T + p
                     for tid, (Q, p, sz) in known.items()}

    def solve_frame(obs, depth_m):
        """Room->camera pose from EVERY known tag in one frame at once.

        One 138 mm tag gives the solver a 138 mm baseline to read perspective
        from, which is why a camera facing it square can barely tell its two
        pose branches apart. Two tags a metre apart give it a metre — the same
        corner noise then buys an order of magnitude less pose error, with no
        change to the hardware. Returns (reproj_err, rvec, tvec, used_ids)."""
        # Seed from one tag, which is where BOTH ambiguity branches come from,
        # then refine each seed over every corner in the frame. Solving the whole
        # constellation in one shot instead would throw the ambiguity away — the
        # general solvers return a single answer, and near-coplanar tags are
        # still genuinely ambiguous, so the vertical would have nothing to choose
        # between.
        seed_id, seed_corner = next(((t, c) for t, c in obs if t == tag_id), obs[0])
        img_pts = seed_corner.reshape(4, 2).astype(np.float64)
        try:
            count, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_of(seed_id), img_pts, K, dist,
                                                         flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            return None
        if not count:
            return None
        obj_all = np.concatenate([known_corners[tid] for tid, _ in obs])
        img_all = np.concatenate([c.reshape(4, 2).astype(np.float64) for _, c in obs])
        Q, p, _seed_size = known[seed_id]

        scored = []
        for rvec_t, tvec_t in zip(rvecs, tvecs):
            R_cam_room = cv2.Rodrigues(rvec_t)[0] @ Q.T   # cam<-room = cam<-tag . tag<-room
            rvec = cv2.Rodrigues(R_cam_room)[0]
            tvec = tvec_t - R_cam_room @ p.reshape(3, 1)
            if len(obs) > 1:
                ok, rvec, tvec = cv2.solvePnP(obj_all, img_all, K, dist, rvec.copy(),
                                              tvec.copy(), useExtrinsicGuess=True,
                                              flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    continue
            proj, _ = cv2.projectPoints(obj_all, rvec, tvec, K, dist)
            err = float(np.linalg.norm(proj.reshape(-1, 2) - img_all, axis=1).mean())
            scored.append((err, branch_up_error(rvec),
                           corner_depth_error(rvec_t, tvec_t, img_pts, depth_m), rvec, tvec))
        if not scored:
            return None

        by_up = sorted((s for s in scored if s[1] is not None), key=lambda s: s[1])
        by_depth = sorted((s for s in scored if s[2] is not None), key=lambda s: s[2])
        if len(by_up) > 1 and by_up[1][1] - by_up[0][1] > BRANCH_MARGIN_DEG:
            best, how = by_up[0], "vertical"
        elif len(by_depth) > 1 and by_depth[0][2] < DEPTH_TIEBREAK_MARGIN * by_depth[1][2]:
            best, how = by_depth[0], "corner depth"
        else:
            best, how = min(scored, key=lambda s: s[0]), "reprojection"
        branch_picks.append(f"{how} ({len(obs)} tag{'s' if len(obs) > 1 else ''})")
        return best[0], best[3], best[4], [tid for tid, _ in obs]

    # results: (err, room_rvec, room_tvec, raw_rvec, raw_tvec, img_pts,
    #           color_bgr, depth_m, used_ids)
    # room_rvec/room_tvec express the ROOM frame in the camera; raw_* is the
    # reference tag's own pose, kept for the depth cross-check and the overlay.
    results = []
    other_tags = {}   # other tag id -> list of (err, R_room, pos_room_mm) per frame
    seen_ids = set()
    skipped_px = {}   # tag id -> its size in px, for tags dropped as too small
    tag_px_by_id = {}  # every detected tag -> its largest observed size in px
    for color_bgr, depth_m in samples:
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY))
        if ids is None:
            continue
        flat = [int(i) for i in ids.flatten()]
        seen_ids.update(flat)
        # Drop detections too few pixels across to be worth solving from. A small
        # tag does not merely add little, it actively drags the pose: anchoring
        # the D455 on an 18px tag at 5m mapped its two good tags with 200-330mm
        # of scatter, where the 74px tag 2m away was sitting in the same frame.
        # Off by default (min_tag_px 0) so a node with only small tags still
        # calibrates; set SMARTROOM_TAG_MIN_PIXELS once bigger tags are up.
        if min_tag_px > 0:
            keep = []
            for j, tid in enumerate(flat):
                p = corners[j].reshape(4, 2)
                px = float(np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1).max())
                if px >= min_tag_px:
                    keep.append(j)
                else:
                    skipped_px[tid] = round(min(px, skipped_px.get(tid, 1e9)), 1)
            corners = [corners[j] for j in keep]
            flat = [flat[j] for j in keep]
        for j, tid in enumerate(flat):     # per-tag pixel size, for the UI/report
            p = corners[j].reshape(4, 2)
            px = float(np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1).max())
            tag_px_by_id[tid] = round(max(px, tag_px_by_id.get(tid, 0.0)), 1)
        obs = [(tid, corners[j]) for j, tid in enumerate(flat) if tid in known]
        solved = solve_frame(obs, depth_m) if obs else None
        if solved is None:
            continue
        err, rvec, tvec, used_ids = solved
        # the reference tag's own pose, for the depth cross-check and overlay
        anchor_tid = tag_id if tag_id in used_ids else used_ids[0]
        own = tag_pose(corners[flat.index(anchor_tid)], depth_m, anchor_tid)
        if own is None:
            continue
        _, raw_rvec, raw_tvec, img_pts = own
        results.append((err, rvec, tvec, raw_rvec, raw_tvec, img_pts,
                        color_bgr, depth_m, used_ids))

        # Tag chaining: any tag NOT yet in the map gets its pose in the room
        # frame from this frame's solve, so it can join the map for next time.
        R_frame, _ = cv2.Rodrigues(rvec)
        for j, other_id in enumerate(flat):
            if other_id in known:
                continue
            # its OWN size: solving a 235mm tag with the reference tag's 138.4mm
            # model mapped it at 59% of its true range, and wrote that into
            # tags.json where every later joint solve would have trusted it
            other = tag_pose(corners[j], depth_m, other_id)
            if other is None:
                continue
            err2, rvec2, tvec2, _pts2 = other
            pos_room = (R_frame.T @ (tvec2 - tvec)).flatten()      # tag centre, mm
            R_room = R_frame.T @ cv2.Rodrigues(rvec2)[0]           # tag axes in room frame
            other_tags.setdefault(other_id, []).append((max(err, err2), R_room, pos_room))

    if not results:
        listed = ", ".join(str(k) for k in sorted(known))
        dropped = ("; " + ", ".join(f"tag {t} was ignored at {px:.0f} px"
                                    for t, px in sorted(skipped_px.items()))
                   + f" (SMARTROOM_TAG_MIN_PIXELS={min_tag_px:g}) — lower it or "
                     f"use a bigger tag" if skipped_px else "")
        return False, (f"no usable tag seen (36h11 ids detected: {sorted(seen_ids) or 'none'}) — "
                       f"need one of the mapped tags: {listed}{dropped}")

    # Frames that include the reference tag beat ones anchored only through
    # other tags, whose errors stack on top of the map's own.
    direct = [r for r in results if tag_id in r[-1]]
    if direct:
        results = direct
    used_ids = sorted({tid for r in results for tid in r[-1]})
    anchor_id = None if tag_id in used_ids else used_ids[0]

    positions = []
    for err, rvec, tvec, *_ in results:
        R, _ = cv2.Rodrigues(rvec)
        positions.append((-R.T @ tvec).flatten())
    spread = float(np.linalg.norm(np.std(np.array(positions), axis=0))) if len(positions) > 1 else 0.0

    # The reference frame (the frame kept for the debug overlay and the depth
    # cross-check) is still the lowest-reprojection one, but the POSE is the
    # average over the burst — see _average_pose.
    (err, _, _, raw_rvec, raw_tvec, img_pts,
     color_bgr, depth_m, best_ids) = min(results, key=lambda r: r[0])
    # which tag that raw pose belongs to — same rule as anchor_tid above. The
    # depth cross-check and the overlay below need ITS corner model, not the
    # reference tag's, when the reference tag was not the one in view.
    raw_tid = tag_id if tag_id in best_ids else best_ids[0]
    rvec, tvec = _average_pose([(r[1], r[2]) for r in results])
    R, _ = cv2.Rodrigues(rvec)

    # How obliquely this camera sees the tag. A tag viewed head-on makes planar
    # PnP ill-conditioned: its tilt can be several degrees out while the
    # reprojection error stays at zero. Such a view may not define the room.
    R_raw_pose, _ = cv2.Rodrigues(raw_rvec)
    obliquity_deg = round(float(np.degrees(np.arccos(np.clip(abs(
        R_raw_pose[:, 2] @ (raw_tvec.reshape(3) / np.linalg.norm(raw_tvec))), -1.0, 1.0)))), 1)

    # ---- gravity levelling -------------------------------------------------
    # The tag fixes the room frame's origin and yaw; its VERTICAL is only as
    # plumb as whoever hung it, and a camera's own PnP tilt is only as good as
    # its view of the tag. Both are fixed against the vertical measured from the
    # camera's own depth (find_room_vertical), in two separate steps because
    # they are two different errors:
    #
    #   1. the TAG hangs crooked -> the whole room frame is relabelled, so every
    #      camera in it rotates about the tag origin. Measured once, by a camera
    #      that sees the tag obliquely enough to be trusted, and shared through
    #      room_level.json.
    #   2. THIS camera's PnP tilt is off -> its orientation is corrected in
    #      place, keeping the position PnP put it at (translation is the
    #      well-conditioned half of a planar solve) and its yaw.
    #
    # An ANCHORED solve arrives already levelled (tags.json holds levelled
    # poses), so step 1 is skipped for it.
    stored = _load_room_level(out_dir, tag_id)
    defines_room = (vertical is not None and anchor_id is None
                    and obliquity_deg >= MIN_TAG_OBLIQUITY_DEG)

    up_room = (R.T @ vertical["up_cam"]) if defines_room else (
        stored["up"] if stored is not None and anchor_id is None else None)
    if up_room is not None:                  # step 1 — re-level the room frame
        M = _minimal_rotation(up_room, [0.0, -1.0, 0.0])
        tilt_deg = round(float(np.degrees(np.arccos(np.clip(
            np.asarray(up_room) @ [0.0, -1.0, 0.0], -1.0, 1.0)))), 2)
        R = R @ M.T
    else:
        tilt_deg = None
    cam_pos = (-R.T @ tvec).flatten()

    residual_deg = None
    if vertical is not None:                 # step 2 — level this camera in place
        N = _minimal_rotation(R.T @ vertical["up_cam"], [0.0, -1.0, 0.0])
        residual_deg = round(float(np.degrees(np.arccos(np.clip(
            (R.T @ vertical["up_cam"]) @ [0.0, -1.0, 0.0], -1.0, 1.0)))), 2)
        R = R @ N.T
        tvec = (-R @ cam_pos).reshape(3, 1)  # keep the camera where PnP put it

    # ---- yaw from the wall -------------------------------------------------
    # Levelling above took pitch and roll off the tag; yaw is the tag's LAST and
    # worst axis. Replace it with the normal of the wall the tag hangs on,
    # measured from the depth (thousands of pixels vs a 30 px tag). Rotate the
    # frame about the now-correct vertical so that wall normal points along room
    # +Z, keeping the tag's (well-conditioned) position. Skipped — leaving the
    # tag's yaw untouched — whenever no trustworthy wall is in view.
    yaw_deg = None
    yaw_source = "tag corners (no usable wall in view)"
    up_cam_meas = vertical["up_cam"] if vertical is not None else (R @ np.array([0.0, -1.0, 0.0]))
    wall = find_room_wall(samples_mm, intr, R @ np.array([0.0, 0.0, 1.0]),
                          up_cam_meas, level_notes)
    if wall is not None:
        fwd_room = R.T @ wall["normal_cam"]
        fwd_h = fwd_room - (fwd_room @ [0.0, -1.0, 0.0]) * np.array([0.0, -1.0, 0.0])
        if np.linalg.norm(fwd_h) > 1e-6:
            fwd_h /= np.linalg.norm(fwd_h)
            Y = _minimal_rotation(fwd_h, [0.0, 0.0, 1.0])   # rotation about the vertical
            yaw_deg = wall["moved_deg"]
            yaw_source = (f"{wall['normals_used']} wall normals "
                          f"({wall['scatter_deg']}° scatter)")
            R = R @ Y.T
            tvec = (-R @ cam_pos).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)

    if vertical is None:
        level_source = ("room_level.json" if up_room is not None else
                        f"inherited from tag {anchor_id}" if anchor_id is not None else
                        "NOT LEVELLED — no horizontal surface in view")
    else:
        level_source = (f"{vertical['normals_used']} horizontal-surface normals "
                        f"({vertical['normals_used_ref']} at 640x480, "
                        f"{vertical['scatter_deg']}° scatter)"
                        + ("" if defines_room else
                           f", tag seen too head-on ({obliquity_deg}°) to define the room"))

    # Native cross-check: the camera's own depth at each tag corner vs the
    # distance the PnP pose predicts for that corner. Independent measurements —
    # small disagreement means both the pose and the printed tag size are right.
    # Uses the DETECTED tag's raw pose (obj is that tag's corner model).
    R_raw, _ = cv2.Rodrigues(raw_rvec)
    pnp_corners_cam = (R_raw @ obj_of(raw_tid).T + raw_tvec).T  # 4x3, mm, camera frame
    diffs = []
    for (px, py), pnp_pt in zip(img_pts, pnp_corners_cam):
        z = _depth_at(depth_m, px, py)
        if z is None:
            continue
        diffs.append(abs(z * 1000.0 - float(np.linalg.norm(pnp_pt))))
    depth_agreement_mm = round(float(np.mean(diffs)), 1) if diffs else None

    level = _save_room_level(vertical if defines_room else None, up_room, tilt_deg,
                             tvec, serial, tag_id, tag_size_mm, out_dir)

    h, w = color_bgr.shape[:2]
    debug_dir = out_dir / "debug" / serial
    debug_dir.mkdir(parents=True, exist_ok=True)
    overlay = color_bgr.copy()
    cv2.aruco.drawDetectedMarkers(overlay, [img_pts.reshape(1, 4, 2).astype(np.float32)])
    cv2.drawFrameAxes(overlay, K, dist, raw_rvec, raw_tvec, size_of(raw_tid) * 0.75)
    cv2.imwrite(str(debug_dir / "extrinsic_live.jpg"), overlay)

    # The tag's longest edge in pixels — the single number that bounds how well
    # this solve could possibly do. Corner-localisation noise is fixed in PIXELS
    # (~0.3 px), so the angular error of the pose scales as 1/tag_px: the 138 mm
    # tag is ~26 px across at 640x480 and ~77 px at 1920x1080. Recorded here so
    # a recalibration can be compared against the one it replaced.
    tag_px = float(np.linalg.norm(img_pts - np.roll(img_pts, 1, axis=0), axis=1).max())

    out_path = out_dir / f"{serial}.extrinsics.json"
    _atomic_write_json(out_path, {
        "schema_version": "1",
        "camera_id": serial,
        "camera": camera_name,
        "source": "realsense_native",  # factory intrinsics + depth cross-check
        "node": socket.gethostname(),
        "frame": "tag: origin=center, X=right, Y=up, Z=out of tag (gravity-levelled); units mm",
        "tag": {"family": "36h11", "id": tag_id, "size_mm": tag_size_mm},
        "image_size": [w, h],
        "tag_pixels": round(tag_px, 1),           # longest tag edge; bounds the angular accuracy
        # Per-tag detail, so a report can say WHY a solve came out as it did
        # without re-deriving it: what each tag measured, how big it was believed
        # to be, and which were dropped for being too small to help.
        "tag_pixels_by_id": {str(k): v for k, v in sorted(tag_px_by_id.items())},
        "tag_sizes_mm_by_id": {str(k): size_of(k) for k in sorted(tag_px_by_id)},
        "tags_ignored_px": {str(k): v for k, v in sorted(skipped_px.items())},
        "min_tag_pixels": min_tag_px,
        "rvec": rvec.flatten().tolist(),          # room -> camera rotation (Rodrigues)
        "tvec_mm": tvec.flatten().tolist(),       # tag origin in camera frame
        "rotation_cam_to_room": R.T.tolist(),
        "camera_position_mm": [round(float(v), 1) for v in cam_pos],
        "reprojection_error_px": round(err, 3),
        "depth_agreement_mm": depth_agreement_mm,
        "frames_used": len(results),
        "solved_from_tags": used_ids,
        "position_spread_mm": round(spread, 1),
        # how the frame's vertical was fixed — see find_room_vertical
        "levelled": {"source": level_source, "tag_tilt_deg": tilt_deg,
                     "camera_tilt_corrected_deg": residual_deg,
                     "yaw_source": yaw_source, "yaw_from_wall_deg": yaw_deg,
                     "tag_obliquity_deg": obliquity_deg,
                     "defines_room_vertical": defines_room, "notes": level_notes,
                     "branch_chosen_by": sorted(set(branch_picks)),
                     **({k: v for k, v in vertical.items() if k != "up_cam"}
                        if vertical else {})},
        # present when the pose came via a secondary tag (camera couldn't see
        # the reference tag) — chained through tags.json, so two PnP errors stack
        **({"anchored_by_tag": anchor_id} if anchor_id is not None else {}),
        "calibrated_at": dt.datetime.now().astimezone().isoformat(),
    })

    M = _minimal_rotation(up_room, [0.0, -1.0, 0.0]) if up_room is not None else np.eye(3)
    tag_notes = _save_room_tags(other_tags, M, serial, tag_id, tag_size_mm, out_dir,
                                size_of, tag_sizes)

    # How well YAW is pinned down. When the wall gave us the heading (yaw_deg is
    # set), the tag's size no longer matters — the wall normal, not the tag
    # corners, fixes forward. Only when we FELL BACK to the tag's corners does
    # their pixel count bound the yaw: a 20 px tag is worth several degrees, and
    # the noise is a static bias so averaging frames cannot help. (tag_px is
    # measured above, next to the record it is written into.)
    yaw_note = ""
    if yaw_deg is not None:
        yaw_note = f"; yaw from the wall ({yaw_source}), tag corners {yaw_deg}° off"
    elif tag_px < MIN_TAG_PIXELS:
        yaw_note = (f"; !! no usable wall in view, so yaw fell back to the tag — only "
                    f"{tag_px:.0f} px across, too small to pin heading (spread between runs: "
                    f"several degrees). Aim the camera at a flat wall, use a bigger tag, "
                    f"move closer, or capture at higher resolution")
    if skipped_px:
        yaw_note += ("; ignored " + ", ".join(
            f"tag {t} ({px:.0f} px)" for t, px in sorted(skipped_px.items()))
            + f" as under SMARTROOM_TAG_MIN_PIXELS={min_tag_px:g}")
    edge = min(img_pts[:, 0].min(), img_pts[:, 1].min(),
               w - img_pts[:, 0].max(), h - img_pts[:, 1].max())
    if edge < 0.08 * min(w, h):
        yaw_note += (f"; !! the tag sits {edge:.0f} px from the frame edge, where lens "
                     f"distortion is worst — re-aim so it lands nearer the centre")

    dist_m = float(np.linalg.norm(cam_pos)) / 1000.0
    range_mm = float(np.linalg.norm(raw_tvec))
    agree = (f", depth agrees within {depth_agreement_mm} mm" if depth_agreement_mm is not None
             else ", no depth at corners")
    if depth_agreement_mm is not None and depth_agreement_mm > 0.04 * range_mm:
        # the two independent measurements of the tag's range disagree by more
        # than the sensor's spec — one of them (usually the depth) needs a look
        agree += (f" (!! {100 * depth_agreement_mm / range_mm:.1f}% of the {range_mm:.0f} mm "
                  f"range — check this camera's depth calibration or the tag size)")
    via = (f" from tag{'s' if len(used_ids) > 1 else ''} "
           f"{', '.join(str(t) for t in used_ids)}"
           + (f" (anchored — the reference tag {tag_id} was not in view)"
              if anchor_id is not None else ""))
    level_note = f"; levelled off {level_source}"
    if tilt_deg is not None:
        level_note += f", tag hangs {tilt_deg}° off plumb"
    if residual_deg:
        level_note += f", camera tilt corrected by {residual_deg}°"
    floor_note = ""
    if level and level.get("measured_floor_mm") is not None:
        floor_note = (f"; lowest broad horizontal surface is {level['measured_floor_mm']:.0f} mm "
                      f"below the tag — the floor, unless it is a desk "
                      f"(SMARTROOM_TAG3_HEIGHT_MM says {_env_tag_height_mm():.0f})")
    return True, (f"camera at [{cam_pos[0]:.0f}, {cam_pos[1]:.0f}, {cam_pos[2]:.0f}] mm in room frame"
                  f"{via} ({dist_m:.2f} m from origin), reproj {err:.2f} px over {len(results)} frame(s)"
                  f"{agree}{level_note}{floor_note}{yaw_note}{tag_notes} — saved {out_path.name}")


LEVEL_FILENAME = "room_level.json"


def _load_room_level(out_dir, tag_id=None):
    """room_level.json -> {"up": measured vertical in the RAW tag frame, ...},
    or None. Shared by every camera: one tag, one room, one vertical.

    `up_in_tag_frame` is expressed in the REFERENCE TAG's own axes, so a file
    measured against a different tag is not merely stale, it is in the wrong
    basis: reusing it would tilt the room frame by however differently the two
    tags happen to be mounted. Pass tag_id to reject that."""
    try:
        data = json.loads((out_dir / LEVEL_FILENAME).read_text())
        up = np.array(data["up_in_tag_frame"], dtype=np.float64)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if tag_id is not None:
        stored_id = (data.get("reference_tag") or {}).get("id")
        if stored_id is not None and int(stored_id) != int(tag_id):
            return None
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-6:
        return None
    data["up"] = up / np.linalg.norm(up)
    return data


def _save_room_level(plane, up_room, tilt_deg, tvec, serial, tag_id, tag_size_mm,
                     out_dir):
    """Record the room's measured vertical (and, when the plane seen was the
    floor, how far the tag centre sits above it) for cameras that can't see a
    horizontal plane themselves.

    The floor height is REPORTED, never auto-applied: the lowest broad
    horizontal surface can just as well be a platform, and
    SMARTROOM_TAG3_HEIGHT_MM stays the authority until a human confirms."""
    if plane is None or up_room is None:
        return _load_room_level(out_dir, tag_id)
    floor_mm = None
    if plane["floor_below_camera_mm"] is not None:
        # tag centre's height above the floor: the tag origin sits at tvec in
        # camera coordinates, and the floor is that far below the camera.
        floor_mm = float(plane["up_cam"] @ tvec.reshape(3) - plane["floor_below_camera_mm"])
    previous = _load_room_level(out_dir, tag_id) or {}
    # Compared in 640x480-equivalents: the cameras calibrate at DIFFERENT
    # resolutions (the D455 caps at 1280x800, the D435 reaches 1920x1080), so raw
    # counts would hand the room's vertical to whichever camera has more pixels
    # rather than to whichever sees more room.
    prev_by = previous.get("measured_by") or {}
    incumbent = prev_by.get("normals_used_ref", prev_by.get("normals_used", 0))
    if prev_by.get("camera") != serial and incumbent > plane["normals_used_ref"]:
        return previous          # someone else measured it off more surface
    level = {
        "schema_version": "1",
        "reference_tag": {"family": "36h11", "id": tag_id, "size_mm": tag_size_mm},
        "frame": "vertical measured from the camera's own depth, expressed in the raw tag frame",
        "up_in_tag_frame": [round(float(v), 6) for v in up_room],
        "tag_tilt_deg": tilt_deg,
        "measured_by": {"camera": serial, "normals_used": plane["normals_used"],
                        "normals_used_ref": plane["normals_used_ref"],
                        "scatter_deg": plane["scatter_deg"]},
        # informational — SMARTROOM_TAG3_HEIGHT_MM remains what capture.py embeds
        "measured_floor_mm": round(floor_mm, 1) if floor_mm is not None else
                             previous.get("measured_floor_mm"),
        "configured_floor_mm": _env_tag_height_mm(),
        "measured_at": dt.datetime.now().astimezone().isoformat(),
    }
    _atomic_write_json(out_dir / LEVEL_FILENAME, level)
    level["up"] = np.asarray(up_room, dtype=np.float64)
    return level


def _load_room_tags(out_dir):
    """tags.json -> {tag id: (Q rotation tag->room, p position mm, size_mm)}.

    The recorded size travels with each anchor: the joint solve reconstructs
    every known tag's corners in the room, and doing that with the wrong edge
    length misplaces them regardless of how good the centre is."""
    try:
        data = json.loads((out_dir / "tags.json").read_text())
    except (OSError, ValueError):
        return {}
    if not data.get("levelled"):
        return {}   # written before gravity levelling — in the old tilted frame
    anchors = {}
    for key, entry in (data.get("tags") or {}).items():
        try:
            anchors[int(key)] = (
                np.array(entry["rotation_tag_to_room"], dtype=np.float64),
                np.array(entry["position_mm"], dtype=np.float64),
                float(entry["size_mm"]),
            )
        except (KeyError, ValueError, TypeError):
            continue
    return anchors


def _save_room_tags(other_tags, level_rotation, serial, ref_tag_id, tag_size_mm,
                    out_dir, size_of=None, explicit=None):
    """Merge every chained tag's room-frame pose into calibration/tags.json —
    the shared room tag map that capture.py embeds into each recording's
    metadata. `level_rotation` re-levels the poses the same way the camera's own
    extrinsics were, so anchors and cameras stay in the one frame. Returns a
    short note for the status message ('' if no other tags were seen)."""
    if not other_tags:
        return ""
    path = out_dir / "tags.json"
    try:
        tag_map = json.loads(path.read_text())
        if not tag_map.get("levelled"):
            raise ValueError("pre-levelling tag map")
    except (OSError, ValueError):
        tag_map = {"schema_version": "1", "tags": {}}
    tag_map.update({
        "frame": f"tag {ref_tag_id}: origin=center, X=right, Y=up (gravity-levelled), "
                 f"Z=out of tag; units mm",
        "levelled": True,
        "reference_tag": {"family": "36h11", "id": ref_tag_id, "size_mm": tag_size_mm},
    })
    notes = []
    for other_id, observations in sorted(other_tags.items()):
        # A hand-measured entry ("source": "measured" — tape measure beats PnP on
        # a 35 px tag, and the joint solve is only ever as accurate as this map)
        # is never overwritten by a solve.
        existing = tag_map["tags"].get(str(other_id)) or {}
        if existing.get("source") == "measured":
            notes.append(f"tag {other_id} left at its measured pose")
            continue
        # best (lowest joint reprojection error) observation gives the rotation;
        # the position is the mean, with the spread as the consistency check
        observations.sort(key=lambda o: o[0])
        best_err, R_room, _ = observations[0]
        R_room = level_rotation @ R_room
        positions = np.array([o[2] for o in observations]) @ level_rotation.T
        pos = positions.mean(axis=0)
        spread = float(np.linalg.norm(positions.std(axis=0))) if len(positions) > 1 else 0.0
        tag_map["tags"][str(other_id)] = {
            "position_mm": [round(float(v), 1) for v in pos],
            "rotation_tag_to_room": R_room.tolist(),
            # the size this tag was actually SOLVED with, not the reference
            # tag's — a wrong value here silently rescales every future joint
            # solve that uses this anchor
            "size_mm": (size_of(other_id) if size_of else tag_size_mm),
            "reprojection_error_px": round(best_err, 3),
            "position_spread_mm": round(spread, 1),
            "frames_used": len(observations),
            "source": "solved",
            "observed_by": serial,
            "measured_at": dt.datetime.now().astimezone().isoformat(),
        }
        # State the size each new tag was solved at. It is the one input nothing
        # downstream can sanity-check on its own, and getting it wrong scales the
        # tag's whole position — say it out loud, and flag when it was ASSUMED
        # rather than configured.
        used_size = tag_map["tags"][str(other_id)]["size_mm"]
        assumed = "" if (explicit and other_id in explicit) else " ASSUMED - set SMARTROOM_TAG_SIZES"
        notes.append(f"tag {other_id} at [{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}] mm "
                     f"({used_size:g} mm{assumed})")
    _atomic_write_json(path, tag_map)
    return "; " + ", ".join(notes) + " (room frame, saved to tags.json)"


def main():
    import pyrealsense2 as rs  # standalone mode only; the page imports us without it

    # standalone runs need node.env too (the page loads it for the in-process path)
    try:
        for line in (PROJECT_ROOT / "node.env").read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    except OSError:
        pass

    ap = argparse.ArgumentParser(description="AprilTag extrinsic calibration for a RealSense camera.")
    ap.add_argument("--serial", default=None, help="camera USB serial (default: first device)")
    ap.add_argument("--tag-id", type=int, default=_env_tag_id())
    ap.add_argument("--tag-size-mm", type=float, default=_env_tag_size_mm())
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    # Solve the tag at the camera's HIGHEST colour mode, not the recording
    # profile — see CALIB_COLOR_ATTEMPTS. Depth stays low (no D4xx does depth
    # above 1280x720, and align() reprojects it onto colour anyway); it only
    # feeds the vertical and the range cross-check.
    profile = None
    for width, height, fps in CALIB_COLOR_ATTEMPTS:
        for dw, dh in CALIB_DEPTH_ATTEMPTS:
            config = rs.config()
            if args.serial:
                config.enable_device(args.serial)
            config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            try:
                profile = pipeline.start(config)
                break
            except RuntimeError:
                continue
        if profile is not None:
            break
    if profile is None:
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
