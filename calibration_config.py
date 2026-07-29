#!/usr/bin/env python3
"""Every constant and node.env setting that intrinsic or extrinsic calibration
depends on, in one place.

They used to be spread across realsense_extrinsics.py, calibrate_camera.py and
both web pages, and the cost was not tidiness — it was correctness. Four
independent copies of the node.env parser meant a bug fixed in one stayed in the
other three (an inline `# comment` was kept as part of the value, so
SMARTROOM_TAG_ID=4 parsed as "4  # floor tag" and int() threw). A threshold
tuned against 640x480 frames was compared against counts from 1920x1080 ones. A
per-tag height lookup was added while the two call sites kept asking for the
global one. Each of those is a wrong number that looks like a right number.

So: thresholds here, env readers here, one parser. If a calibration decision
depends on a number, the number lives in this file.

WHAT IS AND IS NOT HERE
  Here      thresholds and gates, capture profiles used FOR calibrating,
            checkerboard geometry, the node.env keys and their parsing.
  Not here  the measured results (calibration/*.json — machine-generated), and
            the live/recording profiles in realsense_depth_page.py, which are a
            capture concern rather than a calibration one.

NODE.ENV KEYS
  SMARTROOM_TAG_ID              reference tag; its centre is the room origin
  SMARTROOM_TAG_SIZE_MM         black-square edge for tags not listed below
  SMARTROOM_TAG_SIZES           per tag, e.g. 3:235,4:335 — sizes may differ
  SMARTROOM_TAG_HEIGHTS         per tag, centre above the floor, e.g. 3:1330,4:0
  SMARTROOM_TAG_FLOOR           tags lying flat, whose 'up' is their normal
  SMARTROOM_TAG_MIN_PIXELS      ignore detections smaller than this (0 = off)
  SMARTROOM_TAG34_DISTANCE_MM   tape-measured tag3<->tag4, cross-checked
  SMARTROOM_TAG4_WALL_OFFSET_MM tape-measured tag4 off tag3's wall
  SMARTROOM_TAG_HEIGHT_MM       deprecated single height; read only as a fallback
  SMARTROOM_TAG3_HEIGHT_MM      likewise
"""

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_ROOT / "calibration"
LEVEL_FILENAME = "room_level.json"
TAGS_FILENAME = "tags.json"

# ── node.env ──────────────────────────────────────────────────────────────────


def load_node_env(path=None, override=False):
    """Apply per-node overrides from <repo>/node.env (KEY=VALUE, # comments).

    THE one parser. Both `# whole line` and `value  # trailing` comments are
    stripped; a trailing comment needs whitespace before the `#` so a value that
    contains one (a URL fragment, say) survives. The real environment wins unless
    `override`, so a variable set for one run beats the file."""
    path = Path(path) if path else PROJECT_ROOT / "node.env"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    applied = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = re.split(r"\s+#", value.strip(), maxsplit=1)[0].strip()
        if key and (override or key not in os.environ):
            os.environ[key] = value
            applied[key] = value
    return applied


def _float_env(key):
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return None


def _id_map(key):
    """Parse `3:235,4:335` into {3: 235.0, 4: 335.0}, complaining about junk."""
    out = {}
    for part in os.environ.get(key, "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        tag, _, value = part.partition(":")
        try:
            out[int(tag.strip())] = float(value.strip())
        except ValueError:
            print(f"ignoring unparseable {key} entry {part!r} (want id:mm)", file=sys.stderr)
    return out


def _id_set(key):
    out = set()
    for part in os.environ.get(key, "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                out.add(int(part))
            except ValueError:
                print(f"ignoring unparseable {key} entry {part!r}", file=sys.stderr)
    return out


# ── the tags ──────────────────────────────────────────────────────────────────

TAG_FAMILY = "36h11"
DEFAULT_TAG_ID = 1
DEFAULT_TAG_SIZE_MM = 173.0


def tag_id():
    """The reference tag: its centre is the room frame's origin."""
    try:
        return int(os.environ.get("SMARTROOM_TAG_ID", DEFAULT_TAG_ID))
    except ValueError:
        return DEFAULT_TAG_ID


def tag_size_mm():
    """Black-square edge for any tag not named in SMARTROOM_TAG_SIZES."""
    return _float_env("SMARTROOM_TAG_SIZE_MM") or DEFAULT_TAG_SIZE_MM


def tag_sizes():
    """{id: mm} for tags that differ from the default.

    Sizes differ on purpose: pose error scales as 1/tag_pixels, so a camera with
    a wide field of view needs a physically bigger tag for the same accuracy.
    Assuming one size mapped a 235mm tag at 59% of its true range, because PnP
    translation is linear in the assumed edge length."""
    return _id_map("SMARTROOM_TAG_SIZES")


def tag_heights():
    """{id: centre height above the floor, mm}.

    The floor plane comes from the REFERENCE tag's height, and which tag that is
    is a choice — a tag on the floor is 0, a wall tag is not."""
    return _id_map("SMARTROOM_TAG_HEIGHTS")


def tag_height_mm(tag_id_=None):
    """That tag's height above the floor, or None if nothing says.

    None rather than a guess. A stale default (this used to fall back to 1110mm)
    is indistinguishable downstream from a measurement, and it fabricated a
    1099mm 'contradiction' against a floor tag configured as 0 while the depth
    had measured 11mm — correct, and reported as broken."""
    heights = tag_heights()
    if tag_id_ is not None and int(tag_id_) in heights:
        return heights[int(tag_id_)]
    for key in ("SMARTROOM_TAG3_HEIGHT_MM", "SMARTROOM_TAG_HEIGHT_MM"):
        value = _float_env(key)
        if value is not None:
            return value
    return None


def floor_tags():
    """Tags lying flat, whose 'up' is their own normal (+Z) not their -Y.

    The pose-ambiguity check compares a branch's idea of up against the measured
    vertical. For an upright tag that is the tag frame's -Y; for one on the floor
    -Y is horizontal, so the test is ~90 degrees out for both branches and
    separates nothing."""
    return _id_set("SMARTROOM_TAG_FLOOR")


def tag_min_pixels():
    """Ignore tag detections smaller than this across (0 = off).

    A small tag does not merely add little, it drags the solve: anchoring on an
    18px tag 5m away mapped two good tags with 200-330mm of scatter while a 74px
    tag sat 2m away in the same frame."""
    return _float_env("SMARTROOM_TAG_MIN_PIXELS") or 0.0


def measured_pair():
    """(tag3<->tag4 mm, tag4 off tag3's wall mm) from a tape, or Nones.

    A solved map cannot tell that it is wrong — the joint solve removes scatter,
    not bias — so these are the only independent facts available."""
    return (_float_env("SMARTROOM_TAG34_DISTANCE_MM"),
            _float_env("SMARTROOM_TAG4_WALL_OFFSET_MM"))


# ── extrinsic solve thresholds ────────────────────────────────────────────────

# How obliquely a camera must see the tag before its solve is trusted to define
# the room's vertical. Below this the planar-PnP tilt is not observable enough.
MIN_TAG_OBLIQUITY_DEG = 20.0

# How much horizontal surface the depth must show before its measured vertical is
# trusted over the tag's. A camera staring at a wall sees a few thousand normals
# off one desk — not enough to overrule anything.
MIN_LEVEL_NORMALS = 15000
MAX_LEVEL_SCATTER_DEG = 6.0

# Normal COUNTS above are quoted for a 640x480 frame. _surface_normals walks a
# fixed 1-in-step grid, so its output scales with pixel AREA, not with how much
# room is in view — at 1920x1080 the same starved sliver of desk yields 6.75x
# more normals and would clear a 640x480 threshold. Scale by area (normals_scale)
# so a gate means the same thing at any resolution.
NORMALS_REF_PIXELS = 640 * 480

# Below this the tag is too few pixels across for its corners to pin down yaw.
MIN_TAG_PIXELS = 60.0

# Depth only breaks the planar-PnP tie when it beats the other branch by this
# much; a near-tie is noise, not a decision.
DEPTH_TIEBREAK_MARGIN = 0.75

# How much closer to the measured vertical one branch must be before it is taken
# as the right one. Measured separation on real frames: 1.3 vs 12.2 deg.
BRANCH_MARGIN_DEG = 3.0

# ── yaw from the wall ─────────────────────────────────────────────────────────
# The tag's yaw is its worst axis (a 30px tag pins heading to a few degrees). The
# wall it hangs on fills thousands of depth pixels, so its normal fixes the room's
# forward far better — the same trade the vertical already makes for pitch/roll.
WALL_MAX_TILT_DEG = 25.0
MIN_YAW_NORMALS = 4000
MAX_YAW_SCATTER_DEG = 7.0
MAX_YAW_CORRECTION_DEG = 20.0

# ── capture profiles used FOR calibrating ─────────────────────────────────────
# Not the live/recording profiles — those live in realsense_depth_page.py.
#
# Extrinsics are solved at the best resolution the camera offers, because corner
# noise is fixed in PIXELS (~0.3px) so pose error scales as 1/tag_px, and
# tag_px = f*size/range with f proportional to capture width. Safe because a
# camera's POSE is a property of its body, not of the stream: extrinsics solved
# at high resolution apply unchanged to a 640x480 recording, since each profile
# carries its own intrinsics and both share one optical centre.
#
# Ordered by WIDTH, then height. Width sets the tag's pixel size AND how much of
# the room is in frame — and a frame holding several mapped tags is what lets the
# joint solve fix yaw off a metre of baseline instead of one small square. The
# 4:3 entries sit last: they are horizontal crops at unchanged focal length, so
# they cost field of view without buying any tag pixels. Not every mode exists on
# every model (the D435 reaches 1920x1080; the D455's 1MP sensor stops at
# 1280x800), so these are ATTEMPTS — the first that starts wins.
CALIB_COLOR_ATTEMPTS = ((1920, 1080, 15), (1280, 800, 15), (1280, 720, 15),
                        (960, 540, 15), (848, 480, 15), (640, 480, 30))

# Depth is enabled SEPARATELY from colour because no D4xx does depth above
# 1280x720, so it cannot match a 1920x1080 colour stream.
#
# 640x480 DELIBERATELY, not the sensor's best: measured on both cameras, raising
# depth to 1280x720 made every depth-derived quantity worse, because a
# downsampled depth frame is a DENOISED one. The vertical lost valid normals (the
# D455 went 28661 -> 15940 640x480-equivalents, the discontinuity test rejecting
# more of a noisier frame) and depth_agreement_mm rose on both (D455 50.6 -> 84.2,
# D435 151.7 -> 219.2) because the fixed 5x5 median window spans less physical
# area at higher resolution. Stereo depth on a flat low-texture printed tag is
# exactly where per-pixel noise hurts most.
CALIB_DEPTH_ATTEMPTS = ((640, 480), (848, 480), (1280, 720))

# ── intrinsic (checkerboard) calibration ──────────────────────────────────────
# The DFvision Q18-100-4.5 glass plate: 18x18 squares on 100x100mm at 4.5mm, so
# 17x17 INNER corners. It is small — hold it 15-40cm from the lens. The printed
# paper board is 9x6 at 25mm.
BOARD_COLS = 17
BOARD_ROWS = 17
BOARD_SQUARE_MM = 4.5
INTRINSIC_FRAMES = 15
INTRINSIC_MIN_MOVE_PX = 40.0
# RMS reprojection error above this means the intrinsic solve is not usable.
MAX_INTRINSIC_RMS_PX = 1.0
# Above this, a joint solve has not reconciled its tags: the corners of two
# mapped anchors cannot both be where the map says. Consistent solves here run
# 0.2-3 px; a wrong anchor produced 150.
MAX_REPROJ_PX = 8.0


def normals_scale(intr):
    """Multiplier taking a 640x480-tuned normal count to this frame's size."""
    px = float(getattr(intr, "width", 0) or 0) * float(getattr(intr, "height", 0) or 0)
    return px / NORMALS_REF_PIXELS if px > 0 else 1.0
