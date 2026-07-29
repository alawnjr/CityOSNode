#!/usr/bin/env python3
"""Live RGB + depth view + recorder for Intel RealSense cameras, on port 8001.

Serves http://<node>.local:8001 with one section per connected RealSense
device: color and colorized depth side by side (depth aligned to color), click
any pixel for its distance in meters, per-camera AprilTag extrinsic
calibration, and recording (triggered by capture.py via POST /record/start).

ARCHITECTURE — one WORKER PROCESS per camera: the capture/align/record work
for two 30fps cameras is more than one Python process schedules cleanly (the
GIL made the cameras steal each other's frames), so the HTTP front end spawns
a subprocess per camera (multiprocessing "spawn") and talks to it over a
command pipe + a view-frame queue. Endpoints are unchanged from the
single-process page. If the set of plugged cameras changes, a watchdog exits
the whole page so systemd respawns it into a clean re-enumeration.

TIMESTAMPS — every recorded frame carries the SDK's own hardware timestamp
(librealsense global time: the sensor's mid-exposure time mapped into the
host clock, ms since epoch) in the timestamps CSV's hw_timestamp_ms column.
Frames from different cameras are matched by that column (~1-2 ms accuracy),
and each stream's room-frame extrinsics in metadata.json then put the two
cameras' 3D points in the same coordinate system.

Run with the venv python (pyrealsense2 is built from source on the Pi by
setup_realsense_pi.sh — no aarch64 wheel exists):

    ~/CityOS/.venv/bin/python ~/CityOS/realsense_depth_page.py
"""
import collections
import csv
import datetime as dt
import fcntl
import json
import math
import multiprocessing
import re
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import realsense_extrinsics

# Frame threads block on the GIL behind other threads for multiples of the
# switch interval; the default 5ms costs a 33ms frame budget fast.
sys.setswitchinterval(0.001)


# Per-node overrides from ~/CityOS/node.env (gitignored, machine-local), same
# loader as capture.py / smartroom_video_page.py.
def _load_node_env():
    try:
        lines = (Path.home() / "CityOS" / "node.env").read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        # Trailing comment after the value. Without this,
        #     SMARTROOM_TAG_ID=4   # the floor tag
        # yields "4   # the floor tag" and the first int() of it dies. Only " #"
        # (preceded by whitespace) counts, so a value containing # survives.
        value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if key and key not in os.environ:
            os.environ[key] = value


_load_node_env()

PORT = int(os.environ.get("SMARTROOM_DEPTH_PORT", "8001"))
STREAM_BOUNDARY = "frame"
IDLE_TIMEOUT = 5.0   # stop a camera pipeline this many seconds after its last client leaves
FIRST_FRAME_TIMEOUT = 10.0  # pipeline start + first frames can take a few seconds
# A profile SWAP additionally negotiates: each colour mode the sensor lacks costs a
# failed pipeline.start() of a second or more before the next is tried, so the swap
# needs materially longer than a plain start. Too short here does not merely delay
# the calibration, it aborts it.
SWAP_FRAME_TIMEOUT = 30.0
JPEG_QUALITY = 70
# Browser-bound frame rate cap (newest frame wins, so slow wifi sees lower fps
# instead of latency), and the colorize/JPEG budget for the live view.
VIEW_FPS = float(os.environ.get("SMARTROOM_DEPTH_VIEW_FPS", "30"))
VIEW_ENCODE_FPS = float(os.environ.get("SMARTROOM_DEPTH_VIEW_ENCODE_FPS", "12"))
# Comma-separated camera serials whose frames are rotated 180 degrees (for
# upside-down mounted cameras); the raw depth grid rotates too.
FLIP_SERIALS = {s.strip() for s in os.environ.get("SMARTROOM_DEPTH_FLIP", "").split(",") if s.strip()}

try:
    import cv2
    import pyrealsense2 as rs
    RS_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on node state
    rs = None
    RS_IMPORT_ERROR = str(exc)

# Depth and color at the same resolution so the aligned panes map 1:1.
# Per-model capture profiles (recordings run at the pipeline rate); override
# with e.g. SMARTROOM_DEPTH_PROFILE_D455=848x480@30 in node.env. On USB 2 the
# low fallbacks apply (the whole USB 2 bus is ~40 MB/s).
#
# 848x480 rather than 640x480: the 4:3 modes are a horizontal CROP of the sensor
# at UNCHANGED focal length, so they throw away field of view for nothing. The
# factory fx of 616 (D435) at 640 wide implies a 54.9 deg horizontal FOV against
# a 69.4 deg datasheet figure — 14.5 deg discarded. A 16:9 mode of the same
# height recovers it: 2*atan(424/616) = 69.0 deg, which is the datasheet number.
# Because fx is identical, nothing shrinks — a tag stays the same pixel size and
# there is simply more room either side of it. Same for the D455 (79.3 -> ~90 deg).
# Costs 32% more pixels through the view encoder and the raw depth writer; see
# SHM_RAW_LIMIT for what that does to the in-RAM clip length.
FALLBACK_ATTEMPTS = ((848, 480, 30), (640, 480, 30), (640, 480, 15), (424, 240, 15))
# USB 2 cannot carry 848x480@15 in BGR8 plus depth (~30 MB/s of a ~40 MB/s bus),
# so the wide option there is 640x360 — still 16:9, just fewer pixels.
PROFILE_ATTEMPTS_USB2 = ((640, 360, 15), (640, 480, 15), (424, 240, 15))
_MODEL_DEFAULTS = {"d455": "848x480@30", "d435": "848x480@30"}


def model_profile(model):
    """(w, h, fps) for a camera model like 'd455', from env or defaults."""
    spec = os.environ.get(f"SMARTROOM_DEPTH_PROFILE_{model.upper()}",
                          _MODEL_DEFAULTS.get(model, "640x480@30"))
    try:
        size, fps = spec.lower().split("@")
        width, height = size.split("x")
        return int(width), int(height), int(fps)
    except ValueError:
        return 640, 480, 30


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
# raw depth buffers smaller than this go to /dev/shm (RAM); bigger ones go to
# the SD card next to the recording. /dev/shm is 1.9GB total and BOTH cameras
# buffer there at once, so the per-camera cap is just under half.
# At the 848x480@30 default this covers clips up to ~32s (it was ~42s at
# 640x480); past that, raw depth streams to the SD card during capture.
SHM_RAW_LIMIT = 850_000_000


def _load_depth_extrinsics(serial):
    """This camera's room-frame pose from calibration/<serial>.extrinsics.json,
    trimmed the same way capture.py embeds the webcam's. None if uncalibrated."""
    try:
        ext = json.loads((PROJECT_ROOT / "calibration" / f"{serial}.extrinsics.json").read_text())
    except (OSError, ValueError):
        return None
    keys = ("camera_id", "frame", "tag", "rvec", "tvec_mm", "rotation_cam_to_room",
            "camera_position_mm", "reprojection_error_px", "depth_agreement_mm",
            "levelled", "anchored_by_tag", "calibrated_at")
    return {k: ext[k] for k in keys if k in ext}


# ===========================================================================
# WORKER SIDE — everything below runs inside the per-camera subprocess.
# ===========================================================================

class CameraWorker:
    """One camera: pipeline (on demand), capture loop with a keep() queue so
    slowness becomes latency instead of frame loss, viewer JPEG encoding into
    the parent's queue, recording (raw depth + hardware color, post-encode),
    extrinsic calibration, and point depth queries."""

    def __init__(self, serial, name, usb, view_queue):
        self.serial = serial
        self.name = name
        self.usb = usb
        self.flip = serial in FLIP_SERIALS
        self.view_queue = view_queue
        self.cond = threading.Condition()
        self.depth_z16 = None        # raw uint16 depth, aligned to color
        self.color_bgr = None
        self.color_intr = None       # factory color intrinsics (rs.intrinsics)
        self.depth_scale = None
        self.hw_domain = None        # timestamp domain of hw_ts (want global_time)
        self.last_hw_ts = 0.0        # ms, latest frame's hardware timestamp
        self.frame_id = 0
        self.clients = 0
        self.viewers = 0
        self.running = False
        self.starting = False
        self.last_active = 0.0
        self.error = RS_IMPORT_ERROR
        self.info = {"name": name, "serial": serial, "usb": usb}
        self.record_queues = []      # per-recording frame queues
        self.stats = {"recv": 0, "proc": 0, "sensor_gaps": 0}
        self._last_fn = None
        # recording state (one recording at a time per camera)
        self.rec_lock = threading.Lock()
        self.rec = {"running": False, "encoding": False, "streams": {}, "errors": {}}
        # extrinsic calibration state
        self.cal_lock = threading.Lock()
        self.cal = {"running": False, "ok": None, "message": ""}
        # Set while calibrating: the pipeline is then running a high-resolution
        # COLOUR profile instead of the recording one (see _swap_profile).
        self.calib_profile = False
        self.restart = False         # asks the capture loop to tear down and let
        #                              _maybe_start_locked bring the pipeline back

    # ---------------------------------------------------------- clients ----
    def add_client(self):
        with self.cond:
            self.clients += 1
            self.last_active = time.monotonic()
            self._maybe_start_locked()

    def remove_client(self):
        with self.cond:
            self.clients = max(0, self.clients - 1)
            self.last_active = time.monotonic()

    def set_viewers(self, n):
        with self.cond:
            delta = n - self.viewers
            self.viewers = n
            self.clients = max(0, self.clients + delta)
            self.last_active = time.monotonic()
            if n > 0:
                self._maybe_start_locked()
            self.cond.notify_all()

    def _maybe_start_locked(self):
        # `restart` means a swap is mid-flight: starting here would race the
        # teardown and come up on the profile the swap is trying to leave.
        if not self.running and not self.starting and not self.restart and rs is not None:
            self.starting = True
            threading.Thread(target=self._run, daemon=True).start()

    # --------------------------------------------------------- pipeline ----
    def _start_pipeline(self):
        model = self.name.split()[-1].lower()
        if self.calib_profile:
            # Extrinsics are solved at the best colour mode the camera has, not
            # at the recording profile — the tag's pixel size is what bounds the
            # pose accuracy. See CALIB_COLOR_ATTEMPTS.
            attempts = realsense_extrinsics.CALIB_COLOR_ATTEMPTS
        elif str(self.usb).startswith("2"):
            attempts = PROFILE_ATTEMPTS_USB2
        else:
            preferred = model_profile(model)
            attempts = (preferred,) + tuple(a for a in FALLBACK_ATTEMPTS if a != preferred)
        # Calibration enables depth SEPARATELY from colour and takes the best of
        # each: no D4xx does depth above 1280x720, so it cannot simply match a
        # 1920x1080 colour stream. Live/recording keeps them equal so the aligned
        # panes map 1:1.
        #
        # DEPTH IS THE OUTER LOOP. Nested the other way, a colour mode the sensor
        # does not have burns one failed pipeline.start() per depth variant before
        # moving on — the D455's 1MP sensor cannot do 1920x1080, so it spent three
        # failed starts (seconds each) getting to 1280x800 and blew the
        # first-frame timeout, which aborted the calibration. Sweeping all colour
        # modes at 640x480 depth first costs at most ONE wasted start, because
        # every D4xx has 640x480 depth. Preferring lower depth over higher colour
        # is also the right trade here: colour resolution is what sets the tag's
        # pixel size, and depth is deliberately kept low anyway.
        depth_sizes = (realsense_extrinsics.CALIB_DEPTH_ATTEMPTS if self.calib_profile
                       else (None,))
        pipeline = rs.pipeline()
        last_error = None
        for depth_size in depth_sizes:
            for width, height, fps in attempts:
                dw, dh = depth_size if depth_size else (width, height)
                config = rs.config()
                config.enable_device(self.serial)
                config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                try:
                    profile = pipeline.start(config)
                    return pipeline, profile, (width, height, fps)
                except RuntimeError as exc:
                    last_error = exc
        raise last_error if last_error else RuntimeError("no usable RealSense profile")

    def _run(self):
        try:
            pipeline, profile, (width, height, fps) = self._start_pipeline()
        except Exception as exc:  # noqa: BLE001 - surfaced on the page
            with self.cond:
                self.error = f"Could not start RealSense pipeline: {exc}"
                self.starting = False
                self.cond.notify_all()
            return

        device = profile.get_device()
        depth_scale = device.first_depth_sensor().get_depth_scale()
        for sensor in device.query_sensors():
            # Auto-exposure PRIORITY lets the color sensor stretch exposure
            # beyond the frame interval in dim light, halving delivered fps —
            # disable it. global_time maps sensor timestamps into the host
            # clock (ms since epoch), which is what makes frames from
            # different cameras directly comparable.
            for option, value in ((getattr(rs.option, "auto_exposure_priority", None), 0),
                                  (getattr(rs.option, "global_time_enabled", None), 1)):
                try:
                    if option is not None and sensor.supports(option):
                        sensor.set_option(option, value)
                except RuntimeError:
                    pass

        info = dict(self.info)
        for label, key in (("name", rs.camera_info.name),
                           ("firmware", rs.camera_info.firmware_version),
                           ("usb", rs.camera_info.usb_type_descriptor)):
            try:
                info[label] = device.get_info(key)
            except RuntimeError:
                pass
        info["profile"] = f"{width}x{height}@{fps}"

        align = rs.align(rs.stream.color)
        try:
            color_intr = (profile.get_stream(rs.stream.color)
                          .as_video_stream_profile().get_intrinsics())
        except RuntimeError:
            color_intr = None
        if color_intr is not None:
            # Report the negotiated field of view. Which modes are crops of the
            # sensor and which use its full width is not documented per-profile
            # and is invisible in the resolution alone, so surface it rather than
            # inferring it: a mode that widens the view raises hfov at unchanged
            # fx, a mode that merely upscales raises fx at unchanged hfov.
            info["fx"] = round(float(color_intr.fx), 1)
            info["hfov_deg"] = round(math.degrees(
                2.0 * math.atan(color_intr.width / 2.0 / color_intr.fx)), 1)
            info["vfov_deg"] = round(math.degrees(
                2.0 * math.atan(color_intr.height / 2.0 / color_intr.fy)), 1)

        with self.cond:
            self.running = True
            self.starting = False
            self.error = None
            self.info = info
            self.usb = info.get("usb", self.usb)
            self.color_intr = color_intr
            self.depth_scale = depth_scale
            self.last_active = time.monotonic()

        threading.Thread(target=self._view_encoder, daemon=True).start()

        # Waiter/processor split: the waiter only receives framesets and queues
        # them (keep() detaches them from the SDK's recycle pool), so a slow
        # moment in processing costs LATENCY, never a dropped frame.
        # ~2s of framesets (~75MB via keep()). Sized for recording COMPLETENESS:
        # the processor thread intermittently stalls for a few hundred ms under
        # recording load, and at maxlen=8 (267ms) those stalls overflowed the
        # deque and silently evicted frames (recordings shipped at ~25fps with
        # 66-133ms holes). A deep buffer rides the stalls out; the live view
        # just runs a moment behind until the backlog drains.
        fs_queue = collections.deque(maxlen=60)
        fs_state = {"stop": False, "error": None}
        fs_cond = threading.Condition()

        def waiter():
            try:
                while True:
                    with fs_cond:
                        if fs_state["stop"]:
                            return
                    frameset = pipeline.wait_for_frames(5000)
                    frameset.keep()
                    self.stats["recv"] += 1
                    with fs_cond:
                        fs_queue.append(frameset)
                        fs_cond.notify()
            except Exception as exc:  # noqa: BLE001 - USB drop etc.
                with fs_cond:
                    fs_state["error"] = str(exc)
                    fs_state["stop"] = True
                    fs_cond.notify()

        waiter_thread = threading.Thread(target=waiter, daemon=True)
        waiter_thread.start()
        try:
            while True:
                with self.cond:
                    if self.restart:   # calibration is swapping the colour profile
                        break
                    if self.clients == 0 and (time.monotonic() - self.last_active) > IDLE_TIMEOUT:
                        break
                with fs_cond:
                    if not fs_queue and not fs_state["stop"]:
                        fs_cond.wait(timeout=0.5)
                    if fs_state["stop"]:
                        raise RuntimeError(fs_state["error"] or "capture stopped")
                    frames = fs_queue.popleft() if fs_queue else None
                if frames is None:
                    continue
                try:
                    hw_ts = float(frames.get_timestamp())  # ms (global time domain)
                    if self.hw_domain is None:
                        self.hw_domain = str(frames.get_frame_timestamp_domain())
                except RuntimeError:
                    hw_ts = 0.0
                frames = align.process(frames)
                depth = frames.get_depth_frame()
                color = frames.get_color_frame()
                if not depth or not color:
                    continue
                self.stats["proc"] += 1
                number = color.get_frame_number()
                if self._last_fn is not None and number > self._last_fn + 1:
                    self.stats["sensor_gaps"] += number - self._last_fn - 1
                self._last_fn = number
                color_img = np.asanyarray(color.get_data())
                depth_raw = np.asanyarray(depth.get_data())
                # NOTE: frames are stored UNROTATED even for flipped cameras —
                # rotating here costs the 30fps budget. The flip happens at the
                # consumers: view encoder (12fps), ffmpeg (recordings),
                # depth_at (coordinate transform), calibration (6 frames).
                stamp = time.monotonic()
                with self.cond:
                    # copies: the arrays are views over the SDK's recycled frame buffers
                    self.depth_z16 = depth_raw.copy()
                    self.color_bgr = color_img.copy()
                    self.last_hw_ts = hw_ts
                    self.frame_id += 1
                    for rq in self.record_queues:
                        rq.append((self.color_bgr, self.depth_z16, stamp, hw_ts))
                    self.cond.notify_all()
        except Exception as exc:  # noqa: BLE001 - USB drop etc.
            with self.cond:
                self.error = f"RealSense stream stopped: {exc}"
        finally:
            with fs_cond:
                fs_state["stop"] = True
            try:
                pipeline.stop()
            except Exception:
                pass
            waiter_thread.join(timeout=6)
            with self.cond:
                self.running = False
                self.cond.notify_all()

    # ------------------------------------------------------- view frames ---
    def _view_encoder(self):
        """Colorize + JPEG-encode into the parent's view queue, at most
        VIEW_ENCODE_FPS, only while the parent reports viewers. Exits when the
        pipeline stops."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        interval = 1.0 / VIEW_ENCODE_FPS if VIEW_ENCODE_FPS > 0 else 0.0
        last_id = -1
        while True:
            with self.cond:
                while (self.frame_id == last_id or self.viewers == 0) and self.running:
                    self.cond.wait(timeout=1.0)
                if not (self.running or self.starting):
                    return
                last_id = self.frame_id
                bgr, z16, scale = self.color_bgr, self.depth_z16, self.depth_scale
                hw_ts = self.last_hw_ts   # this frame's sensor timestamp (global clock)
            if bgr is None or z16 is None:
                continue
            t_enc = time.monotonic()
            if self.flip:  # flipped camera: rotate at view rate, not capture rate
                bgr = cv2.rotate(bgr, cv2.ROTATE_180)
                z16 = cv2.rotate(z16, cv2.ROTATE_180)
            # colorized depth from the raw z16, fixed 0-6m range. All-OpenCV
            # ops (they release the GIL); near=red, far=blue.
            d8 = cv2.convertScaleAbs(z16, alpha=255.0 * (scale or 0.001) / 6.0)
            vis = cv2.applyColorMap(cv2.subtract(255, d8), cv2.COLORMAP_JET)
            vis[z16 == 0] = 0
            ok_rgb, rgb_buf = cv2.imencode(".jpg", bgr, encode_params)
            ok_depth, depth_buf = cv2.imencode(".jpg", vis, encode_params)
            if ok_rgb and ok_depth:
                try:
                    self.view_queue.put_nowait((rgb_buf.tobytes(), depth_buf.tobytes(),
                                                hw_ts))
                except Exception:
                    pass  # parent slow — drop, the next frame supersedes
            # rate-limit: sleep only the time LEFT in the frame budget, not a full
            # interval on top of the ~work already spent (that capped us well below
            # VIEW_ENCODE_FPS — e.g. 30fps target + ~19ms work gave only ~19fps).
            time.sleep(max(0.0, interval - (time.monotonic() - t_enc)))

    # ----------------------------------------------------------- queries ---
    def wait_first_frame(self, timeout):
        end = time.monotonic() + timeout
        with self.cond:
            while self.frame_id == 0:
                if not (self.running or self.starting):
                    return False
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return False
                self.cond.wait(timeout=remaining)
            return True

    def depth_at(self, x_frac, y_frac):
        if self.flip:  # viewers see the rotated image — flip the query back
            x_frac, y_frac = 1.0 - x_frac, 1.0 - y_frac
        with self.cond:
            depth = self.depth_z16
            scale = self.depth_scale
        if depth is None or scale is None:
            return None, None, None
        h, w = depth.shape
        px = min(w - 1, max(0, int(x_frac * w)))
        py = min(h - 1, max(0, int(y_frac * h)))
        window = depth[max(0, py - 2):py + 3, max(0, px - 2):px + 3]
        valid = window[window > 0]
        meters = float(np.median(valid)) * scale if valid.size else None
        return meters, px, py

    def status(self):
        with self.cond:
            return {
                "running": self.running,
                "starting": self.starting,
                "error": self.error,
                "info": dict(self.info),
                "stats": dict(self.stats),
                # hw-clock health: latest frame's global-time stamp + the host
                # clock at reply time. host_ms - last_hw_ts should be small
                # (exposure + USB transfer) and STABLE; a wandering delta means
                # the sensor→host global-time mapping is off (frames will land
                # at the wrong place on the sync timeline).
                "last_hw_ts": self.last_hw_ts,
                "host_ms": time.time() * 1000.0,
            }

    # --------------------------------------------------------- recording ---
    def record_start(self, out_dir, duration):
        with self.cal_lock:
            # Calibration has the pipeline on a high-resolution, low-fps colour
            # profile — a recording started now would be captured at that
            # profile instead of the one node.env pins.
            if self.cal["running"]:
                return False, "extrinsic calibration is running — retry when it finishes", None
        with self.rec_lock:
            if self.rec["running"]:
                return False, "recording already in progress", None
            self.rec = {"running": True, "encoding": False, "streams": {}, "errors": {}}
        model = self.name.split()[-1].lower()
        threading.Thread(target=self._record, args=(Path(out_dir), duration, model),
                         daemon=True).start()
        return True, "recording", model

    def record_status(self):
        with self.rec_lock:
            return json.loads(json.dumps(self.rec))  # deep copy, pickle-safe

    def _rec_finish(self, error=None, key=None):
        with self.rec_lock:
            if error is not None and key is not None:
                self.rec["errors"][key] = error
            self.rec["running"] = False
            self.rec["encoding"] = False

    def _record(self, out_dir, duration, model):
        key = f"camera_{model}"
        self.add_client()  # hold the pipeline open for the whole recording
        color_proc = None
        raw_path = None
        try:
            if not self.wait_first_frame(FIRST_FRAME_TIMEOUT):
                raise RuntimeError(self.status().get("error") or "camera unavailable")
            with self.cond:
                height, width = self.depth_z16.shape
                intr = self.color_intr
                depth_scale = self.depth_scale
                info = dict(self.info)
                hw_domain = self.hw_domain
            try:
                fps_nominal = float(info.get("profile", "@30").rsplit("@", 1)[1])
            except (IndexError, ValueError):
                fps_nominal = 30.0

            # Depth goes to a RAW buffer during capture (plain writes always
            # keep up; live lossless encoding could not) — RAM when it fits,
            # else the SD card — and is FFV1-encoded after the recording ends.
            estimated = int(width * height * 2 * fps_nominal * duration * 1.1)
            shm = Path("/dev/shm")
            raw_dir = shm if shm.is_dir() and estimated < SHM_RAW_LIMIT else out_dir
            raw_path = raw_dir / f".{key}_depth_{os.getpid()}.raw"
            raw_file = raw_path.open("wb", buffering=1 << 20)

            # flipped cameras: frames arrive unrotated (capture-loop budget) —
            # ffmpeg applies the 180° rotation, off the worker's GIL
            flip_args = ["-vf", "hflip,vflip"] if self.flip else []
            color_path = out_dir / f"{key}_color.mp4"
            color_proc = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                 "-r", str(fps_nominal), "-i", "pipe:0", *flip_args,
                 # Pi 4 hardware encoder — near-zero CPU. ALL-INTRA (-g 1):
                 # the dashboard's synced player steps frames by seeking, and
                 # long GOPs make seek decode cost large and asymmetric across
                 # cameras (visible as one camera lagging). Higher bitrate
                 # compensates for intra-only coding.
                 "-c:v", "h264_v4l2m2m", "-g", "1", "-b:v", "4M", "-pix_fmt", "yuv420p",
                 str(color_path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:  # 1MB pipe so a whole frame never blocks the sampling loop
                fcntl.fcntl(color_proc.stdin.fileno(), 1031, 1 << 20)
            except OSError:
                pass

            # Frames arrive via a queue the capture thread fills (bounded, so
            # a stalled recorder degrades instead of leaking memory).
            frame_queue = collections.deque(maxlen=90)
            start_time = dt.datetime.now().astimezone()
            mono0 = time.monotonic()
            deadline = mono0 + duration
            with self.cond:
                self.record_queues.append(frame_queue)
            times, hw_times = [], []
            try:
                while True:
                    with self.cond:
                        if not frame_queue:
                            if time.monotonic() >= deadline:
                                break
                            self.cond.wait(timeout=0.25)
                        item = frame_queue.popleft() if frame_queue else None
                    if item is None:
                        continue
                    bgr, z16, stamp, hw_ts = item
                    if stamp >= deadline:
                        break
                    # numpy arrays go straight to the pipes (buffer protocol)
                    color_proc.stdin.write(bgr)
                    raw_file.write(z16)
                    times.append(stamp - mono0)
                    hw_times.append(hw_ts)
            finally:
                with self.cond:
                    if frame_queue in self.record_queues:
                        self.record_queues.remove(frame_queue)

            raw_file.close()
            color_proc.stdin.close()
            color_proc.wait(timeout=60)
            if not times:
                raise RuntimeError("no frames captured")

            # Time the containers to the MEASURED rate so playback duration
            # matches wall clock. Hardware timestamps are the most accurate
            # measure when available (exact per-frame times live in the CSVs).
            if len(hw_times) > 1 and hw_times[-1] > hw_times[0] > 0:
                fps_actual = (len(hw_times) - 1) * 1000.0 / (hw_times[-1] - hw_times[0])
            elif len(times) > 1:
                fps_actual = (len(times) - 1) / (times[-1] - times[0])
            else:
                fps_actual = fps_nominal

            depth_path = out_dir / f"{key}_depth.mkv"
            with self.rec_lock:
                self.rec["encoding"] = True
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y",
                 "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{width}x{height}",
                 "-r", f"{fps_actual:.4f}", "-i", str(raw_path), *flip_args,
                 # golomb-rice coder + slices: ~2x faster than the default
                 # range coder on the Pi, still lossless
                 "-c:v", "ffv1", "-level", "3", "-coder", "0", "-context", "0",
                 "-slices", "4", "-threads", "4", str(depth_path)],
                check=True, timeout=duration * 10 + 300,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if abs(fps_actual - fps_nominal) / fps_nominal > 0.02:
                # retime the color container to the measured rate (remux, no
                # re-encode): itsscale multiplies the CFR input timestamps
                fixed = out_dir / f".{key}_color_retimed.mp4"
                subprocess.run(
                    ["ffmpeg", "-loglevel", "error", "-y",
                     "-itsscale", f"{fps_nominal / fps_actual:.6f}",
                     "-i", str(color_path), "-c", "copy", str(fixed)],
                    check=True, timeout=300,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                fixed.replace(color_path)
            raw_path.unlink(missing_ok=True)

            # hw_timestamp_ms: librealsense global time (host clock, ms since
            # epoch) — THE column to match frames across cameras with.
            for suffix in ("color", "depth"):
                with (out_dir / f"{key}_{suffix}_timestamps.csv").open("w", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["frame_index", "timestamp_seconds", "hw_timestamp_ms"])
                    for i, (t, hw) in enumerate(zip(times, hw_times)):
                        writer.writerow([i, f"{t:.6f}", f"{hw:.3f}"])

            calibration = None
            # the flipped camera's frames are stored rotated — so are these
            frame_intr = self.frame_intrinsics(intr)
            if frame_intr is not None:
                calibration = {"fx": frame_intr.fx, "fy": frame_intr.fy,
                               "ppx": frame_intr.ppx, "ppy": frame_intr.ppy,
                               "width": frame_intr.width, "height": frame_intr.height,
                               "model": getattr(frame_intr.model, "name", str(frame_intr.model)),
                               "coeffs": list(frame_intr.coeffs),
                               "source": "realsense_factory",
                               **({"rotated_180": True} if self.flip else {})}
            extrinsics = _load_depth_extrinsics(self.serial)
            # measured inter-camera clock offset (calibration/camera_timing.json):
            # subtract from THIS stream's hw timestamps to align with the reference
            timing = None
            try:
                timing = json.loads((PROJECT_ROOT / "calibration" / "camera_timing.json").read_text())
            except (OSError, ValueError):
                pass
            hw_offset = float((timing or {}).get("offsets_ms", {}).get(self.serial, 0.0))
            # Drop accounting: at the nominal rate the hw-timestamp span should
            # hold span*fps frames — the shortfall is frames the camera captured
            # but the pipeline lost (each hole = the player holding a stale
            # frame, i.e. visible desync). gap_count = number of holes.
            frames_dropped = gap_count = 0
            if len(hw_times) > 1 and hw_times[-1] > hw_times[0] > 0:
                span_s = (hw_times[-1] - hw_times[0]) / 1000.0
                frames_dropped = max(0, round(span_s * fps_nominal) + 1 - len(hw_times))
                gap_ms = 1.5 * 1000.0 / fps_nominal
                gap_count = sum(1 for a, b in zip(hw_times, hw_times[1:]) if b - a > gap_ms)
            common = {"device": f"realsense:{self.serial}", "camera": info.get("name"),
                      "resolution": [width, height],
                      "fps": round(fps_actual, 2), "nominal_fps": fps_nominal,
                      "frames_dropped": frames_dropped, "gap_count": gap_count,
                      "frame_count": len(times), "start_time": start_time.isoformat(),
                      "hw_timestamp_domain": hw_domain,
                      "hw_clock_offset_ms": hw_offset,
                      "sync": "match frames across cameras on hw_timestamp_ms - hw_clock_offset_ms"}
            color_entry = {"modality": "video", "path": f"streams/{color_path.name}",
                           "codec": "h264",
                           "timestamps_path": f"streams/{key}_color_timestamps.csv", **common}
            depth_entry = {"modality": "depth", "path": f"streams/{depth_path.name}",
                           "codec": "ffv1/gray16le (lossless z16)",
                           "depth_scale_m": depth_scale, "aligned_to": f"{key}_color",
                           "timestamps_path": f"streams/{key}_depth_timestamps.csv", **common}
            if calibration:
                color_entry["calibration"] = calibration
                depth_entry["calibration"] = calibration
            if extrinsics:
                color_entry["extrinsics"] = extrinsics
                depth_entry["extrinsics"] = extrinsics
            with self.rec_lock:
                self.rec["streams"][f"{key}_color"] = color_entry
                self.rec["streams"][f"{key}_depth"] = depth_entry
            self._rec_finish()
        except Exception as exc:  # noqa: BLE001 - reported via record_status
            if color_proc is not None and color_proc.poll() is None:
                try:
                    color_proc.kill()
                except Exception:
                    pass
            if raw_path is not None:
                Path(raw_path).unlink(missing_ok=True)
            self._rec_finish(error=str(exc), key=key)
        finally:
            self.remove_client()

    # ---------------------------------------------------- timing samples ---
    def motion_series(self, duration):
        """(hw_timestamp_ms, motion_energy) per frame for `duration` seconds —
        the raw material for cross-camera timing calibration. Motion energy is
        the mean abs difference between consecutive downscaled gray frames."""
        self.add_client()
        try:
            if not self.wait_first_frame(FIRST_FRAME_TIMEOUT):
                raise RuntimeError(self.status().get("error") or "camera unavailable")
            series, prev, last_id = [], None, -1
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                with self.cond:
                    if self.frame_id == last_id:
                        self.cond.wait(timeout=0.25)
                        if self.frame_id == last_id:
                            continue
                    last_id = self.frame_id
                    bgr, hw = self.color_bgr, self.last_hw_ts
                if bgr is None or not hw:
                    continue
                gray = cv2.cvtColor(cv2.resize(bgr, (160, 120)), cv2.COLOR_BGR2GRAY)
                if prev is not None:
                    series.append((hw, float(cv2.absdiff(gray, prev).mean())))
                prev = gray
            return series
        finally:
            self.remove_client()

    # ------------------------------------------------------- calibration ---
    def frame_intrinsics(self, intr):
        """Intrinsics matching the frames we STORE, not the raw sensor.

        A flipped camera's recorded video and calibration frames are rotated
        180 degrees, so the sensor's principal point is on the wrong side of
        centre for them — off by twice its offset (~1 degree of pose on the
        D435). Everything that projects those pixels must use these."""
        if intr is None or not self.flip:
            return intr
        return realsense_extrinsics.rotate180_intrinsics(intr)

    def cal_start(self):
        with self.cal_lock:
            if self.cal["running"]:
                return False, "Calibration already running for this camera."
            self.cal = {"running": True, "ok": None, "message": "Capturing frames…"}
        threading.Thread(target=self._calibrate, daemon=True).start()
        return True, "Calibration started."

    def cal_status(self):
        with self.cal_lock:
            return dict(self.cal)

    def cal_summary(self):
        """The last calibration's numbers, structured, for the UI to render.

        Read back from the file the solve just wrote rather than parsed out of the
        status message: the message is prose meant for a human, and scraping it in
        JavaScript would break every time its wording changed."""
        try:
            e = json.loads((PROJECT_ROOT / "calibration" /
                            f"{self.serial}.extrinsics.json").read_text())
        except (OSError, ValueError):
            return None
        lev = e.get("levelled") or {}
        try:
            room = json.loads((PROJECT_ROOT / "calibration" / "room_level.json").read_text())
        except (OSError, ValueError):
            room = {}
        px, sizes = e.get("tag_pixels_by_id") or {}, e.get("tag_sizes_mm_by_id") or {}
        used = [str(t) for t in (e.get("solved_from_tags") or [])]
        rng = e.get("tvec_mm") and float(np.linalg.norm(e["tvec_mm"]))
        agree = e.get("depth_agreement_mm")
        warnings = []
        if agree is not None and rng and agree > 0.04 * rng:
            warnings.append(f"depth and the pose disagree on the tag's range by "
                            f"{100 * agree / rng:.1f}% ({agree:.0f} mm of {rng:.0f} mm)")
        if lev.get("camera_tilt_corrected_deg") is None:
            warnings.append("levelling was SKIPPED — too little horizontal surface in view, "
                            "so pitch and roll came from the tag's ill-conditioned solve")
        m_floor, c_floor = room.get("measured_floor_mm"), room.get("configured_floor_mm")
        if m_floor and c_floor and abs(m_floor - c_floor) > 100:
            warnings.append(f"floor measures {m_floor:.0f} mm below the tag but node.env says "
                            f"{c_floor:.0f} mm — a {abs(m_floor - c_floor):.0f} mm contradiction")
        # unknown != small: a file written before tag_pixels_by_id existed has no
        # pixel counts, and treating those as 0 warned about tags it knew nothing
        # about
        small = [t for t in used
                 if px.get(t) is not None
                 and float(px[t]) < realsense_extrinsics.MIN_TAG_PIXELS]
        if small:
            warnings.append("solved from tags under "
                            f"{realsense_extrinsics.MIN_TAG_PIXELS:.0f} px: "
                            + ", ".join(f"tag {t} ({px.get(t)} px)" for t in small))
        return {
            "resolution": e.get("image_size"),
            "position_mm": e.get("camera_position_mm"),
            "range_mm": round(rng) if rng else None,
            "reproj_px": e.get("reprojection_error_px"),
            "depth_agreement_mm": agree,
            "depth_agreement_pct": round(100 * agree / rng, 1) if (agree is not None and rng) else None,
            "used": [{"id": t, "px": px.get(t), "size_mm": sizes.get(t)} for t in used],
            "ignored": [{"id": t, "px": v} for t, v in (e.get("tags_ignored_px") or {}).items()],
            "min_tag_pixels": e.get("min_tag_pixels"),
            "anchored_by_tag": e.get("anchored_by_tag"),
            "levelled": {"tilt_corrected_deg": lev.get("camera_tilt_corrected_deg"),
                         "defines_room_vertical": lev.get("defines_room_vertical"),
                         "normals_ref": lev.get("normals_used_ref"),
                         "yaw_source": lev.get("yaw_source")},
            "floor_mm": m_floor, "configured_floor_mm": c_floor,
            "calibrated_at": e.get("calibrated_at"),
            "warnings": warnings,
        }

    def _swap_profile(self, calib):
        """Restart this camera's pipeline in (or out of) calibration mode.

        The camera is single-access, so capturing calibration frames at a higher
        resolution than the live/recording profile means tearing the running
        pipeline down and bringing it back. Blocks until frames flow again."""
        with self.cond:
            self.calib_profile = calib
            self.restart = True
            self.cond.notify_all()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:      # let the capture loop finish
            with self.cond:
                if not (self.running or self.starting):
                    break
            time.sleep(0.1)
        with self.cond:
            self.restart = False
            # Frames and intrinsics belong to the OLD profile — drop them so
            # nothing mixes resolutions, and so wait_first_frame really waits.
            self.color_bgr = self.depth_z16 = self.color_intr = None
            self.frame_id = 0
            self._maybe_start_locked()
        if not self.wait_first_frame(SWAP_FRAME_TIMEOUT):
            raise RuntimeError(self.status().get("error")
                               or f"camera did not restart ({'calibration' if calib else 'live'} profile) "
                                  f"within {SWAP_FRAME_TIMEOUT:.0f}s")

    def _calibrate(self):
        self.add_client()
        # Set BEFORE the swap, not after it returns. _swap_profile assigns
        # calib_profile as its first act, so a failure part-way through still
        # leaves the camera committed to the calibration profile — and with this
        # flag set afterwards, the restore in `finally` was skipped and the D455
        # sat on 1280x800@15 indefinitely, which is also what any recording
        # started next would have been captured at.
        swapped = True
        try:
            if self.record_queues:
                raise RuntimeError("a recording is in progress — calibrate when it finishes")
            if not self.wait_first_frame(FIRST_FRAME_TIMEOUT):
                raise RuntimeError(self.status().get("error") or "camera unavailable")
            # Up to the camera's best colour mode for the six frames we solve
            # from: pose error scales as 1/tag_px, and the recording profile's
            # 640x480 leaves the 138mm tag only ~26px across.
            self._swap_profile(True)
            samples, last_id = [], -1
            intr = None
            deadline = time.monotonic() + 6.0
            while len(samples) < 6 and time.monotonic() < deadline:
                with self.cond:
                    frame_id = self.frame_id
                    color, z16, intr = self.color_bgr, self.depth_z16, self.color_intr
                    scale = self.depth_scale
                if frame_id != last_id and color is not None and z16 is not None:
                    last_id = frame_id
                    if self.flip:  # calibration must match the recorded (flipped) view
                        color = cv2.rotate(color, cv2.ROTATE_180)
                        z16 = cv2.rotate(z16, cv2.ROTATE_180)
                    samples.append((color, z16.astype(np.float32) * (scale or 0.001)))
                time.sleep(0.25)
            if intr is None:
                raise RuntimeError("no color intrinsics from the camera")
            ok, message = realsense_extrinsics.calibrate_from_samples(
                samples, self.frame_intrinsics(intr), self.serial, camera_name=self.name)
            message = f"solved at {intr.width}x{intr.height}: {message}"
        except Exception as exc:  # noqa: BLE001 - reported on the page
            ok, message = False, str(exc)
        finally:
            if swapped:
                # Always go back to the recording profile, even on failure — the
                # live view and any later recording depend on it.
                try:
                    self._swap_profile(False)
                except Exception as exc:  # noqa: BLE001
                    ok, message = False, f"{message}; !! could not restore the live profile: {exc}"
            self.remove_client()
        with self.cal_lock:
            self.cal = {"running": False, "ok": ok, "message": message}


def worker_main(serial, name, usb, conn, view_queue):
    """Entry point of the per-camera subprocess: build the CameraWorker and
    serve commands from the parent until the pipe closes."""
    worker = CameraWorker(serial, name, usb, view_queue)
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            os._exit(0)
        op = msg.get("op")
        try:
            if op == "status":
                reply = worker.status()
            elif op == "viewers":
                worker.set_viewers(int(msg["n"]))
                reply = {"ok": True}
            elif op == "value":
                meters, px, py = worker.depth_at(msg["x"], msg["y"])
                center, _, _ = worker.depth_at(0.5, 0.5)
                reply = {"ok": meters is not None,
                         "m": round(meters, 3) if meters is not None else None,
                         "px": px, "py": py,
                         "center_m": round(center, 3) if center is not None else None}
            elif op == "record_start":
                ok, message, model = worker.record_start(msg["dir"], msg["duration"])
                reply = {"ok": ok, "message": message, "model": model}
            elif op == "record_status":
                reply = worker.record_status()
            elif op == "cal_start":
                ok, message = worker.cal_start()
                reply = {"ok": ok, "message": message}
            elif op == "cal_status":
                reply = worker.cal_status()
                if not reply.get("running") and reply.get("ok"):
                    reply["summary"] = worker.cal_summary()
            elif op == "motion":
                reply = {"ok": True, "series": worker.motion_series(float(msg["duration"]))}
            else:
                reply = {"ok": False, "error": f"unknown op {op}"}
        except Exception as exc:  # noqa: BLE001 - never kill the cmd loop
            reply = {"ok": False, "error": str(exc)}
        # echo the request id so the parent can tell our reply from a stale one
        if isinstance(reply, dict) and msg.get("_rid") is not None:
            reply = {**reply, "_rid": msg["_rid"]}
        try:
            conn.send(reply)
        except (BrokenPipeError, OSError):
            os._exit(0)


# ===========================================================================
# PARENT SIDE — HTTP front end + worker supervision.
# ===========================================================================

class ViewCache:
    """Latest viewer JPEGs from one worker, for the MJPEG generators."""

    def __init__(self):
        self.cond = threading.Condition()
        self.rgb = None
        self.depth = None
        self.hw_ts = 0.0          # sensor timestamp of the frame in rgb/depth
        self.view_id = 0

    def put(self, rgb, depth, hw_ts=0.0):
        with self.cond:
            self.rgb, self.depth, self.hw_ts = rgb, depth, hw_ts
            self.view_id += 1
            self.cond.notify_all()


class Worker:
    """Parent-side handle: subprocess + command pipe + view queue pump."""

    def __init__(self, dev):
        self.dev = dev
        self.lock = threading.Lock()      # serializes command round-trips
        self.seq = 0                      # request id, so a late reply is not mistaken for ours
        self.view = ViewCache()
        self.viewer_lock = threading.Lock()
        self.viewer_count = 0
        self._spawn()

    def _spawn(self):
        ctx = multiprocessing.get_context("spawn")
        self.view_queue = ctx.Queue(maxsize=4)
        self.conn, child_conn = ctx.Pipe()
        self.proc = ctx.Process(
            target=worker_main,
            args=(self.dev["serial"], self.dev["name"], self.dev["usb"],
                  child_conn, self.view_queue),
            daemon=True)
        self.proc.start()
        child_conn.close()
        threading.Thread(target=self._pump, daemon=True).start()

    def respawn(self):
        with self.lock:
            try:
                self.conn.close()
            except OSError:
                pass
            self._spawn()
        with self.viewer_lock:
            n = self.viewer_count
        if n:
            self.call("viewers", n=n)

    def _pump(self):
        queue = self.view_queue
        while True:
            try:
                item = queue.get(timeout=2.0)
            except Exception:
                if queue is not self.view_queue or not self.proc.is_alive():
                    return  # respawned or dead — a new pump owns the new queue
                continue
            rgb, depth, hw_ts = (item if len(item) == 3 else (*item, 0.0))
            self.view.put(rgb, depth, hw_ts)

    def call(self, op, timeout=10.0, **kw):
        """Command round-trip; {'ok': False, 'error': ...} on any failure.

        Replies carry the request id they answer. Draining whatever has already
        arrived is not enough on its own: a call that timed out leaves its reply
        still in flight, and the next call would then receive THAT as its own —
        observed in the wild as a calibration status poll being handed the
        watchdog's device status, which has no 'running' key, so the UI read the
        run as finished and reported nonsense."""
        with self.lock:
            if not self.proc.is_alive():
                return {"ok": False, "error": "worker not running"}
            self.seq += 1
            rid = self.seq
            try:
                while self.conn.poll(0):  # discard replies to earlier calls
                    self.conn.recv()
                self.conn.send({"op": op, "_rid": rid, **kw})
                deadline = time.monotonic() + timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self.conn.poll(remaining):
                        return {"ok": False, "error": f"worker timeout on {op}"}
                    reply = self.conn.recv()
                    # a late reply to a previous call: keep waiting for ours
                    if isinstance(reply, dict) and reply.get("_rid") not in (None, rid):
                        continue
                    return reply
            except (BrokenPipeError, EOFError, OSError) as exc:
                return {"ok": False, "error": str(exc)}

    def add_viewer(self):
        with self.viewer_lock:
            self.viewer_count += 1
            n = self.viewer_count
        self.call("viewers", n=n, timeout=3.0)

    def remove_viewer(self):
        with self.viewer_lock:
            self.viewer_count = max(0, self.viewer_count - 1)
            n = self.viewer_count
        self.call("viewers", n=n, timeout=3.0)


WORKERS = {}  # serial -> Worker, populated at startup


def _startup_enumerate():
    """Enumerate once with a short-lived context, BEFORE any worker exists —
    a context probing the bus while another process holds the devices' USB
    interfaces enumerates empty (learned the hard way)."""
    devices = []
    if rs is None:
        return devices
    try:
        ctx = rs.context()
        for device in ctx.devices:
            entry = {}
            for label, key in (("serial", rs.camera_info.serial_number),
                               ("name", rs.camera_info.name),
                               ("usb", rs.camera_info.usb_type_descriptor)):
                try:
                    entry[label] = device.get_info(key)
                except RuntimeError:
                    entry[label] = "?"
            if entry.get("serial", "?") != "?":
                devices.append(entry)
        del ctx
    except Exception:  # noqa: BLE001
        pass
    import gc
    gc.collect()
    return devices


def _usb_realsense_count():
    n = 0
    for vendor_file in Path("/sys/bus/usb/devices").glob("*/idVendor"):
        try:
            if vendor_file.read_text().strip() == "8086":
                n += 1
        except OSError:
            continue
    return n


def _watchdog():
    """Respawn dead workers; if the set of PLUGGED cameras stops matching the
    workers (hotplug), exit so systemd respawns the page into a clean
    re-enumeration — never mid-recording."""
    misses = 0
    while True:
        time.sleep(5)
        for worker in WORKERS.values():
            if not worker.proc.is_alive():
                print(f"worker {worker.dev['serial']} died — respawning", flush=True)
                worker.respawn()
        if _usb_realsense_count() != len(WORKERS):
            recording = any((w.call("record_status", timeout=2) or {}).get("running")
                            for w in WORKERS.values())
            misses = 0 if recording else misses + 1
            if misses >= 4:
                print("camera set changed; exiting for a clean re-enumeration", flush=True)
                os._exit(1)
        else:
            misses = 0


# ---------------------------------------------------------------------------
# Cross-camera timing calibration: both cameras watch the same motion for a
# few seconds; the lag that maximizes the cross-correlation of their motion-
# energy series IS the constant offset between their hardware clocks.
# Stored in calibration/camera_timing.json; recordings embed each stream's
# hw_clock_offset_ms (subtract it from that stream's hw timestamps to align
# with the reference camera).
# ---------------------------------------------------------------------------
TIMING = {"running": False, "ok": None, "message": ""}
TIMING_LOCK = threading.Lock()
TIMING_PATH = PROJECT_ROOT / "calibration" / "camera_timing.json"


def start_timing_calibration(duration):
    with TIMING_LOCK:
        if TIMING["running"]:
            return False, "Timing calibration already running."
        if len(WORKERS) < 2:
            return False, "need two cameras connected"
        TIMING.update(running=True, ok=None,
                      message="Watching both cameras — toggle the room lights on/off "
                              "3-4 times (a light change hits every pixel of both "
                              "cameras at once, no shared view needed)…")
    threading.Thread(target=_run_timing, args=(duration,), daemon=True).start()
    return True, "Timing calibration started."


def _run_timing(duration):
    try:
        results = {}
        threads = []

        def grab(serial, worker):
            results[serial] = worker.call("motion", duration=duration, timeout=duration + 20)

        for serial, worker in WORKERS.items():
            t = threading.Thread(target=grab, args=(serial, worker), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=duration + 25)

        series = {}
        for serial, reply in results.items():
            if not (isinstance(reply, dict) and reply.get("ok")):
                raise RuntimeError(f"{serial}: {(reply or {}).get('error', 'no data')}")
            if len(reply["series"]) < 30:
                raise RuntimeError(f"{serial}: too few frames")
            series[serial] = reply["series"]

        serials = sorted(series)
        reference = serials[0]
        offsets, corrs = {}, {}
        for other in serials[1:]:
            offset, corr, overlap = _correlate(series[reference], series[other])
            if corr < 0.35:
                raise RuntimeError(
                    f"correlation too weak ({corr:.2f}) — the cameras didn't see a "
                    "common change; toggle the room lights on/off a few times and rerun")
            offsets[other] = round(offset, 1)
            corrs[other] = round(corr, 3)
        offsets[reference] = 0.0

        TIMING_PATH.parent.mkdir(parents=True, exist_ok=True)
        TIMING_PATH.write_text(json.dumps({
            "schema_version": "1",
            "reference": reference,
            "offsets_ms": offsets,   # subtract from that camera's hw timestamps
            "peak_correlation": corrs,
            "duration_s": duration,
            "measured_at": dt.datetime.now().astimezone().isoformat(),
        }, indent=2))
        detail = ", ".join(f"{s}: {offsets[s]:+.1f} ms (r={corrs[s]:.2f})" for s in serials[1:])
        message = f"clock offsets vs {reference}: {detail} — saved camera_timing.json"
        with TIMING_LOCK:
            TIMING.update(running=False, ok=True, message=message)
    except Exception as exc:  # noqa: BLE001 - reported via status
        with TIMING_LOCK:
            TIMING.update(running=False, ok=False, message=str(exc))


def _correlate(series_a, series_b, step_ms=5.0, max_lag_ms=300.0):
    """(offset_ms, peak_correlation, overlap_s): offset > 0 means camera B's
    hardware clock stamps the same physical event LATER than camera A's."""
    ta = np.array([p[0] for p in series_a]); ea = np.array([p[1] for p in series_a])
    tb = np.array([p[0] for p in series_b]); eb = np.array([p[1] for p in series_b])
    t0, t1 = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    if t1 - t0 < 3000:
        raise RuntimeError("cameras overlapped for under 3s — rerun")
    grid = np.arange(t0, t1, step_ms)
    a = np.interp(grid, ta, ea)
    b = np.interp(grid, tb, eb)
    a -= a.mean(); b -= b.mean()
    if a.std() < 1e-6 or b.std() < 1e-6:
        raise RuntimeError("no motion seen — wave a hand visible to both cameras")
    a /= a.std(); b /= b.std()
    n = len(grid)
    lags = list(range(-int(max_lag_ms / step_ms), int(max_lag_ms / step_ms) + 1))
    corr = []
    for lag in lags:  # b shifted right (later) by `lag` samples vs a
        if lag >= 0:
            corr.append(float(np.dot(a[: n - lag], b[lag:]) / (n - lag)) if n - lag > 50 else -2.0)
        else:
            corr.append(float(np.dot(a[-lag:], b[: n + lag]) / (n + lag)) if n + lag > 50 else -2.0)
    i = int(np.argmax(corr))
    best_lag, best_corr = float(lags[i]), corr[i]
    # parabolic sub-sample refinement around the peak — a sharp common edge
    # (lights toggling) supports better than one grid step of precision
    if 0 < i < len(corr) - 1:
        denom = corr[i - 1] - 2 * corr[i] + corr[i + 1]
        if abs(denom) > 1e-9:
            best_lag += max(-1.0, min(1.0, 0.5 * (corr[i - 1] - corr[i + 1]) / denom))
    return best_lag * step_ms, best_corr, (t1 - t0) / 1000.0


def record_start_all(target, duration):
    if not WORKERS:
        return False, "no RealSense cameras connected"
    for worker in WORKERS.values():
        if (worker.call("record_status", timeout=3) or {}).get("running"):
            return False, "depth recording already in progress"
    names = []
    for worker in WORKERS.values():
        reply = worker.call("record_start", dir=str(target), duration=duration, timeout=5.0)
        if reply.get("ok"):
            names.append(reply.get("model") or worker.dev["serial"])
    if not names:
        return False, "no camera could start recording"
    return True, f"recording {len(names)} depth camera(s): {', '.join(names)}"


def record_status_all():
    agg = {"running": False, "encoding": False, "streams": {}, "errors": {}}
    for serial, worker in WORKERS.items():
        st = worker.call("record_status", timeout=3.0)
        if not isinstance(st, dict) or "running" not in st:
            agg["errors"][serial] = (st or {}).get("error", "no status")
            continue
        agg["running"] = agg["running"] or st["running"]
        agg["encoding"] = agg["encoding"] or st.get("encoding", False)
        agg["streams"].update(st.get("streams") or {})
        agg["errors"].update(st.get("errors") or {})
    return agg


def list_devices():
    devices = []
    for serial, worker in WORKERS.items():
        st = worker.call("status", timeout=2.0)
        if not isinstance(st, dict) or "running" not in st:
            st = {"running": False, "starting": False, "error": (st or {}).get("error"),
                  "info": {}}
        info = st.get("info") or {}
        devices.append({
            "name": info.get("name") or worker.dev["name"],
            "serial": serial,
            "usb": info.get("usb") or worker.dev["usb"],
            "status": st,
        })
    devices.sort(key=lambda d: d.get("name", ""), reverse=True)
    return devices


class Handler(BaseHTTPRequestHandler):
    server_version = "SmartroomDepthPage/3.0"

    def log_message(self, fmt, *args):
        return

    def send_bytes(self, body, content_type="text/html; charset=utf-8", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # the main video page (port 8000, different origin) embeds this page's
        # devices/value/calibrate endpoints
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def send_json(self, payload, status=200):
        self.send_bytes(json.dumps(payload).encode("utf-8"),
                        "application/json; charset=utf-8", status)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        serial = (params.get("s") or [""])[0]
        if parsed.path == "/":
            self.send_bytes(PAGE.encode("utf-8"))
            return
        if parsed.path == "/devices":
            self.send_json({"devices": list_devices(), "sdk_error": RS_IMPORT_ERROR})
            return
        if parsed.path in ("/rgb.mjpg", "/depth.mjpg"):
            self.serve_stream("rgb" if parsed.path == "/rgb.mjpg" else "depth", serial)
            return
        if parsed.path == "/value":
            self.serve_value(params, serial)
            return
        if parsed.path == "/calibrate/extrinsic/status":
            worker = WORKERS.get(serial)
            if worker is None:
                self.send_json({"ok": False, "message": "unknown camera"}, 404)
                return
            self.send_json(worker.call("cal_status", timeout=3.0))
            return
        if parsed.path == "/record/status":
            self.send_json(record_status_all())
            return
        if parsed.path == "/calibrate/timing/status":
            with TIMING_LOCK:
                self.send_json(dict(TIMING))
            return
        self.send_bytes(b"Not found", "text/plain; charset=utf-8", 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        serial = (params.get("s") or [""])[0]
        if parsed.path == "/calibrate/extrinsic":
            worker = WORKERS.get(serial)
            if worker is None:
                self.send_json({"ok": False, "message": "unknown camera"}, 404)
                return
            reply = worker.call("cal_start", timeout=5.0)
            ok = bool(reply.get("ok"))
            self.send_json({"ok": ok, "message": reply.get("message") or reply.get("error", "")},
                           200 if ok else 409)
            return
        if parsed.path == "/calibrate/timing":
            try:
                duration = max(5, min(int(float((params.get("duration") or ["12"])[0])), 60))
            except ValueError:
                duration = 12
            ok, message = start_timing_calibration(duration)
            self.send_json({"ok": ok, "message": message}, 200 if ok else 409)
            return
        if parsed.path == "/record/start":
            out_dir = (params.get("dir") or [""])[0]
            try:
                duration = max(1, min(int(float((params.get("duration") or ["30"])[0])), 3600))
            except ValueError:
                duration = 30
            # only ever record into this repo's data/ tree
            target = Path(out_dir).resolve() if out_dir else None
            if target is None or DATA_DIR.resolve() not in target.parents:
                self.send_json({"ok": False, "message": "dir must be under data/"}, 400)
                return
            target.mkdir(parents=True, exist_ok=True)
            ok, message = record_start_all(target, duration)
            self.send_json({"ok": ok, "message": message}, 200 if ok else 409)
            return
        self.send_bytes(b"Not found", "text/plain; charset=utf-8", 404)

    def serve_value(self, params, serial):
        worker = WORKERS.get(serial)
        if worker is None:
            self.send_json({"ok": False, "message": "unknown camera"}, 404)
            return
        try:
            x = min(1.0, max(0.0, float(params.get("x", ["0.5"])[0])))
            y = min(1.0, max(0.0, float(params.get("y", ["0.5"])[0])))
        except ValueError:
            self.send_json({"ok": False, "message": "bad coordinates"}, 400)
            return
        self.send_json(worker.call("value", x=x, y=y, timeout=3.0))

    def serve_stream(self, which, serial):
        if rs is None:
            self.send_bytes(
                f"pyrealsense2 is not installed on this node: {RS_IMPORT_ERROR}\n"
                f"Build it with setup_realsense_pi.sh, then restart this page.".encode("utf-8"),
                "text/plain; charset=utf-8", 503)
            return
        worker = WORKERS.get(serial)
        if worker is None:
            self.send_bytes(b"unknown camera", "text/plain; charset=utf-8", 404)
            return
        # Bound how much video can sit in the kernel's send buffer: a slow
        # wifi client should see lower fps, not seconds-old frames.
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 128 * 1024)
        except OSError:
            pass
        worker.add_viewer()
        try:
            view = worker.view
            first_seen = view.view_id
            end = time.monotonic() + FIRST_FRAME_TIMEOUT
            with view.cond:
                while view.view_id == first_seen or view.rgb is None:
                    remaining = end - time.monotonic()
                    if remaining <= 0 or not worker.proc.is_alive():
                        status = worker.call("status", timeout=2.0)
                        message = (status.get("error") if isinstance(status, dict) else None) \
                            or "RealSense camera unavailable."
                        self.send_bytes(message.encode("utf-8"), "text/plain; charset=utf-8", 503)
                        return
                    view.cond.wait(timeout=min(1.0, remaining))
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}")
            self.end_headers()
            last_sent = -1
            interval = 1.0 / VIEW_FPS if VIEW_FPS > 0 else 0.0
            while True:
                with view.cond:
                    while view.view_id == last_sent:
                        if not worker.proc.is_alive():
                            return
                        view.cond.wait(timeout=2.0)
                    last_sent = view.view_id
                    frame = view.rgb if which == "rgb" else view.depth
                    hw_ts = view.hw_ts
                if frame is None:
                    continue
                t_send = time.monotonic()
                try:
                    self.wfile.write(b"--" + STREAM_BOUNDARY.encode() + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    # sensor timestamp (librealsense global clock, ms since epoch) —
                    # the cross-camera sync key; consumed by live_forward.py.
                    self.wfile.write(b"X-Hw-Timestamp-Ms: " + f"{hw_ts:.3f}".encode() + b"\r\n")
                    self.wfile.write(b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
                time.sleep(max(0.0, interval - (time.monotonic() - t_send)))
        finally:
            worker.remove_viewer()


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RealSense Depth View</title>
  <style>
    :root { color-scheme: light; --bg:#f6f8fb; --panel:#fff; --ink:#18202a;
            --muted:#687384; --line:#d9e0e8; --accent:#1267c3; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Arial, Helvetica, sans-serif; background:var(--bg); color:var(--ink); }
    header { padding:24px clamp(16px,4vw,44px) 14px; border-bottom:1px solid var(--line); background:var(--panel); }
    h1 { margin:0; font-size:clamp(24px,4vw,36px); }
    header p { margin:6px 0 0; color:var(--muted); font-size:15px; }
    .wrap { width:min(1400px, calc(100% - 32px)); margin:0 auto; }
    .device { margin:26px 0 10px; padding:16px; background:var(--panel);
              border:1px solid var(--line); border-radius:12px; }
    .device > h2 { margin:0 0 2px; font-size:20px; }
    .device .meta { color:var(--muted); font-size:14px; margin:0 0 12px; }
    .usb-warn { color:#b23c3c; font-weight:700; }
    .panes { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; }
    .pane h3 { margin:0 0 6px; font-size:15px; color:var(--muted); font-weight:700; }
    .stage { position:relative; background:#10151c; border:1px solid var(--line);
             border-radius:10px; overflow:hidden; cursor:crosshair; }
    .stage img { width:100%; display:block; min-height:120px; }
    .marker { position:absolute; width:14px; height:14px; margin:-7px 0 0 -7px;
              border:2px solid #fff; border-radius:50%; box-shadow:0 0 3px #000;
              pointer-events:none; display:none; }
    .tag { position:absolute; transform:translate(10px,-50%); background:rgba(0,0,0,.75);
           color:#fff; padding:2px 7px; border-radius:5px; font-size:13px; font-weight:700;
           pointer-events:none; white-space:nowrap; display:none; }
    .cross { position:absolute; left:50%; top:50%; width:12px; height:12px; margin:-6px 0 0 -6px;
             pointer-events:none; opacity:.85; }
    .cross:before, .cross:after { content:""; position:absolute; background:#ffd23c; }
    .cross:before { left:5px; top:0; width:2px; height:12px; }
    .cross:after  { left:0; top:5px; width:12px; height:2px; }
    .readout { margin:10px 0 2px; font-size:15px; }
    .readout b { font-size:19px; }
    .note { background:#fdeaea; border:1px solid #e6b5b5; color:#8d2323;
            border-radius:8px; padding:12px 16px; margin:16px 0; }
    .empty { color:var(--muted); padding:30px 0; }
  </style>
</head>
<body>
  <header>
    <h1>RealSense Depth View</h1>
    <p id="summary">Looking for cameras&hellip;</p>
  </header>
  <div class="wrap">
    <div class="note" id="note" style="display:none"></div>
    <div id="devices"></div>
    <p class="empty" id="empty" style="display:none">
      No RealSense camera detected. Check the USB connection, then refresh.</p>
  </div>
  <script>
    (function () {
      var built = {};    // serial -> section element
      var bumped = {};   // serial -> last time we restarted its <img> streams

      function buildSection(dev) {
        var s = dev.serial;
        var section = document.createElement('div');
        section.className = 'device';
        section.innerHTML =
          '<h2></h2><p class="meta"></p>' +
          '<div class="panes">' +
            '<div class="pane"><h3>Color (RGB)</h3>' +
              '<div class="stage" data-role="rgb"><img alt="RGB">' +
              '<span class="cross"></span><span class="marker"></span><span class="tag"></span></div></div>' +
            '<div class="pane"><h3>Depth (aligned to color)</h3>' +
              '<div class="stage" data-role="depth"><img alt="Depth">' +
              '<span class="cross"></span><span class="marker"></span><span class="tag"></span></div></div>' +
          '</div>' +
          '<div class="readout">Center distance: <b>&mdash;</b>' +
          ' <span style="color:var(--muted)">&nbsp;&mdash; click either image to measure that point</span></div>';
        section.querySelector('h2').textContent = dev.name + '  (S/N ' + s + ')';
        var q = encodeURIComponent(s);
        var rgbImg = section.querySelector('[data-role="rgb"] img');
        var depthImg = section.querySelector('[data-role="depth"] img');
        var center = section.querySelector('.readout b');

        // Each camera's RGB + depth are persistent MJPEG connections, and a
        // browser only allows ~6 per host. Two cameras' 4 streams plus the
        // /value + /devices polls exceed that at load, so a camera loaded later
        // (e.g. the D435, lower on the page) gets its stream requests starved and
        // never shows. Stream + poll a camera ONLY while its section is on
        // screen (the sections are ~viewport tall, so usually one at a time),
        // and drop its connections when scrolled away — keeping well under the cap.
        var valueTimer = null;
        section.__live = false;
        section.__start = function () {
          if (section.__live) return;
          section.__live = true;
          rgbImg.src = '/rgb.mjpg?s=' + q + '&t=' + Date.now();
          depthImg.src = '/depth.mjpg?s=' + q + '&t=' + Date.now();
          valueTimer = setInterval(function () {
            fetch('/value?s=' + q + '&x=0.5&y=0.5')
              .then(function (r) { return r.json(); })
              .then(function (j) {
                center.textContent = j.center_m != null ? j.center_m.toFixed(2) + ' m' : 'no depth';
              }).catch(function () {});
          }, 600);
        };
        section.__stop = function () {
          if (!section.__live) return;
          section.__live = false;
          rgbImg.src = '';      // aborts the MJPEG load -> frees the connection
          depthImg.src = '';
          if (valueTimer) { clearInterval(valueTimer); valueTimer = null; }
        };
        var io = new IntersectionObserver(function (entries) {
          entries.forEach(function (e) { e.isIntersecting ? section.__start() : section.__stop(); });
        }, { rootMargin: '150px 0px' });
        io.observe(section);

        section.querySelectorAll('.stage').forEach(function (stage) {
          stage.addEventListener('click', function (e) {
            var rect = stage.getBoundingClientRect();
            var x = (e.clientX - rect.left) / rect.width;
            var y = (e.clientY - rect.top) / rect.height;
            fetch('/value?s=' + q + '&x=' + x.toFixed(4) + '&y=' + y.toFixed(4))
              .then(function (r) { return r.json(); })
              .then(function (j) {
                section.querySelectorAll('.stage').forEach(function (st) {
                  var m = st.querySelector('.marker'), t = st.querySelector('.tag');
                  m.style.left = (x * 100) + '%'; m.style.top = (y * 100) + '%';
                  t.style.left = (x * 100) + '%'; t.style.top = (y * 100) + '%';
                  t.textContent = j.m != null ? j.m.toFixed(2) + ' m' : 'no depth';
                  m.style.display = ''; t.style.display = '';
                });
              }).catch(function () {});
          });
        });
        return section;
      }

      function refresh() {
        fetch('/devices').then(function (r) { return r.json(); }).then(function (j) {
          var devs = j.devices || [];
          var note = document.getElementById('note');
          if (j.sdk_error) {
            note.textContent = 'RealSense SDK not available: ' + j.sdk_error;
            note.style.display = '';
          }
          document.getElementById('empty').style.display =
            (devs.length === 0 && !j.sdk_error) ? '' : 'none';
          document.getElementById('summary').textContent =
            devs.length ? devs.length + ' camera' + (devs.length > 1 ? 's' : '') + ' connected'
                        : 'No cameras connected';

          var container = document.getElementById('devices');
          devs.forEach(function (dev) {
            var s = dev.serial;
            if (!built[s]) {
              built[s] = buildSection(dev);
              container.appendChild(built[s]);
              bumped[s] = Date.now();
            }
            var st = dev.status || {}, info = st.info || {};
            // stream down while the device is present (USB replug, worker
            // restart) — restart the streams after a grace period, but only for
            // an on-screen (live) section so we don't re-open connections for a
            // camera the visibility gate has intentionally detached.
            if (!st.running && !st.starting && built[s].__live &&
                Date.now() - (bumped[s] || 0) > 8000) {
              bumped[s] = Date.now();
              built[s].__stop();
              built[s].__start();
            }
            var meta = built[s].querySelector('.meta');
            var usb = (dev.usb && dev.usb !== '?') ? dev.usb : (info.usb || '?');
            var bits = [];
            if (info.firmware) bits.push('FW ' + info.firmware);
            if (st.running && info.profile) bits.push(info.profile);
            bits.push('USB ' + usb);
            meta.innerHTML = bits.join(' &middot; ') +
              (String(usb).indexOf('3') !== 0
                ? ' <span class="usb-warn">&#9888; on USB ' + usb +
                  ' — use a blue USB 3 port for full resolution</span>' : '') +
              (!st.running && st.error
                ? ' <span class="usb-warn">' + st.error + ' — reconnecting&hellip;</span>' : '');
          });
        }).catch(function () {});
      }
      refresh();
      setInterval(refresh, 3000);
    })();
  </script>
</body>
</html>"""


def main():
    devices = _startup_enumerate()
    if not devices and rs is not None:
        print("no RealSense cameras found at startup (will watch for hotplug)")
    for dev in devices:
        WORKERS[dev["serial"]] = Worker(dev)
        print(f"worker started for {dev['name']} (S/N {dev['serial']}, USB {dev['usb']})")
    threading.Thread(target=_watchdog, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"RealSense depth page running at http://0.0.0.0:{PORT}")
    if RS_IMPORT_ERROR:
        print(f"WARNING: pyrealsense2 not available yet: {RS_IMPORT_ERROR}")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
