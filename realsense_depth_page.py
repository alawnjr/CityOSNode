#!/usr/bin/env python3
"""Live RGB + depth view for Intel RealSense cameras, on port 8001.

Serves http://<node>.local:8001 with one section per connected RealSense
device (e.g. a D455 and a D435 at the same time): the color stream and the
colorized depth stream side by side (depth aligned to color, so the same pixel
in both panes is the same point in the room). Click anywhere on either pane to
read the depth at that pixel in meters; the center-pixel distance is shown
continuously per camera.

Uses the Intel RealSense SDK (pyrealsense2) — on a Pi that's built from source
into the venv by setup_realsense_pi.sh (no aarch64 pip wheel exists). Run with
the venv python:

    ~/CityOS/.venv/bin/python ~/CityOS/realsense_depth_page.py

This is separate from smartroom_video_page.py (port 8000): the RealSense
cameras are their own USB devices, so both pages run at once without fighting
over a camera. Each device's pipeline starts on demand and is released a few
seconds after its last viewer leaves.

Bandwidth note: RealSense streams are uncompressed (~35+ MB/s per camera at
640x480), so two of them cannot share the Pi 4's single USB 2 bus — put them
in the blue USB 3 ports. The compressed MJPG webcam (C920) is fine on USB 2.
"""
import collections
import csv
import datetime as dt
import fcntl
import json
import os
import socket
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import realsense_extrinsics


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
        if key and key not in os.environ:
            os.environ[key] = value


_load_node_env()

PORT = int(os.environ.get("SMARTROOM_DEPTH_PORT", "8001"))
STREAM_BOUNDARY = "frame"
IDLE_TIMEOUT = 5.0   # release a camera this many seconds after its last viewer leaves
FIRST_FRAME_TIMEOUT = 8.0  # pipeline start + first frames can take a few seconds
JPEG_QUALITY = 70
# Browser-bound frame rate cap. The capture runs at the camera rate; viewers are
# sent only the newest frame at most this often, so a slow wifi link shows a
# lower frame rate instead of accumulating seconds of latency.
VIEW_FPS = float(os.environ.get("SMARTROOM_DEPTH_VIEW_FPS", "30"))
# The colorize+JPEG work for the live view is the most expensive part of the
# capture loop — cap how often it runs so 30fps pipelines don't burn the CPU
# the recorder needs. Purely a preview smoothness knob.
VIEW_ENCODE_FPS = float(os.environ.get("SMARTROOM_DEPTH_VIEW_ENCODE_FPS", "12"))
# Comma-separated camera serials whose frames are rotated 180 degrees (for
# upside-down mounted cameras) — set per node in node.env. The raw depth grid
# is rotated too, so click-to-measure coordinates stay correct.
FLIP_SERIALS = {s.strip() for s in os.environ.get("SMARTROOM_DEPTH_FLIP", "").split(",") if s.strip()}

# The SDK import is allowed to fail so the page can still come up and explain
# itself while librealsense is not built yet (or the module is missing).
try:
    import cv2
    import pyrealsense2 as rs
    RS_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on node state
    rs = None
    RS_IMPORT_ERROR = str(exc)

# Depth and color at the same resolution so the aligned panes map 1:1.
# Per-model capture profiles (the recording runs at the pipeline rate): the
# D455 is the primary depth camera (30fps), the D435 the secondary (15fps) —
# 640x480 keeps two-camera capture + post-encode within the Pi 4's budget.
# Override per model with e.g. SMARTROOM_DEPTH_PROFILE_D455=848x480@30 in
# node.env. On USB 2 the low fallbacks apply: the whole USB 2 bus is ~40 MB/s
# and one 640x480@30 depth+color pair nearly fills it.
FALLBACK_ATTEMPTS = ((640, 480, 30), (640, 480, 15), (424, 240, 15))
PROFILE_ATTEMPTS_USB2 = ((640, 480, 15), (424, 240, 15))
_MODEL_DEFAULTS = {"d455": "640x480@30", "d435": "640x480@15"}


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


class RealSenseStream:
    """One shared pyrealsense2 pipeline for one device (by serial),
    broadcasting color + colorized-depth JPEGs to any number of MJPEG viewers,
    plus the latest raw depth (meters) for point queries. Starts on demand,
    stops when idle."""

    def __init__(self, serial):
        self.serial = serial
        self.cond = threading.Condition()
        self.rgb_jpeg = None
        self.depth_jpeg = None
        self.depth_z16 = None        # raw uint16 depth, aligned to color
        self.depth_scale = None      # meters per z16 unit
        self.color_bgr = None        # raw color frame (for extrinsic calibration)
        self.color_intr = None       # factory color intrinsics (rs.intrinsics)
        self.frame_id = 0
        self.view_id = 0     # bumps when fresh viewer JPEGs land (encoder thread)
        self.clients = 0     # anything holding the pipeline open (viewers, recorder, calibration)
        self.viewers = 0     # MJPEG viewers only — gates the colorize/JPEG-encode work
        self.running = False
        self.starting = False
        self.last_active = 0.0
        self.error = RS_IMPORT_ERROR
        self.info = {}               # device name/serial/fw/usb + active profile
        self.record_queues = []      # per-recorder frame queues (see _record_one)

    def add_client(self, viewer=True):
        with self.cond:
            self.clients += 1
            if viewer:
                self.viewers += 1
            self.last_active = time.monotonic()
            if not self.running and not self.starting and rs is not None:
                self.starting = True
                threading.Thread(target=self._run, daemon=True).start()

    def remove_client(self, viewer=True):
        with self.cond:
            self.clients = max(0, self.clients - 1)
            if viewer:
                self.viewers = max(0, self.viewers - 1)
            self.last_active = time.monotonic()

    def _device_info(self, key):
        try:
            for device in _rs_context().devices:
                if device.get_info(rs.camera_info.serial_number) == self.serial:
                    return device.get_info(key)
        except RuntimeError:
            pass
        return "?"

    def _start_pipeline(self):
        # prefer the model's configured profile; low fallbacks cover USB 2 and
        # rejected modes (on known USB 2 skip straight to them)
        if self._device_info(rs.camera_info.usb_type_descriptor).startswith("2"):
            attempts = PROFILE_ATTEMPTS_USB2
        else:
            model = self._device_info(rs.camera_info.name).split()[-1].lower()
            preferred = model_profile(model)
            attempts = (preferred,) + tuple(a for a in FALLBACK_ATTEMPTS if a != preferred)
        pipeline = rs.pipeline()
        last_error = None
        for width, height, fps in attempts:
            config = rs.config()
            config.enable_device(self.serial)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
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
        # Auto-exposure PRIORITY lets the color sensor stretch exposure beyond
        # the frame interval in dim light, halving/thirding the delivered fps
        # (and the whole synced frameset with it). Disable it: constant fps,
        # auto-exposure adapts within the frame budget instead.
        for sensor in device.query_sensors():
            try:
                if sensor.supports(rs.option.auto_exposure_priority):
                    sensor.set_option(rs.option.auto_exposure_priority, 0)
            except RuntimeError:
                pass
        info = {}
        for label, key in (("name", rs.camera_info.name),
                           ("serial", rs.camera_info.serial_number),
                           ("firmware", rs.camera_info.firmware_version),
                           ("usb", rs.camera_info.usb_type_descriptor)):
            try:
                info[label] = device.get_info(key)
            except RuntimeError:
                info[label] = "?"
        info["profile"] = f"{width}x{height}@{fps}"

        align = rs.align(rs.stream.color)
        try:
            color_intr = (profile.get_stream(rs.stream.color)
                          .as_video_stream_profile().get_intrinsics())
        except RuntimeError:
            color_intr = None

        with self.cond:
            self.running = True
            self.starting = False
            self.error = None
            self.info = info
            self.color_intr = color_intr
            self.depth_scale = depth_scale
            self.last_active = time.monotonic()

        threading.Thread(target=self._view_encoder, daemon=True).start()

        # Waiter/processor split: the waiter only receives framesets and queues
        # them (keep() detaches them from the SDK's recycle pool), so a slow
        # moment in processing costs LATENCY, never a dropped frame — the
        # camera's full rate arrives as long as processing keeps up on average.
        fs_queue = collections.deque(maxlen=4)
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
                frames = align.process(frames)
                depth = frames.get_depth_frame()
                color = frames.get_color_frame()
                if not depth or not color:
                    continue
                color_img = np.asanyarray(color.get_data())
                depth_raw = np.asanyarray(depth.get_data())
                if self.serial in FLIP_SERIALS:
                    color_img = cv2.rotate(color_img, cv2.ROTATE_180)
                    depth_raw = cv2.rotate(depth_raw, cv2.ROTATE_180)
                stamp = time.monotonic()
                with self.cond:
                    # copies: the arrays are views over the SDK's recycled frame buffers
                    self.depth_z16 = depth_raw.copy()
                    self.color_bgr = color_img.copy()
                    self.frame_id += 1
                    for rq in self.record_queues:
                        rq.append((self.color_bgr, self.depth_z16, stamp))
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

    def _view_encoder(self):
        """Colorize + JPEG-encode for MJPEG viewers, decoupled from the capture
        loop so a slow encode never costs the recorder a frame. Runs at most
        VIEW_ENCODE_FPS; exits when the pipeline stops."""
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
            if bgr is None or z16 is None:
                continue
            # colorized depth from the raw z16, fixed 0-6m range. All-OpenCV
            # ops (they release the GIL — numpy math here starved the capture
            # threads); near=red, far=blue like the SDK's colorizer.
            max_m = 6.0
            d8 = cv2.convertScaleAbs(z16, alpha=255.0 * (scale or 0.001) / max_m)
            vis = cv2.applyColorMap(cv2.subtract(255, d8), cv2.COLORMAP_JET)
            vis[z16 == 0] = 0
            ok_rgb, rgb_buf = cv2.imencode(".jpg", bgr, encode_params)
            ok_depth, depth_buf = cv2.imencode(".jpg", vis, encode_params)
            if ok_rgb and ok_depth:
                with self.cond:
                    self.rgb_jpeg = rgb_buf.tobytes()
                    self.depth_jpeg = depth_buf.tobytes()
                    self.view_id += 1
                    self.cond.notify_all()
            time.sleep(interval)

    def wait_first_frame(self, timeout, view=False):
        """Wait for the first captured frame (view=True: first viewer JPEG)."""
        end = time.monotonic() + timeout
        with self.cond:
            while (self.view_id if view else self.frame_id) == 0:
                if not (self.running or self.starting):
                    return False
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return False
                self.cond.wait(timeout=remaining)
            return True

    def frames(self, which):
        last_sent = -1
        interval = 1.0 / VIEW_FPS if VIEW_FPS > 0 else 0.0
        while True:
            with self.cond:
                while self.view_id == last_sent and (self.running or self.starting):
                    self.cond.wait(timeout=2.0)
                if self.view_id == last_sent and not (self.running or self.starting):
                    return
                last_sent = self.view_id
                frame = self.rgb_jpeg if which == "rgb" else self.depth_jpeg
            if frame is not None:
                yield frame
                # throttle: on wake the loop above picks the NEWEST frame, so a
                # viewer sees fresh frames at <= VIEW_FPS instead of a backlog.
                time.sleep(interval)

    def depth_at(self, x_frac, y_frac):
        """Depth in meters at fractional image coords (0..1), median of the
        valid readings in a small window (single pixels are often 0/no-data).
        Computed from the raw z16 on demand — converting whole frames to
        meters at 30fps was pure waste."""
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
                "info": self.info,
            }


# One stream per device serial, created on first use.
_STREAMS = {}
_STREAMS_LOCK = threading.Lock()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
# raw depth buffers smaller than this go to /dev/shm (RAM); bigger ones go to
# the SD card next to the recording (the card sustains ~33 MB/s, enough for
# both cameras' raw z16 at the configured rates)
SHM_RAW_LIMIT = 1_200_000_000


def _load_depth_extrinsics(serial):
    """This camera's room-frame pose from calibration/<serial>.extrinsics.json,
    trimmed the same way capture.py embeds the webcam's. None if uncalibrated."""
    try:
        ext = json.loads((PROJECT_ROOT / "calibration" / f"{serial}.extrinsics.json").read_text())
    except (OSError, ValueError):
        return None
    keys = ("camera_id", "frame", "tag", "rvec", "tvec_mm", "rotation_cam_to_room",
            "camera_position_mm", "reprojection_error_px", "depth_agreement_mm",
            "anchored_by_tag", "calibrated_at")
    return {k: ext[k] for k in keys if k in ext}


class DepthRecordJob:
    """Records every connected RealSense camera into a recording's streams/
    folder at the camera's full pipeline rate: color as H.264 mp4 (Pi hardware
    encoder, live) and depth as LOSSLESS 16-bit FFV1 mkv (raw z16 units —
    multiply by depth_scale_m for meters), each with a real per-frame
    timestamps CSV. Depth is captured RAW (plain writes keep up where live
    FFV1 could not) and encoded after the recording ends — /record/status
    stays running during that encode, roughly another 1-2x the clip length.
    Both containers are timed to the MEASURED frame rate, so playback duration
    matches wall-clock. capture.py triggers this via POST /record/start and
    merges the returned stream metadata."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.encoding = 0
        self.pending = 0
        self.streams = {}   # metadata stream entries, filled as cameras finish
        self.errors = {}

    def start(self, out_dir, duration):
        devices = list_devices()
        if not devices:
            time.sleep(1.0)  # enumeration can transiently miss devices — one retry
            devices = list_devices()
        if not devices:
            return False, "no RealSense cameras connected"
        with self.lock:
            if self.running:
                return False, "depth recording already in progress"
            self.running = True
            self.pending = len(devices)
            self.streams, self.errors = {}, {}
        names = []
        for dev in devices:
            model = (dev.get("name") or "realsense").split()[-1].lower()  # 'd455'
            names.append(model)
            threading.Thread(target=self._record_one,
                             args=(dev["serial"], model, Path(out_dir), duration),
                             daemon=True).start()
        return True, f"recording {len(devices)} depth camera(s): {', '.join(names)}"

    def status(self):
        with self.lock:
            return {"running": self.running, "encoding": self.encoding > 0,
                    "streams": self.streams, "errors": self.errors}

    def _finish_one(self, key, error=None):
        with self.lock:
            if error is not None:
                self.errors[key] = error
            self.pending -= 1
            if self.pending <= 0:
                self.running = False

    def _record_one(self, serial, model, out_dir, duration):
        key = f"camera_{model}"
        stream = get_stream(serial)
        stream.add_client(viewer=False)  # hold the pipeline open, no viewer encoding
        color_proc = None
        raw_path = None
        try:
            if not stream.wait_first_frame(FIRST_FRAME_TIMEOUT):
                raise RuntimeError(stream.status().get("error") or "camera unavailable")
            with stream.cond:
                height, width = stream.depth_z16.shape
                intr = stream.color_intr
                depth_scale = stream.depth_scale
                info = dict(stream.info)
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

            color_path = out_dir / f"{key}_color.mp4"
            color_proc = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                 "-r", str(fps_nominal), "-i", "pipe:0",
                 # Pi 4 hardware encoder — near-zero CPU
                 "-c:v", "h264_v4l2m2m", "-b:v", "2M", "-pix_fmt", "yuv420p",
                 str(color_path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:  # 1MB pipe so a whole frame never blocks the sampling loop
                fcntl.fcntl(color_proc.stdin.fileno(), 1031, 1 << 20)
            except OSError:
                pass

            # Frames arrive via a queue the capture thread fills (bounded, so a
            # stalled recorder degrades instead of leaking memory) — every
            # captured frame gets recorded, not just the ones we win a race for.
            frame_queue = collections.deque(maxlen=90)
            start_time = dt.datetime.now().astimezone()
            mono0 = time.monotonic()
            deadline = mono0 + duration
            with stream.cond:
                stream.record_queues.append(frame_queue)
            times = []
            try:
                while True:
                    with stream.cond:
                        if not frame_queue:
                            if time.monotonic() >= deadline:
                                break
                            stream.cond.wait(timeout=0.25)
                        item = frame_queue.popleft() if frame_queue else None
                    if item is None:
                        continue
                    bgr, z16, stamp = item
                    if stamp >= deadline:
                        break
                    color_proc.stdin.write(bgr.tobytes())
                    raw_file.write(z16.tobytes())
                    times.append(stamp - mono0)
            finally:
                with stream.cond:
                    if frame_queue in stream.record_queues:
                        stream.record_queues.remove(frame_queue)

            raw_file.close()
            color_proc.stdin.close()
            color_proc.wait(timeout=60)
            if not times:
                raise RuntimeError("no frames captured")

            # Time the containers to the MEASURED rate so playback duration
            # matches wall clock (exact per-frame times live in the CSVs).
            fps_actual = ((len(times) - 1) / (times[-1] - times[0])
                          if len(times) > 1 else fps_nominal)
            depth_path = out_dir / f"{key}_depth.mkv"
            with self.lock:
                self.encoding += 1
            try:
                subprocess.run(
                    ["ffmpeg", "-loglevel", "error", "-y",
                     "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{width}x{height}",
                     "-r", f"{fps_actual:.4f}", "-i", str(raw_path),
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
            finally:
                with self.lock:
                    self.encoding -= 1
            raw_path.unlink(missing_ok=True)

            for suffix in ("color", "depth"):
                with (out_dir / f"{key}_{suffix}_timestamps.csv").open("w", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["frame_index", "timestamp_seconds"])
                    for i, t in enumerate(times):
                        writer.writerow([i, f"{t:.6f}"])

            calibration = None
            if intr is not None:
                calibration = {"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
                               "width": intr.width, "height": intr.height,
                               "model": getattr(intr.model, "name", str(intr.model)),
                               "coeffs": list(intr.coeffs), "source": "realsense_factory"}
            extrinsics = _load_depth_extrinsics(serial)
            common = {"device": f"realsense:{serial}", "camera": info.get("name"),
                      "resolution": [width, height],
                      "fps": round(fps_actual, 2), "nominal_fps": fps_nominal,
                      "frame_count": len(times), "start_time": start_time.isoformat()}
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
            with self.lock:
                self.streams[f"{key}_color"] = color_entry
                self.streams[f"{key}_depth"] = depth_entry
            self._finish_one(key)
        except Exception as exc:  # noqa: BLE001 - reported via /record/status
            if color_proc is not None and color_proc.poll() is None:
                try:
                    color_proc.kill()
                except Exception:
                    pass
            if raw_path is not None:
                Path(raw_path).unlink(missing_ok=True)
            self._finish_one(key, error=str(exc))
        finally:
            stream.remove_client(viewer=False)


DEPTH_RECORDER = DepthRecordJob()


# Extrinsic calibration jobs, one per camera serial. Runs in-process on live
# frames from the stream (no camera handoff needed — the stream keeps running).
_EXTRINSIC_JOBS = {}
_EXTRINSIC_LOCK = threading.Lock()


def extrinsic_status(serial):
    with _EXTRINSIC_LOCK:
        return dict(_EXTRINSIC_JOBS.get(serial) or {"running": False, "ok": None, "message": ""})


def start_extrinsic_calibration(serial):
    with _EXTRINSIC_LOCK:
        job = _EXTRINSIC_JOBS.get(serial)
        if job and job["running"]:
            return False, "Calibration already running for this camera."
        _EXTRINSIC_JOBS[serial] = {"running": True, "ok": None, "message": "Capturing frames…"}
    threading.Thread(target=_run_extrinsic, args=(serial,), daemon=True).start()
    return True, "Calibration started."


def _run_extrinsic(serial):
    stream = get_stream(serial)
    stream.add_client(viewer=False)  # hold the pipeline open while we sample
    try:
        if not stream.wait_first_frame(FIRST_FRAME_TIMEOUT):
            raise RuntimeError(stream.status().get("error") or "camera unavailable")
        samples, last_id = [], -1
        deadline = time.monotonic() + 6.0
        while len(samples) < 6 and time.monotonic() < deadline:
            with stream.cond:
                frame_id = stream.frame_id
                color, z16, intr = stream.color_bgr, stream.depth_z16, stream.color_intr
                scale = stream.depth_scale
            if frame_id != last_id and color is not None and z16 is not None:
                last_id = frame_id
                samples.append((color, z16.astype(np.float32) * (scale or 0.001)))
            time.sleep(0.25)
        if intr is None:
            raise RuntimeError("no color intrinsics from the camera")
        name = stream.status().get("info", {}).get("name", "RealSense")
        ok, message = realsense_extrinsics.calibrate_from_samples(
            samples, intr, serial, camera_name=name)
    except Exception as exc:  # noqa: BLE001 - reported on the page
        ok, message = False, str(exc)
    finally:
        stream.remove_client(viewer=False)
    with _EXTRINSIC_LOCK:
        _EXTRINSIC_JOBS[serial] = {"running": False, "ok": ok, "message": message}

# One long-lived SDK context, created at startup BEFORE any pipeline exists
# and never rebuilt. With the RSUSB backend a context probes the USB bus when
# created — a context created while our own pipelines hold the devices' USB
# interfaces can't open them and enumerates empty, permanently.
_RS_CTX = {"ctx": None}
_RS_CTX_LOCK = threading.Lock()


def _rs_context():
    with _RS_CTX_LOCK:
        if _RS_CTX["ctx"] is None:
            _RS_CTX["ctx"] = rs.context()
        return _RS_CTX["ctx"]


def get_stream(serial):
    with _STREAMS_LOCK:
        stream = _STREAMS.get(serial)
        if stream is None:
            stream = _STREAMS[serial] = RealSenseStream(serial)
        return stream


_ENUM_FAILS = {"n": 0}


def _usb_realsense_present():
    """True if any Intel USB device is on the bus (sysfs, no SDK involved) —
    distinguishes 'no camera plugged in' from 'SDK enumeration is wedged'."""
    for vendor_file in Path("/sys/bus/usb/devices").glob("*/idVendor"):
        try:
            if vendor_file.read_text().strip() == "8086":
                return True
        except OSError:
            continue
    return False


def _enum_watchdog(found_devices):
    """A replug while a pipeline is open can leave this process holding stale
    USB claims ('failed to set power state') or a cached empty device list,
    poisoning every later enumeration until the process dies. If enumeration
    keeps coming up empty while sysfs says a RealSense is physically present
    and no stream is active, exit so systemd hands us a clean process."""
    if found_devices or not _usb_realsense_present():
        _ENUM_FAILS["n"] = 0
        return
    _ENUM_FAILS["n"] += 1
    with _STREAMS_LOCK:
        active = any(s.running or s.starting for s in _STREAMS.values())
    if not active and _ENUM_FAILS["n"] >= 3:
        print("RealSense enumeration wedged; exiting for a clean restart")
        os._exit(1)


def list_devices():
    """All connected RealSense devices (name/serial/usb), D455 before D435 so
    the primary depth camera renders first, plus each stream's status."""
    if rs is None:
        return []
    by_serial = {}
    try:
        for device in _rs_context().devices:
            entry = {}
            for label, key in (("name", rs.camera_info.name),
                               ("serial", rs.camera_info.serial_number),
                               ("usb", rs.camera_info.usb_type_descriptor)):
                try:
                    entry[label] = device.get_info(key)
                except RuntimeError:
                    entry[label] = "?"
            by_serial[entry["serial"]] = entry
    except Exception:  # noqa: BLE001 - enumeration is best-effort
        pass
    # Enumeration can miss devices whose USB interfaces our own pipelines have
    # claimed — merge in the cameras we are actively streaming from.
    with _STREAMS_LOCK:
        streams = list(_STREAMS.items())
    for serial, stream in streams:
        status = stream.status()
        info = status.get("info") or {}
        if serial not in by_serial and (status["running"] or status["starting"]) and info:
            by_serial[serial] = {"name": info.get("name", "RealSense"),
                                 "serial": serial,
                                 "usb": info.get("usb", "?")}
    devices = []
    for serial, entry in by_serial.items():
        with _STREAMS_LOCK:
            stream = _STREAMS.get(serial)
        entry["status"] = stream.status() if stream else {}
        devices.append(entry)
    _enum_watchdog(devices)
    devices.sort(key=lambda d: d.get("name", ""), reverse=True)
    return devices


class Handler(BaseHTTPRequestHandler):
    server_version = "SmartroomDepthPage/2.0"

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
            if not serial:
                self.send_json({"ok": False, "message": "missing ?s=<serial>"}, 400)
                return
            self.send_json(extrinsic_status(serial))
            return
        if parsed.path == "/record/status":
            self.send_json(DEPTH_RECORDER.status())
            return
        self.send_bytes(b"Not found", "text/plain; charset=utf-8", 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        serial = (params.get("s") or [""])[0]
        if parsed.path == "/calibrate/extrinsic":
            if not serial:
                self.send_json({"ok": False, "message": "missing ?s=<serial>"}, 400)
                return
            ok, message = start_extrinsic_calibration(serial)
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
            ok, message = DEPTH_RECORDER.start(target, duration)
            self.send_json({"ok": ok, "message": message}, 200 if ok else 409)
            return
        self.send_bytes(b"Not found", "text/plain; charset=utf-8", 404)

    def serve_value(self, params, serial):
        if not serial:
            self.send_json({"ok": False, "message": "missing ?s=<serial>"}, 400)
            return
        try:
            x = min(1.0, max(0.0, float(params.get("x", ["0.5"])[0])))
            y = min(1.0, max(0.0, float(params.get("y", ["0.5"])[0])))
        except ValueError:
            self.send_json({"ok": False, "message": "bad coordinates"}, 400)
            return
        stream = get_stream(serial)
        meters, px, py = stream.depth_at(x, y)
        center_m, _, _ = stream.depth_at(0.5, 0.5)
        self.send_json({
            "ok": meters is not None,
            "m": round(meters, 3) if meters is not None else None,
            "px": px, "py": py,
            "center_m": round(center_m, 3) if center_m is not None else None,
        })

    def serve_stream(self, which, serial):
        if rs is None:
            self.send_bytes(
                f"pyrealsense2 is not installed on this node: {RS_IMPORT_ERROR}\n"
                f"Build it with setup_realsense_pi.sh, then restart this page.".encode("utf-8"),
                "text/plain; charset=utf-8", 503)
            return
        if not serial:
            self.send_bytes(b"missing ?s=<serial>", "text/plain; charset=utf-8", 400)
            return
        stream = get_stream(serial)
        # Bound how much video can sit in the kernel's send buffer: with the
        # default several-hundred-KB buffer a slow wifi client watches seconds-old
        # frames; ~128KB is a couple of frames at most.
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 128 * 1024)
        except OSError:
            pass
        stream.add_client()
        try:
            if not stream.wait_first_frame(FIRST_FRAME_TIMEOUT, view=True):
                message = stream.status().get("error") or "RealSense camera unavailable."
                self.send_bytes(message.encode("utf-8"), "text/plain; charset=utf-8", 503)
                return
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}")
            self.end_headers()
            for frame in stream.frames(which):
                try:
                    self.wfile.write(b"--" + STREAM_BOUNDARY.encode() + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
        finally:
            stream.remove_client()


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
        section.querySelector('[data-role="rgb"] img').src = '/rgb.mjpg?s=' + q + '&t=' + Date.now();
        section.querySelector('[data-role="depth"] img').src = '/depth.mjpg?s=' + q + '&t=' + Date.now();

        var center = section.querySelector('.readout b');
        setInterval(function () {
          fetch('/value?s=' + q + '&x=0.5&y=0.5')
            .then(function (r) { return r.json(); })
            .then(function (j) {
              center.textContent = j.center_m != null ? j.center_m.toFixed(2) + ' m' : 'no depth';
            }).catch(function () {});
        }, 600);

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
            // the device is present but its stream is down (USB replug, bus
            // starvation, ...) — restart the <img> streams, which re-opens the
            // pipeline server-side. Grace period so we don't kill a stream
            // that is still connecting.
            if (!st.running && !st.starting && Date.now() - (bumped[s] || 0) > 8000) {
              bumped[s] = Date.now();
              built[s].querySelectorAll('.stage img').forEach(function (img) {
                img.src = img.src.split('&t=')[0] + '&t=' + Date.now();
              });
            }
            var meta = built[s].querySelector('.meta');
            // live enumeration first: info.* is from the last pipeline run and
            // goes stale when the camera is re-plugged into a different port
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
    if rs is not None:
        try:
            _rs_context()  # create the context before any pipeline can exist
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not create RealSense context: {exc}")
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
