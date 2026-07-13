#!/usr/bin/env python3
"""Live RGB + depth view for the Intel RealSense D455, on port 8001.

Serves http://<node>.local:8001 with the color stream and the colorized depth
stream side by side (depth aligned to color, so the same pixel in both panes is
the same point in the room). Click anywhere on either pane to read the depth at
that pixel in meters; the center-pixel distance is shown continuously.

Uses the Intel RealSense SDK (pyrealsense2) — on a Pi that's built from source
into the venv by setup_realsense_pi.sh (no aarch64 pip wheel exists). Run with
the venv python:

    ~/CityOS/.venv/bin/python ~/CityOS/realsense_depth_page.py

This is separate from smartroom_video_page.py (port 8000): the D455 is its own
USB device, so both pages can run at once without fighting over a camera. Like
the main page, the RealSense pipeline starts on demand and is released a few
seconds after the last viewer leaves.
"""
import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


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
IDLE_TIMEOUT = 5.0   # release the camera this many seconds after the last viewer leaves
FIRST_FRAME_TIMEOUT = 8.0  # pipeline start + first frames can take a few seconds
JPEG_QUALITY = 80

# The SDK import is allowed to fail so the page can still come up and explain
# itself while librealsense is not built yet (or the module is missing).
try:
    import cv2
    import pyrealsense2 as rs
    RS_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on node state
    rs = None
    RS_IMPORT_ERROR = str(exc)

# Depth and color at the same resolution so the aligned panes map 1:1. Tried in
# order; the later entries are what the SDK will actually accept when the D455
# is on a USB 2 port (USB 3 allows the higher ones).
PROFILE_ATTEMPTS = (
    (848, 480, 30),
    (640, 480, 30),
    (640, 480, 15),
    (424, 240, 15),
)


class RealSenseStream:
    """One shared pyrealsense2 pipeline broadcasting color + colorized-depth
    JPEGs to any number of MJPEG viewers, plus the latest raw depth (meters)
    for point queries. Starts on demand, stops when idle."""

    def __init__(self):
        self.cond = threading.Condition()
        self.rgb_jpeg = None
        self.depth_jpeg = None
        self.depth_m = None          # float32 HxW, meters, aligned to color
        self.frame_id = 0
        self.clients = 0
        self.running = False
        self.starting = False
        self.last_active = 0.0
        self.error = RS_IMPORT_ERROR
        self.info = {}               # device name/serial/fw/usb + active profile

    def add_client(self):
        with self.cond:
            self.clients += 1
            self.last_active = time.monotonic()
            if not self.running and not self.starting and rs is not None:
                self.starting = True
                threading.Thread(target=self._run, daemon=True).start()

    def remove_client(self):
        with self.cond:
            self.clients = max(0, self.clients - 1)
            self.last_active = time.monotonic()

    def _start_pipeline(self):
        pipeline = rs.pipeline()
        last_error = None
        for width, height, fps in PROFILE_ATTEMPTS:
            config = rs.config()
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            try:
                profile = pipeline.start(config)
                return pipeline, profile, (width, height, fps)
            except RuntimeError as exc:
                last_error = exc
        raise last_error if last_error else RuntimeError("no usable D455 profile")

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
        colorizer = rs.colorizer()
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

        with self.cond:
            self.running = True
            self.starting = False
            self.error = None
            self.info = info
            self.last_active = time.monotonic()

        try:
            while True:
                with self.cond:
                    if self.clients == 0 and (time.monotonic() - self.last_active) > IDLE_TIMEOUT:
                        break
                frames = pipeline.wait_for_frames(5000)
                frames = align.process(frames)
                depth = frames.get_depth_frame()
                color = frames.get_color_frame()
                if not depth or not color:
                    continue
                color_img = np.asanyarray(color.get_data())
                depth_vis = np.asanyarray(colorizer.colorize(depth).get_data())
                depth_raw = np.asanyarray(depth.get_data())
                ok_rgb, rgb_jpeg = cv2.imencode(".jpg", color_img, encode_params)
                # the colorizer outputs RGB; cv2 encodes BGR
                ok_depth, depth_jpeg = cv2.imencode(
                    ".jpg", cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR), encode_params)
                if not (ok_rgb and ok_depth):
                    continue
                with self.cond:
                    self.rgb_jpeg = rgb_jpeg.tobytes()
                    self.depth_jpeg = depth_jpeg.tobytes()
                    self.depth_m = depth_raw.astype(np.float32) * depth_scale
                    self.frame_id += 1
                    self.cond.notify_all()
        except Exception as exc:  # noqa: BLE001 - USB drop etc.
            with self.cond:
                self.error = f"RealSense stream stopped: {exc}"
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass
            with self.cond:
                self.running = False
                self.cond.notify_all()

    def wait_first_frame(self, timeout):
        end = time.monotonic() + timeout
        with self.cond:
            while self.frame_id == 0 or self.rgb_jpeg is None:
                if not (self.running or self.starting):
                    return False
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return False
                self.cond.wait(timeout=remaining)
            return True

    def frames(self, which):
        last_sent = -1
        while True:
            with self.cond:
                while self.frame_id == last_sent and (self.running or self.starting):
                    self.cond.wait(timeout=2.0)
                if self.frame_id == last_sent and not (self.running or self.starting):
                    return
                last_sent = self.frame_id
                frame = self.rgb_jpeg if which == "rgb" else self.depth_jpeg
            if frame is not None:
                yield frame

    def depth_at(self, x_frac, y_frac):
        """Depth in meters at fractional image coords (0..1), median of the
        valid readings in a small window (single pixels are often 0/no-data)."""
        with self.cond:
            depth = self.depth_m
        if depth is None:
            return None, None, None
        h, w = depth.shape
        px = min(w - 1, max(0, int(x_frac * w)))
        py = min(h - 1, max(0, int(y_frac * h)))
        window = depth[max(0, py - 2):py + 3, max(0, px - 2):px + 3]
        valid = window[window > 0]
        meters = float(np.median(valid)) if valid.size else None
        return meters, px, py

    def status(self):
        with self.cond:
            return {
                "running": self.running,
                "starting": self.starting,
                "error": self.error,
                "info": self.info,
            }


STREAM = RealSenseStream()


class Handler(BaseHTTPRequestHandler):
    server_version = "SmartroomDepthPage/1.0"

    def log_message(self, fmt, *args):
        return

    def send_bytes(self, body, content_type="text/html; charset=utf-8", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload, status=200):
        self.send_bytes(json.dumps(payload).encode("utf-8"),
                        "application/json; charset=utf-8", status)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(PAGE.encode("utf-8"))
            return
        if parsed.path in ("/rgb.mjpg", "/depth.mjpg"):
            self.serve_stream("rgb" if parsed.path == "/rgb.mjpg" else "depth")
            return
        if parsed.path == "/value":
            self.serve_value(parsed.query)
            return
        if parsed.path == "/info":
            self.send_json(STREAM.status())
            return
        self.send_bytes(b"Not found", "text/plain; charset=utf-8", 404)

    def serve_value(self, query):
        params = urllib.parse.parse_qs(query)
        try:
            x = min(1.0, max(0.0, float(params.get("x", ["0.5"])[0])))
            y = min(1.0, max(0.0, float(params.get("y", ["0.5"])[0])))
        except ValueError:
            self.send_json({"ok": False, "message": "bad coordinates"}, 400)
            return
        meters, px, py = STREAM.depth_at(x, y)
        center_m, _, _ = STREAM.depth_at(0.5, 0.5)
        self.send_json({
            "ok": meters is not None,
            "m": round(meters, 3) if meters is not None else None,
            "px": px, "py": py,
            "center_m": round(center_m, 3) if center_m is not None else None,
        })

    def serve_stream(self, which):
        if rs is None:
            self.send_bytes(
                f"pyrealsense2 is not installed on this node: {RS_IMPORT_ERROR}\n"
                f"Build it with setup_realsense_pi.sh, then restart this page.".encode("utf-8"),
                "text/plain; charset=utf-8", 503)
            return
        STREAM.add_client()
        try:
            if not STREAM.wait_first_frame(FIRST_FRAME_TIMEOUT):
                message = STREAM.status().get("error") or "RealSense camera unavailable."
                self.send_bytes(message.encode("utf-8"), "text/plain; charset=utf-8", 503)
                return
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}")
            self.end_headers()
            for frame in STREAM.frames(which):
                try:
                    self.wfile.write(b"--" + STREAM_BOUNDARY.encode() + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
        finally:
            STREAM.remove_client()


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>D455 Depth View</title>
  <style>
    :root { color-scheme: light; --bg:#f6f8fb; --panel:#fff; --ink:#18202a;
            --muted:#687384; --line:#d9e0e8; --accent:#1267c3; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Arial, Helvetica, sans-serif; background:var(--bg); color:var(--ink); }
    header { padding:24px clamp(16px,4vw,44px) 14px; border-bottom:1px solid var(--line); background:var(--panel); }
    h1 { margin:0; font-size:clamp(24px,4vw,36px); }
    header p { margin:6px 0 0; color:var(--muted); font-size:15px; }
    #usb-warn { color:#b23c3c; font-weight:700; }
    .wrap { width:min(1400px, calc(100% - 32px)); margin:0 auto; }
    .panes { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:16px; margin:20px auto; }
    .pane h2 { margin:0 0 8px; font-size:18px; }
    .stage { position:relative; background:#10151c; border:1px solid var(--line);
             border-radius:10px; overflow:hidden; cursor:crosshair; }
    .stage img { width:100%; display:block; }
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
    .readout { margin:4px 0 30px; font-size:16px; }
    .readout b { font-size:20px; }
    .err { background:#fdeaea; border:1px solid #e6b5b5; color:#8d2323;
           border-radius:8px; padding:12px 16px; margin:16px auto; display:none; }
  </style>
</head>
<body>
  <header>
    <h1>D455 Depth View</h1>
    <p id="dev-info">Connecting to camera&hellip;</p>
  </header>
  <div class="wrap">
    <div class="err" id="err"></div>
    <div class="panes">
      <div class="pane">
        <h2>Color (RGB)</h2>
        <div class="stage" id="rgb-stage">
          <img id="rgb" alt="RGB stream">
          <span class="cross"></span>
          <span class="marker" id="rgb-marker"></span>
          <span class="tag" id="rgb-tag"></span>
        </div>
      </div>
      <div class="pane">
        <h2>Depth (colorized, aligned to color)</h2>
        <div class="stage" id="depth-stage">
          <img id="depth" alt="Depth stream">
          <span class="cross"></span>
          <span class="marker" id="depth-marker"></span>
          <span class="tag" id="depth-tag"></span>
        </div>
      </div>
    </div>
    <div class="readout">
      Center distance: <b id="center">&mdash;</b>
      <span style="color:var(--muted)">&nbsp;&mdash; click anywhere on either image to measure that point</span>
    </div>
  </div>
  <script>
    (function () {
      var err = document.getElementById('err');
      function start() {
        document.getElementById('rgb').src = '/rgb.mjpg?t=' + Date.now();
        document.getElementById('depth').src = '/depth.mjpg?t=' + Date.now();
      }
      function showInfo() {
        fetch('/info').then(function (r) { return r.json(); }).then(function (s) {
          if (s.error) { err.textContent = s.error; err.style.display = ''; }
          else { err.style.display = 'none'; }
          if (s.info && s.info.name) {
            var usb = s.info.usb || '?';
            var warn = usb.indexOf('3') !== 0
              ? ' <span id="usb-warn">&#9888; on USB ' + usb +
                ' — replug into a blue USB 3 port for full resolution</span>' : '';
            document.getElementById('dev-info').innerHTML =
              s.info.name + ' · S/N ' + s.info.serial + ' · FW ' + s.info.firmware +
              ' · ' + s.info.profile + ' · USB ' + usb + warn;
          }
        }).catch(function () {});
      }
      setInterval(showInfo, 3000); showInfo();

      function pollCenter() {
        fetch('/value?x=0.5&y=0.5').then(function (r) { return r.json(); }).then(function (j) {
          document.getElementById('center').textContent =
            j.center_m != null ? j.center_m.toFixed(2) + ' m' : 'no depth';
        }).catch(function () {});
      }
      setInterval(pollCenter, 500);

      function wire(stageId) {
        var stage = document.getElementById(stageId);
        stage.addEventListener('click', function (e) {
          var rect = stage.getBoundingClientRect();
          var x = (e.clientX - rect.left) / rect.width;
          var y = (e.clientY - rect.top) / rect.height;
          fetch('/value?x=' + x.toFixed(4) + '&y=' + y.toFixed(4))
            .then(function (r) { return r.json(); })
            .then(function (j) {
              ['rgb', 'depth'].forEach(function (p) {
                var m = document.getElementById(p + '-marker');
                var t = document.getElementById(p + '-tag');
                m.style.left = (x * 100) + '%'; m.style.top = (y * 100) + '%';
                t.style.left = (x * 100) + '%'; t.style.top = (y * 100) + '%';
                t.textContent = j.m != null ? j.m.toFixed(2) + ' m' : 'no depth';
                m.style.display = ''; t.style.display = '';
              });
            }).catch(function () {});
        });
      }
      wire('rgb-stage'); wire('depth-stage');
      start();
    })();
  </script>
</body>
</html>"""


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"D455 depth page running at http://0.0.0.0:{PORT}")
    if RS_IMPORT_ERROR:
        print(f"WARNING: pyrealsense2 not available yet: {RS_IMPORT_ERROR}")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
