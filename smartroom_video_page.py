#!/usr/bin/env python3
import html
import json
import mimetypes
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


# Per-node overrides from ~/CityOS/node.env (gitignored, machine-local), same
# loader as capture.py — so e.g. SMARTROOM_CAMERA_SIZE pins apply to both the
# page and the recordings it launches. Real environment wins over the file.
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

PORT = int(os.environ.get("SMARTROOM_VIDEO_PORT", "8000"))
RECORDINGS_DIR = Path.home() / "Videos" / "Smartroom Recordings"
DATA_DIR = Path.home() / "CityOS" / "data"
RECORD_SCRIPT = Path.home() / "CityOS" / "run_smartroom_capture.sh"
PREVIEW_PATH = "/dev/shm/smartroom_preview.jpg"  # recorder writes the latest frame here
PHOTOS_DIR = DATA_DIR / "photos"  # full-resolution stills from the Snap photo button
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".h264", ".ts"}

# Each node has a different camera, so auto-detect the first USB camera via its
# stable /dev/v4l/by-id/ symlink (matches capture.py's detection), with the
# SMARTROOM_CAMERA* env vars overriding.
def _detect_camera():
    override = os.environ.get("SMARTROOM_CAMERA")
    if override:
        return override
    by_id = Path("/dev/v4l/by-id")
    if by_id.is_dir():
        cams = sorted(p for p in by_id.iterdir() if p.name.endswith("-video-index0"))
        if cams:
            return str(cams[0])
    return "/dev/video0"


def _detect_camera_size(device, cap_width=640):
    """Largest MJPG mode <= cap_width for the LIVE PREVIEW. Capped at 640 wide so
    the copy-passthrough MJPEG stream stays light enough to render smoothly in a
    browser over wifi (a full 1280x720 feed is ~3.5 MB/s). Recording is separate
    (capture.py) and still uses full resolution. SMARTROOM_CAMERA_SIZE overrides."""
    override = os.environ.get("SMARTROOM_CAMERA_SIZE")
    if override:
        return override
    try:
        out = subprocess.run(
            ["v4l2-ctl", "-d", device, "--list-formats-ext"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        sizes, in_mjpg = [], False
        for line in out.splitlines():
            s = line.strip()
            if "]:" in s and "'" in s:
                in_mjpg = "MJPG" in s
            elif in_mjpg and s.startswith("Size: Discrete"):
                w, h = (int(v) for v in s.split()[-1].split("x"))
                sizes.append((w, h))
        usable = [(w, h) for w, h in sizes if w <= cap_width]
        if usable:
            w, h = max(usable, key=lambda wh: wh[0] * wh[1])
            return f"{w}x{h}"
    except Exception:
        pass
    return "640x480"


CAMERA_DEVICE = _detect_camera()
CAMERA_SIZE = _detect_camera_size(CAMERA_DEVICE)
STREAM_BOUNDARY = "frame"
IDLE_TIMEOUT = 5.0  # release the camera this many seconds after the last viewer leaves
FIRST_FRAME_TIMEOUT = 4.0


def video_files():
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = [p for p in RECORDINGS_DIR.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    files.extend(p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def video_token(path):
    try:
        if DATA_DIR.resolve() in path.resolve().parents:
            return "data/" + path.relative_to(DATA_DIR).as_posix()
    except Exception:
        pass
    return path.name


def video_label(path):
    try:
        if DATA_DIR.resolve() in path.resolve().parents:
            return path.relative_to(DATA_DIR).as_posix()
    except Exception:
        pass
    return path.name


def resolve_video_path(encoded_name):
    token = urllib.parse.unquote(encoded_name)
    candidates = []
    if token.startswith("data/"):
        candidates.append((DATA_DIR / token.removeprefix("data/")).resolve())
    candidates.append((RECORDINGS_DIR / token).resolve())

    roots = [DATA_DIR.resolve(), RECORDINGS_DIR.resolve()]
    for path in candidates:
        # Videos by extension, plus each recording's metadata.json and per-frame
        # *_timestamps.csv (so the laptop's "Save All" can beam them and reproduce
        # the real_dataset layout: metadata.json at the rec root, timestamps csv
        # next to its video in streams/).
        if path.is_file() and (path.suffix.lower() in VIDEO_EXTENSIONS
                               or path.name == "metadata.json"
                               or path.name.endswith("_timestamps.csv")):
            if any(root == path or root in path.parents for root in roots):
                return path
    return None


def recording_dir(path):
    """If a video lives inside a dataset recording (data/.../rec_*/streams/x),
    return its recording folder (the parent of streams/); otherwise None."""
    try:
        resolved = path.resolve()
        data_root = DATA_DIR.resolve()
        if data_root in resolved.parents and resolved.parent.name == "streams":
            rec = resolved.parent.parent
            if rec != data_root and data_root in rec.parents:
                return rec
    except Exception:
        pass
    return None


def dataset_token(rec_dir):
    return "data/" + rec_dir.relative_to(DATA_DIR).as_posix()


def resolve_dataset_path(encoded_name):
    token = urllib.parse.unquote(encoded_name)
    if not token.startswith("data/"):
        return None
    candidate = (DATA_DIR / token.removeprefix("data/")).resolve()
    root = DATA_DIR.resolve()
    if candidate.is_dir() and candidate != root and root in candidate.parents:
        return candidate
    return None


def day_token(day_dir):
    return "data/" + day_dir.relative_to(DATA_DIR).as_posix()


def day_folders():
    """All day_* folders that contain at least one recording, newest first."""
    if not DATA_DIR.exists():
        return []
    days = [d for d in DATA_DIR.glob("day_*") if d.is_dir() and any(d.glob("rec_*"))]
    return sorted(days, key=lambda d: d.name, reverse=True)


def resolve_day_path(encoded_name):
    token = urllib.parse.unquote(encoded_name)
    if not token.startswith("data/"):
        return None
    candidate = (DATA_DIR / token.removeprefix("data/")).resolve()
    root = DATA_DIR.resolve()
    # must be a direct child of data/ named day_*
    if candidate.is_dir() and candidate.parent == root and candidate.name.startswith("day_"):
        return candidate
    return None


def local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "smartroom.local"
    finally:
        sock.close()


class CameraStream:
    """Single shared ffmpeg capture of the webcam, broadcast as MJPEG to any
    number of browser viewers. Starts on demand and releases the camera shortly
    after the last viewer disconnects so the Recorder app can use it."""

    def __init__(self, device, size):
        self.device = device
        self.size = size
        self.cond = threading.Condition()
        self.frame = None
        self.frame_id = 0
        self.clients = 0
        self.process = None
        self.running = False
        self.last_active = 0.0
        self.recording = False  # when True, leave the camera free for the recorder

    def add_client(self):
        with self.cond:
            self.clients += 1
            self.last_active = time.monotonic()
            if not self.running and not self.recording:
                self._start_locked()

    def remove_client(self):
        with self.cond:
            self.clients = max(0, self.clients - 1)
            self.last_active = time.monotonic()

    def shutdown_for_recording(self):
        """Release the camera so run_smartroom_capture can open the device."""
        with self.cond:
            self.recording = True
            proc = self.process
        if proc is not None:
            self._stop(proc)

    def resume_after_recording(self):
        with self.cond:
            self.recording = False

    def _start_locked(self):
        self.frame = None
        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-f", "v4l2", "-input_format", "mjpeg",
            "-video_size", self.size, "-i", self.device,
            "-c:v", "copy", "-f", "mpjpeg", "pipe:1",
        ]
        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
            )
        except Exception:
            self.process = None
            return
        self.running = True
        threading.Thread(target=self._reader, args=(self.process,), daemon=True).start()
        threading.Thread(target=self._monitor, args=(self.process,), daemon=True).start()

    def _reader(self, proc):
        stream = proc.stdout
        try:
            while True:
                content_length = None
                while True:
                    line = stream.readline()
                    if not line:
                        raise EOFError
                    stripped = line.strip()
                    if stripped.lower().startswith(b"content-length:"):
                        content_length = int(stripped.split(b":", 1)[1])
                    elif stripped == b"" and content_length is not None:
                        break
                body = bytearray()
                remaining = content_length
                while remaining > 0:
                    chunk = stream.read(remaining)
                    if not chunk:
                        raise EOFError
                    body += chunk
                    remaining -= len(chunk)
                with self.cond:
                    self.frame = bytes(body)
                    self.frame_id += 1
                    self.cond.notify_all()
        except Exception:
            pass
        finally:
            self._stop(proc)

    def _monitor(self, proc):
        while True:
            time.sleep(1.0)
            if proc.poll() is not None:
                break
            with self.cond:
                if self.clients == 0 and (time.monotonic() - self.last_active) > IDLE_TIMEOUT:
                    break
        self._stop(proc)

    def _stop(self, proc):
        with self.cond:
            if self.process is proc:
                self.process = None
                self.running = False
            self.cond.notify_all()
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def wait_first_frame(self, timeout):
        end = time.monotonic() + timeout
        with self.cond:
            while self.frame is None:
                if not self.running:
                    return False
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return False
                self.cond.wait(timeout=remaining)
            return True

    def frames(self):
        last_sent = -1
        while True:
            with self.cond:
                while self.frame_id == last_sent and self.running:
                    self.cond.wait(timeout=2.0)
                    if self.frame_id == last_sent and not self.running:
                        return
                if not self.running and self.frame_id == last_sent:
                    return
                last_sent = self.frame_id
                frame = self.frame
            if frame is not None:
                yield frame


CAMERA = CameraStream(CAMERA_DEVICE, CAMERA_SIZE)


class Recorder:
    """Runs run_smartroom_capture.sh as a background process and tracks timing
    so the web page can show a countdown and elapsed time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.start_monotonic = 0.0
        self.duration = 0

    def start(self, duration):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return False, "A recording is already in progress."
            if not RECORD_SCRIPT.exists():
                return False, "Recorder script not found on the Pi."
        # Free the camera from the live stream so the recorder can open it. The
        # recorder writes preview frames to PREVIEW_PATH, which we serve directly.
        CAMERA.shutdown_for_recording()
        try:
            os.unlink(PREVIEW_PATH)
        except OSError:
            pass
        time.sleep(0.8)  # let the device free up before the recorder opens it
        env = dict(os.environ)
        env["SMARTROOM_PREVIEW"] = PREVIEW_PATH  # capture.py writes the live frame here
        try:
            process = subprocess.Popen(
                ["/bin/sh", str(RECORD_SCRIPT), str(duration)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # own process group so cancel can kill children
                env=env,
            )
        except Exception as exc:
            CAMERA.resume_after_recording()
            return False, f"Could not start recorder: {exc}"
        with self.lock:
            self.process = process
            self.start_monotonic = time.monotonic()
            self.duration = duration
        threading.Thread(target=self._watch, args=(process,), daemon=True).start()
        return True, "Recording started."

    def cancel(self):
        with self.lock:
            process = self.process
        if process is None or process.poll() is not None:
            return False, "No recording in progress."
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
        return True, "Recording cancelled."

    def _watch(self, process):
        process.wait()
        CAMERA.resume_after_recording()
        try:
            os.unlink(PREVIEW_PATH)
        except OSError:
            pass

    def status(self):
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            elapsed = time.monotonic() - self.start_monotonic if self.start_monotonic else 0.0
            remaining = max(0.0, self.duration - elapsed) if running else 0.0
            return {
                "running": running,
                "duration": self.duration,
                "elapsed": round(elapsed, 1),
                "remaining": round(remaining, 1),
            }


RECORDER = Recorder()

PHOTO_LOCK = threading.Lock()


class CalibrateJob:
    """Runs calibrate_camera.py --photos (venv python) in the background and
    keeps its outcome for the page to poll. Photos mode never opens the camera,
    so the live view / recordings are unaffected while it runs."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.ok = None       # None until a run has finished
        self.message = ""

    def start(self):
        with self.lock:
            if self.running:
                return False, "Calibration already running."
            self.running = True
            self.ok, self.message = None, "Running…"
        threading.Thread(target=self._run, daemon=True).start()
        return True, "Calibration started."

    def _run(self):
        home = Path.home() / "CityOS"
        python = home / ".venv" / "bin" / "python"
        if not python.exists():
            python = Path("/usr/bin/python3")
        try:
            proc = subprocess.run(
                [str(python), str(home / "calibrate_camera.py"), "--photos"],
                capture_output=True, text=True, timeout=600, cwd=str(home),
            )
            # The script's summary (RMS, fx/fy, saved path) is on stderr.
            lines = [l for l in (proc.stderr or "").strip().splitlines() if l.strip()]
            tail = " | ".join(lines[-3:]) if lines else "no output"
            with self.lock:
                self.ok = proc.returncode == 0
                self.message = tail
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.ok, self.message = False, str(e)
        finally:
            with self.lock:
                self.running = False

    def status(self):
        with self.lock:
            return {"running": self.running, "ok": self.ok, "message": self.message}


CALIBRATOR = CalibrateJob()


def calibration_summary():
    """One-line description of this node's saved calibrations, for the page."""
    calib_dir = Path.home() / "CityOS" / "calibration"
    entries = []
    for p in sorted(calib_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            when = (d.get("calibrated_at") or "")[:10]
            entries.append(f"{p.stem} — RMS {d.get('rms')} px, "
                           f"{d.get('image_size', ['?', '?'])[0]}x{d.get('image_size', ['?', '?'])[1]}, {when}")
        except (OSError, ValueError):
            continue
    return entries


def photo_capture_size():
    """Full capture resolution for stills (not the 640-capped preview size):
    SMARTROOM_CAMERA_SIZE override wins (node.env / env), else the largest MJPG
    mode <= 1280 wide — the same rule capture.py records with."""
    override = os.environ.get("SMARTROOM_CAMERA_SIZE")
    if override:
        return override
    return _detect_camera_size(CAMERA_DEVICE, cap_width=1280)


def snap_photo():
    """Grab one full-resolution still to data/photos/. Briefly borrows the camera
    from the live preview (same release/resume dance the recorder uses); reads a
    handful of frames and keeps the last so auto-exposure has settled.
    Returns (ok, message, filename|None)."""
    if RECORDER.status()["running"]:
        return False, "Recording in progress — try after it finishes.", None
    if not PHOTO_LOCK.acquire(blocking=False):
        return False, "Another photo is being taken.", None
    try:
        PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
        name = time.strftime("photo_%Y%m%d_%H%M%S.jpg")
        dest = PHOTOS_DIR / name
        CAMERA.shutdown_for_recording()
        try:
            # The preview's ffmpeg can take a moment to actually release the
            # device (and a viewer reconnect can race a new one in) — retry
            # briefly on "busy" instead of failing the snap.
            for attempt in range(5):
                proc = subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error",
                     "-f", "v4l2", "-input_format", "mjpeg",
                     "-video_size", photo_capture_size(), "-i", CAMERA_DEVICE,
                     "-frames:v", "6", "-update", "1", str(dest)],
                    capture_output=True, text=True, timeout=20,
                )
                if proc.returncode == 0 or "busy" not in (proc.stderr or "").lower():
                    break
                time.sleep(1)
        finally:
            CAMERA.resume_after_recording()
        if proc.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
            dest.unlink(missing_ok=True)
            err = (proc.stderr or "").strip().splitlines()
            return False, err[-1] if err else "ffmpeg failed", None
        return True, f"Saved {name}", name
    except subprocess.TimeoutExpired:
        return False, "Camera timed out.", None
    finally:
        PHOTO_LOCK.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "SmartroomVideoPage/1.1"

    def log_message(self, fmt, *args):
        return

    def send_bytes(self, body, content_type="text/html; charset=utf-8", status=200, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if content_type.startswith("text/html") else "public, max-age=3600")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.show_index()
            return
        if parsed.path == "/stream.mjpg":
            self.serve_stream()
            return
        if parsed.path == "/calibrate/status":
            self.send_bytes(json.dumps(CALIBRATOR.status()).encode("utf-8"),
                            "application/json; charset=utf-8")
            return
        if parsed.path == "/record/status":
            self.send_bytes(json.dumps(RECORDER.status()).encode("utf-8"),
                            "application/json; charset=utf-8")
            return
        if parsed.path == "/preview.jpg":
            self.serve_preview()
            return
        if parsed.path == "/recordings":
            self.serve_recordings()
            return
        if parsed.path.startswith("/photo/"):
            self.serve_photo(parsed.path.removeprefix("/photo/"))
            return
        if parsed.path.startswith("/video/"):
            self.serve_video(parsed.path.removeprefix("/video/"), download=False)
            return
        if parsed.path.startswith("/download/"):
            self.serve_video(parsed.path.removeprefix("/download/"), download=True)
            return
        if parsed.path.startswith("/dataset/"):
            self.serve_dataset(parsed.path.removeprefix("/dataset/"))
            return
        if parsed.path.startswith("/day/"):
            self.serve_day(parsed.path.removeprefix("/day/"))
            return
        self.send_bytes(b"Not found", "text/plain; charset=utf-8", 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/record":
            self.start_recording()
            return
        if parsed.path == "/record/cancel":
            ok, message = RECORDER.cancel()
            payload = json.dumps({"ok": ok, "message": message})
            self.send_bytes(payload.encode("utf-8"), "application/json; charset=utf-8",
                            200 if ok else 409)
            return
        if parsed.path == "/photo":
            ok, message, name = snap_photo()
            payload = json.dumps({"ok": ok, "message": message, "name": name})
            self.send_bytes(payload.encode("utf-8"), "application/json; charset=utf-8",
                            200 if ok else 409)
            return
        if parsed.path == "/calibrate":
            ok, message = CALIBRATOR.start()
            payload = json.dumps({"ok": ok, "message": message})
            self.send_bytes(payload.encode("utf-8"), "application/json; charset=utf-8",
                            200 if ok else 409)
            return
        if parsed.path == "/photo/delete":
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            params = urllib.parse.parse_qs(raw.decode("utf-8", "ignore"))
            name = (params.get("name") or [""])[0]
            path = (PHOTOS_DIR / name).resolve()
            # Same traversal guard as serve_photo: flat dir, .jpg only.
            if (not name or "/" in name or not path.name.endswith(".jpg")
                    or path.parent != PHOTOS_DIR.resolve() or not path.is_file()):
                self.send_bytes(b'{"ok": false, "message": "no such photo"}',
                                "application/json; charset=utf-8", 404)
                return
            try:
                path.unlink()
                self.send_bytes(b'{"ok": true}', "application/json; charset=utf-8")
            except OSError as e:
                payload = json.dumps({"ok": False, "message": str(e)})
                self.send_bytes(payload.encode("utf-8"), "application/json; charset=utf-8", 500)
            return
        self.send_bytes(b"Not found", "text/plain; charset=utf-8", 404)

    def start_recording(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        params = urllib.parse.parse_qs(raw.decode("utf-8", "ignore"))
        try:
            duration = int(float(params.get("duration", ["30"])[0]))
        except (ValueError, TypeError):
            duration = 30
        duration = max(1, min(duration, 3600))
        ok, message = RECORDER.start(duration)
        payload = json.dumps({"ok": ok, "message": message, **RECORDER.status()})
        self.send_bytes(payload.encode("utf-8"), "application/json; charset=utf-8",
                        200 if ok else 409)

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            files = video_files()
            body = f"Smartroom Videos: {len(files)} recording(s)".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def serve_stream(self):
        CAMERA.add_client()
        try:
            if not CAMERA.wait_first_frame(FIRST_FRAME_TIMEOUT):
                self.send_bytes(
                    b"Camera unavailable (it may be in use by the Recorder app).",
                    "text/plain; charset=utf-8",
                    503,
                )
                return
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}")
            self.end_headers()
            for frame in CAMERA.frames():
                try:
                    self.wfile.write(b"--" + STREAM_BOUNDARY.encode() + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
        finally:
            CAMERA.remove_client()

    def show_index(self):
        files = video_files()
        cards = []
        for path in files:
            name = video_label(path)
            safe_name = html.escape(name)
            quoted = urllib.parse.quote(video_token(path), safe="")
            size_mb = path.stat().st_size / (1024 * 1024)
            rec = recording_dir(path)
            dataset_link = ""
            if rec is not None:
                ds_quoted = urllib.parse.quote(dataset_token(rec), safe="")
                dataset_link = f'<a href="/dataset/{ds_quoted}">Download dataset (.zip)</a>'
            cards.append(
                f"""
                <article class="video-card">
                  <video controls preload="metadata" src="/video/{quoted}"></video>
                  <div class="video-info">
                    <h2>{safe_name}</h2>
                    <p>{size_mb:.1f} MB</p>
                    <div class="actions">
                      <a href="/video/{quoted}" target="_blank">Open</a>
                      <a href="/download/{quoted}">Download video</a>
                      {dataset_link}
                    </div>
                  </div>
                </article>
                """
            )

        empty = """
          <article class="empty">
            <h2>No recordings yet</h2>
            <p>Record a video with Smartroom Recorder, then refresh this page.</p>
          </article>
        """

        day_links = []
        for d in day_folders():
            count = len(list(d.glob("rec_*")))
            dq = urllib.parse.quote(day_token(d), safe="")
            plural = "s" if count != 1 else ""
            day_links.append(
                f'<a href="/day/{dq}">{html.escape(d.name)} '
                f'<span>({count} recording{plural}, .zip)</span></a>'
            )
        days_html = "".join(day_links)

        # Full-resolution stills from the Snap photo button, newest first: a
        # thumbnail card each, with view (new tab) and delete.
        photo_items = []
        if PHOTOS_DIR.is_dir():
            for p in sorted(PHOTOS_DIR.glob("*.jpg"), key=lambda q: q.stat().st_mtime, reverse=True):
                pq = urllib.parse.quote(p.name, safe="")
                safe = html.escape(p.name)
                kb = p.stat().st_size // 1024
                photo_items.append(f'''
                <div class="photo-card" style="display:inline-block;width:220px;margin:6px;vertical-align:top;">
                  <a href="/photo/{pq}" target="_blank" title="Open full size">
                    <img src="/photo/{pq}" alt="{safe}" loading="lazy"
                         style="width:100%;border-radius:8px;display:block;">
                  </a>
                  <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;margin-top:4px;font-size:0.8em;">
                    <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{safe} ({kb} KB)</span>
                    <button type="button" class="photo-del" data-name="{safe}"
                            style="flex-shrink:0;cursor:pointer;">Delete</button>
                  </div>
                </div>''')
        photos_html = "".join(photo_items) or '<span class="photo-none">No photos yet.</span>'
        cal_entries = calibration_summary()
        calibration_html = ("Calibrated: " + "; ".join(html.escape(e) for e in cal_entries)
                            if cal_entries else
                            "Not calibrated yet — snap the checkerboard at 10+ positions, then Calibrate.")

        body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smartroom Live</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #687384;
      --line: #d9e0e8;
      --accent: #1267c3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      padding: 28px clamp(16px, 4vw, 44px) 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(26px, 4vw, 42px);
      line-height: 1.1;
    }}
    header p {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 16px;
    }}
    .wrap {{
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
    }}
    .live {{
      margin: 24px auto 8px;
      width: min(560px, calc(100% - 32px));
    }}
    .live-head {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
    }}
    .live-head h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .live-dot {{
      width: 11px;
      height: 11px;
      border-radius: 50%;
      background: #d62b2b;
      box-shadow: 0 0 0 0 rgba(214, 43, 43, 0.6);
      animation: pulse 1.6s infinite;
    }}
    @keyframes pulse {{
      0% {{ box-shadow: 0 0 0 0 rgba(214, 43, 43, 0.6); }}
      70% {{ box-shadow: 0 0 0 9px rgba(214, 43, 43, 0); }}
      100% {{ box-shadow: 0 0 0 0 rgba(214, 43, 43, 0); }}
    }}
    .live-stage {{
      position: relative;
      background: #10151c;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      aspect-ratio: 4 / 3;
      box-shadow: 0 2px 10px rgba(20, 32, 48, 0.10);
    }}
    .live-stage img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }}
    .live-status {{
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 14px;
      color: #cdd6e0;
      text-align: center;
      padding: 24px;
    }}
    .live-status.hidden {{ display: none; }}
    .live-status button {{
      color: #fff;
      background: var(--accent);
      border: 0;
      padding: 10px 16px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 15px;
      cursor: pointer;
    }}
    .record {{
      margin: 18px auto 0;
      width: min(560px, calc(100% - 32px));
    }}
    .record h2 {{ margin: 0 0 12px; font-size: 22px; }}
    .record-controls {{
      display: flex;
      gap: 12px;
      align-items: flex-end;
      flex-wrap: wrap;
    }}
    .record-controls label {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 14px;
      color: var(--muted);
    }}
    .record-controls input {{
      width: 130px;
      padding: 9px 10px;
      font-size: 15px;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    #rec-btn {{
      color: #fff;
      background: #d62b2b;
      border: 0;
      padding: 10px 20px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 15px;
      cursor: pointer;
    }}
    #rec-btn:disabled {{ background: #b98c8c; cursor: default; }}
    #rec-cancel {{
      color: #fff;
      background: #4a5563;
      border: 0;
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 14px;
      cursor: pointer;
      margin-left: 6px;
    }}
    #rec-cancel:disabled {{ background: #99a1ad; cursor: default; }}
    .record-status {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 14px;
      font-weight: 700;
      font-size: 16px;
    }}
    .record-status.hidden {{ display: none; }}
    .rec-dot {{
      width: 11px;
      height: 11px;
      border-radius: 50%;
      background: #d62b2b;
      animation: pulse 1.6s infinite;
    }}
    .section-head {{
      margin: 36px 0 4px;
      font-size: 22px;
    }}
    .day-list {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 10px;
    }}
    .day-list a {{
      display: inline-block;
      color: #fff;
      background: var(--accent);
      text-decoration: none;
      padding: 10px 14px;
      border-radius: 6px;
      font-weight: 700;
      width: fit-content;
    }}
    .day-list a span {{ font-weight: 400; opacity: 0.85; }}
    main {{
      margin: 16px auto 48px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 18px;
    }}
    .video-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 4px rgba(20, 32, 48, 0.06);
    }}
    .video-card video {{
      width: 100%;
      aspect-ratio: 4 / 3;
      display: block;
      background: #10151c;
    }}
    .video-info {{ padding: 14px; }}
    .video-card h2 {{
      margin: 0;
      font-size: 16px;
      line-height: 1.3;
      overflow-wrap: anywhere;
    }}
    .video-info p {{
      margin: 6px 0 12px;
      color: var(--muted);
    }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .actions a {{
      color: #fff;
      background: var(--accent);
      text-decoration: none;
      padding: 9px 12px;
      border-radius: 6px;
      font-weight: 700;
    }}
    .actions a:last-child {{
      color: var(--accent);
      background: #e7f1ff;
    }}
    .empty {{
      grid-column: 1 / -1;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 28px;
    }}
    .empty h2 {{ font-size: 22px; margin: 0 0 8px; }}
  </style>
</head>
<body>
  <header>
    <h1>Smartroom Live</h1>
    <p>{len(files)} recording{"s" if len(files) != 1 else ""} available below</p>
  </header>

  <section class="wrap live">
    <div class="live-head">
      <span class="live-dot"></span>
      <h2>Live view</h2>
    </div>
    <div class="live-stage">
      <img id="live" alt="Live camera">
      <div class="live-status" id="live-status">
        <div id="live-msg">Connecting to camera&hellip;</div>
        <button type="button" id="live-retry" style="display:none">Retry</button>
      </div>
    </div>
  </section>

  <section class="wrap record">
    <h2>Record</h2>
    <div class="record-controls">
      <label>Length (seconds)
        <input type="number" id="rec-seconds" min="1" max="3600" value="30">
      </label>
      <button type="button" id="rec-btn">Record</button>
    </div>
    <div class="record-status hidden" id="rec-status">
      <span class="rec-dot"></span>
      <span id="rec-text">Recording&hellip;</span>
      <button type="button" id="rec-cancel">Cancel</button>
    </div>
    <div class="live-stage" id="rec-preview-stage" style="display:none; margin-top:14px;">
      <img id="rec-preview" alt="Recording preview">
    </div>
  </section>

  <section class="wrap record">
    <h2>Photos</h2>
    <div class="record-controls">
      <button type="button" id="photo-btn">&#128247; Snap photo</button>
      <button type="button" id="cal-btn" title="Checkerboard calibration from the photos below (snap the board at 10+ positions first)">&#128208; Calibrate from photos</button>
      <span id="photo-msg"></span>
    </div>
    <div id="cal-info" style="font-size:0.85em;opacity:0.85;margin:6px 0;">{calibration_html}</div>
    <div class="day-list" id="photo-list">{photos_html}</div>
  </section>

  <section class="wrap days">
    <h2 class="section-head">Full days</h2>
    <div class="day-list">{days_html}</div>
  </section>

  <h2 class="section-head wrap">Recordings</h2>
  <main class="wrap">
    {''.join(cards) if cards else empty}
  </main>

  <script>
    (function () {{
      var img = document.getElementById('live');
      var status = document.getElementById('live-status');
      var msg = document.getElementById('live-msg');
      var retry = document.getElementById('live-retry');
      var retryTimer = null;

      function showStatus(text, allowRetry) {{
        msg.innerHTML = text;
        retry.style.display = allowRetry ? '' : 'none';
        status.classList.remove('hidden');
      }}

      function start() {{
        if (retryTimer) {{ clearTimeout(retryTimer); retryTimer = null; }}
        showStatus('Connecting to camera&hellip;', false);
        img.src = '/stream.mjpg?t=' + Date.now();
      }}

      img.addEventListener('load', function () {{
        status.classList.add('hidden');
      }});

      img.addEventListener('error', function () {{
        img.removeAttribute('src');
        showStatus('Live view unavailable.<br>The camera may be in use by the Recorder app.', true);
        if (!retryTimer) {{ retryTimer = setTimeout(start, 8000); }}
      }});

      retry.addEventListener('click', start);
      window.__liveRestart = start;
      start();
    }})();
  </script>

  <script>
    (function () {{
      var pbtn = document.getElementById('photo-btn');
      var pmsg = document.getElementById('photo-msg');
      pbtn.addEventListener('click', function () {{
        pbtn.disabled = true;
        pmsg.textContent = 'Snapping…';
        fetch('/photo', {{ method: 'POST' }}).then(function (r) {{
          return r.json();
        }}).then(function (j) {{
          pmsg.textContent = j.message || (j.ok ? 'Saved.' : 'Failed.');
          if (j.ok) {{ setTimeout(function () {{ location.reload(); }}, 900); }}
          else {{ pbtn.disabled = false; }}
        }}).catch(function () {{
          pmsg.textContent = 'Request failed.';
          pbtn.disabled = false;
        }});
      }});

      var calBtn = document.getElementById('cal-btn');
      var calInfo = document.getElementById('cal-info');
      var calPoll = null;
      function calCheck() {{
        fetch('/calibrate/status').then(function (r) {{ return r.json(); }}).then(function (s) {{
          if (s.running) {{ calInfo.textContent = 'Calibrating… ' + (s.message || ''); return; }}
          if (calPoll) {{ clearInterval(calPoll); calPoll = null; }}
          calBtn.disabled = false;
          calInfo.textContent = (s.ok ? '✅ ' : (s.ok === false ? '❌ ' : '')) + (s.message || '');
        }}).catch(function () {{}});
      }}
      calBtn.addEventListener('click', function () {{
        calBtn.disabled = true;
        calInfo.textContent = 'Starting calibration…';
        fetch('/calibrate', {{ method: 'POST' }}).then(function (r) {{ return r.json(); }}).then(function (j) {{
          if (!j.ok) {{ calInfo.textContent = j.message || 'Could not start.'; calBtn.disabled = false; return; }}
          calPoll = setInterval(calCheck, 2000);
        }}).catch(function () {{ calInfo.textContent = 'Request failed.'; calBtn.disabled = false; }});
      }});

      document.getElementById('photo-list').addEventListener('click', function (e) {{
        var del = e.target.closest('.photo-del');
        if (!del) return;
        var name = del.getAttribute('data-name');
        if (!confirm('Delete ' + name + '?')) return;
        del.disabled = true;
        fetch('/photo/delete', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
          body: 'name=' + encodeURIComponent(name)
        }}).then(function (r) {{ return r.json(); }}).then(function (j) {{
          if (j.ok) {{ del.closest('.photo-card').remove(); }}
          else {{ alert(j.message || 'Delete failed.'); del.disabled = false; }}
        }}).catch(function () {{ alert('Request failed.'); del.disabled = false; }});
      }});
    }})();
  </script>

  <script>
    (function () {{
      var btn = document.getElementById('rec-btn');
      var cancelBtn = document.getElementById('rec-cancel');
      var secs = document.getElementById('rec-seconds');
      var box = document.getElementById('rec-status');
      var text = document.getElementById('rec-text');
      var liveSection = document.querySelector('.wrap.live');
      var previewStage = document.getElementById('rec-preview-stage');
      var previewImg = document.getElementById('rec-preview');
      var tick = null, poll = null, refresh = null, startMs = 0, duration = 0, cancelling = false;

      function fmt(s) {{
        s = Math.max(0, Math.round(s));
        var m = Math.floor(s / 60), r = s % 60;
        return m + ':' + (r < 10 ? '0' : '') + r;
      }}

      function render() {{
        var elapsed = (Date.now() - startMs) / 1000;
        var remaining = Math.max(0, duration - elapsed);
        text.textContent = fmt(remaining) + ' left / ' + fmt(elapsed) + ' elapsed';
      }}

      function finish(message) {{
        if (tick) {{ clearInterval(tick); tick = null; }}
        if (poll) {{ clearInterval(poll); poll = null; }}
        if (refresh) {{ clearInterval(refresh); refresh = null; }}
        text.textContent = message;
        cancelBtn.style.display = 'none';
        btn.disabled = false;
        secs.disabled = false;
        setTimeout(function () {{ location.reload(); }}, 1800);
      }}

      function checkStatus() {{
        fetch('/record/status').then(function (r) {{ return r.json(); }}).then(function (s) {{
          if (!s.running) {{ finish(cancelling ? 'Cancelled.' : 'Done - saved. Refreshing...'); }}
        }}).catch(function () {{}});
      }}

      btn.addEventListener('click', function () {{
        var d = parseInt(secs.value, 10);
        if (!d || d < 1) {{ d = 30; }}
        cancelling = false;
        btn.disabled = true;
        secs.disabled = true;
        box.classList.remove('hidden');
        cancelBtn.style.display = '';
        cancelBtn.disabled = false;
        text.textContent = 'Starting...';
        fetch('/record', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
          body: 'duration=' + d
        }}).then(function (r) {{
          return r.json().then(function (j) {{ return {{ ok: r.ok, body: j }}; }});
        }}).then(function (res) {{
          if (!res.ok || !res.body.ok) {{ finish(res.body.message || 'Could not start recording.'); return; }}
          duration = res.body.duration;
          startMs = Date.now();
          render();
          tick = setInterval(render, 250);
          poll = setInterval(checkStatus, 1500);
          // the live MJPEG view can't run while recording (camera is busy), so
          // hide it and show the recorder's preview frame, refreshed a few times/sec
          if (liveSection) {{ liveSection.style.display = 'none'; }}
          previewStage.style.display = '';
          refresh = setInterval(function () {{
            previewImg.src = '/preview.jpg?t=' + Date.now();
          }}, 300);
        }}).catch(function () {{ finish('Could not start recording.'); }});
      }});

      cancelBtn.addEventListener('click', function () {{
        cancelling = true;
        cancelBtn.disabled = true;
        text.textContent = 'Cancelling...';
        fetch('/record/cancel', {{ method: 'POST' }}).catch(function () {{}});
      }});
    }})();
  </script>
</body>
</html>"""
        self.send_bytes(body.encode("utf-8"))

    def serve_preview(self):
        """Serve the latest recording preview frame written by the recorder."""
        try:
            data = Path(PREVIEW_PATH).read_bytes()
        except OSError:
            self.send_bytes(b"", "image/jpeg", 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except OSError:
            pass

    def serve_recordings(self):
        """JSON listing of every recorded video, newest first, with a download
        path — used by the laptop control panel's "Save All" button."""
        videos = []
        seen_meta = set()
        for path in video_files():
            try:
                size = path.stat().st_size
                mtime = path.stat().st_mtime
            except OSError:
                continue
            token = video_token(path)
            videos.append({
                "token": token,
                "label": video_label(path),
                "size": size,
                "mtime": round(mtime, 3),
                "download": "/download/" + urllib.parse.quote(token),
            })
            # Also list the per-frame timestamps CSV(s) sitting next to the video
            # in streams/, so Save All beams real frame timing to the laptop too.
            for extra in sorted(path.parent.glob("*_timestamps.csv")):
                ekey = extra.resolve()
                if ekey in seen_meta:
                    continue
                seen_meta.add(ekey)
                try:
                    esize, emtime = extra.stat().st_size, extra.stat().st_mtime
                except OSError:
                    continue
                etoken = video_token(extra)
                videos.append({
                    "token": etoken,
                    "label": video_label(extra),
                    "size": esize,
                    "mtime": round(emtime, 3),
                    "download": "/download/" + urllib.parse.quote(etoken),
                })
            # Also list this recording's metadata.json so Save All beams it too and
            # the laptop ends up with the real_dataset layout (metadata.json at the
            # rec root, next to streams/). One per rec — dedup across its videos.
            rec = recording_dir(path)
            if rec is not None:
                meta = rec / "metadata.json"
                mkey = meta.resolve()
                if meta.is_file() and mkey not in seen_meta:
                    seen_meta.add(mkey)
                    try:
                        msize, mmtime = meta.stat().st_size, meta.stat().st_mtime
                    except OSError:
                        continue
                    mtoken = video_token(meta)
                    videos.append({
                        "token": mtoken,
                        "label": video_label(meta),
                        "size": msize,
                        "mtime": round(mmtime, 3),
                        "download": "/download/" + urllib.parse.quote(mtoken),
                    })
        self.send_bytes(json.dumps({"videos": videos}).encode("utf-8"),
                        "application/json; charset=utf-8")

    def serve_dataset(self, encoded_name):
        rec_dir = resolve_dataset_path(encoded_name)
        if rec_dir is None:
            self.send_bytes(b"Not found", "text/plain; charset=utf-8", 404)
            return
        self.serve_zip(rec_dir)

    def serve_day(self, encoded_name):
        day_dir = resolve_day_path(encoded_name)
        if day_dir is None:
            self.send_bytes(b"Not found", "text/plain; charset=utf-8", 404)
            return
        self.serve_zip(day_dir)

    def serve_zip(self, target_dir):
        """Stream a zip of target_dir, nested under its own name (e.g. day_*/...)."""
        files = sorted(p for p in target_dir.rglob("*") if p.is_file())
        tmp = tempfile.NamedTemporaryFile(prefix="smartroom_", suffix=".zip", delete=False)
        try:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
                for file_path in files:
                    arcname = file_path.relative_to(target_dir.parent)
                    archive.write(file_path, arcname.as_posix())
            tmp.close()
            size = os.path.getsize(tmp.name)
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", f'attachment; filename="{target_dir.name}.zip"')
            self.end_headers()
            with open(tmp.name, "rb") as src:
                self.copy_file(src)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def serve_video(self, encoded_name, download):
        path = resolve_video_path(encoded_name)
        if path is None:
            self.send_bytes(b"Not found", "text/plain; charset=utf-8", 404)
            return

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        headers = {}
        if download:
            headers["Content-Disposition"] = f'attachment; filename="{path.name}"'

        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        if not range_header or not range_header.startswith("bytes="):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            with path.open("rb") as src:
                self.copy_file(src)
            return

        start_text, _, end_text = range_header.removeprefix("bytes=").partition("-")
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return

        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Accept-Ranges", "bytes")
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        with path.open("rb") as src:
            src.seek(start)
            remaining = length
            while remaining > 0:
                chunk = src.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except OSError:
                    break
                remaining -= len(chunk)

    def serve_photo(self, encoded_name):
        # Photos live flat in PHOTOS_DIR; the name check rejects any traversal.
        name = urllib.parse.unquote(encoded_name)
        path = (PHOTOS_DIR / name).resolve()
        if (not name or "/" in name or not path.name.endswith(".jpg")
                or path.parent != PHOTOS_DIR.resolve() or not path.is_file()):
            self.send_bytes(b"Not found", "text/plain; charset=utf-8", 404)
            return
        self.send_bytes(path.read_bytes(), "image/jpeg")

    def copy_file(self, src):
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
            except OSError:
                break


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Smartroom video page running at http://smartroom.local:{PORT}")
    print(f"Direct IP: http://{local_ip()}:{PORT}")
    print(f"Serving: {RECORDINGS_DIR}")
    print(f"Live camera: {CAMERA_DEVICE} @ {CAMERA_SIZE}")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
