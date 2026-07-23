# Smartroom Node API

Every camera node runs two small stdlib web servers. This page documents their
HTTP endpoints. Both are plain `http.server` apps — no framework, no auth (LAN
only).

| Server | Port | What it serves |
|---|---|---|
| `smartroom_video_page.py` | **8000** | Main USB webcam: live view, recording control, calibration, recordings listing |
| `realsense_depth_page.py` | **8001** | Intel RealSense depth/RGB view + depth recording |

The laptop control panel (`smartroom-control`) consumes these over the LAN. The
port-8000 page embeds the port-8001 cameras, so the depth page's JSON endpoints
send permissive CORS headers.

---

## Port 8000 — video page

### GET

- **`/`** — the HTML dashboard (Live view, Record, Photos, Recordings, and this
  Docs tab).
- **`/stream.mjpg`** — live MJPEG feed from the webcam (`multipart/x-mixed-replace`).
- **`/preview.jpg`** — the latest single frame; used while a recording has
  borrowed the camera.
- **`/recordings`** — JSON inventory of every recorded file, newest first. Used
  by the laptop's *Save All*. See the schema below.
- **`/record/status`** — countdown / elapsed JSON for an in-progress recording.
- **`/calibrate/status`**, **`/calibrate/extrinsic/status`** — background
  calibration job state (RMS / result).
- **`/video/<token>`** — serve a file for **inline playback** (no attachment
  header). Honors HTTP `Range`, so the browser can scrub.
- **`/download/<token>`** — serve the same file as an **attachment** (saves to
  disk). Also `Range`-aware.
- **`/photo/<name>`** — serve a snapped still.
- **`/dataset/<rec>`** — zip a whole recording folder.
- **`/day/<day>`** — zip a whole day folder.

### POST

- **`/record`** — start a recording (`duration` seconds, clamped 1–3600). Runs
  `run_smartroom_capture.sh`.
- **`/record/cancel`** — kill the recording's process group.
- **`/photo`** — capture a full-resolution still into `data/photos/`.
- **`/photo/delete`** — remove one photo.
- **`/calibrate`** — run `calibrate_camera.py --photos` (checkerboard intrinsics).
- **`/calibrate/extrinsic`** — run AprilTag extrinsic (camera-pose) calibration.

### `/recordings` response

```json
{
  "videos": [
    {
      "token": "data/day_01_2026-07-23/rec_20260723_001/streams/camera_d455_color.mp4",
      "label": "day_01_2026-07-23/rec_20260723_001/streams/camera_d455_color.mp4",
      "size": 12345678,
      "mtime": 1753300000.123,
      "duration_s": 30.033,
      "download": "/download/data%2F..."
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `token` | Opaque path handle — pass it to `/video/` or `/download/`. |
| `label` | Human-readable relative path. |
| `size` | File size in bytes. |
| `mtime` | File modification time (epoch seconds). |
| `duration_s` | Recorded clip length in seconds, from the sibling `*_timestamps.csv` (last frame's presentation time). `null` when no CSV exists. Present on video rows only. |
| `download` | Ready-made attachment URL. |

Alongside each video, the listing also emits rows for that recording's
`*_timestamps.csv` files and its `metadata.json`, so *Save All* reconstructs the
full dataset layout on the laptop.

---

## Port 8001 — RealSense depth page

### GET

- **`/`** — HTML: live RGB + colorized depth, side by side, per connected camera.
- **`/devices`** — JSON list of connected RealSense cameras (keyed by USB serial).
- **`/rgb.mjpg?s=<serial>`**, **`/depth.mjpg?s=<serial>`** — per-camera live feeds.
- **`/value`** — distance in meters for a clicked pixel.
- **`/record/status`**, **`/calibrate/extrinsic/status`**,
  **`/calibrate/timing/status`** — job state.

### POST

- **`/record/start`** — called by `capture.py` to record every connected depth
  camera into the same `streams/` folder (color `.mp4` + lossless FFV1 depth
  `.mkv`).
- **`/calibrate/extrinsic`** — AprilTag extrinsic solve on live frames.
- **`/calibrate/timing`** — lights-toggle clock-offset calibration between the
  two depth cameras.

---

## Deploying a change

These pages run on the Pi via systemd. After editing on the dev machine:

```bash
git push origin master && git push personal master
ssh smartroom@smartroom2.local 'cd ~/CityOS && git pull origin master'
ssh smartroom@smartroom2.local 'sudo systemctl restart smartroom-video-page.service'
```
