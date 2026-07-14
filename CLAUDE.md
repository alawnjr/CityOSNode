# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multi-camera capture system for a "smart room" research node — cameras placed
in different parts of a room, recording into structured folders for downstream
analysis (e.g. occupancy detection). **Video-only for now**: audio and the custom
I²C/PCB sensors were removed from the capture pipeline (their `test/` scripts are
kept — see below).

## Critical: code runs on the Raspberry Pis, not the dev machine

**As of 2026-07-13 there is ONE active camera node** (`smartroom1` — a Pi 3 with
a Logitech Webcam Pro 9000 pinned to 800×600 — is decommissioned; two-node
instructions below are historical):

| Host | Board | Cameras |
|---|---|---|
| `smartroom2.local` | Raspberry Pi 4 | Logitech C920 (1280×720@30, main capture) + Intel RealSense D455 and D435 (depth, port-8001 page) |

User is `smartroom` on both; credentials in `PRIVATE.md` (gitignored). The cameras
are accessed via `ffmpeg -f v4l2`, so capture itself needs no special libraries —
but the cameras differ per node, so the device/format are **auto-detected and
env-overridable** (see `capture.py` below) rather than hardcoded.

### Sync workflow — push through git, never edit code on the Pis

**Do NOT edit code on a Pi directly, and do NOT `scp`/copy files onto it.** All
code changes flow through git so the checkouts never diverge:

1. On the dev machine: make changes, commit, and push to **both** remotes.
2. On **each** Pi: `git pull origin master`, then run.

Both Pis (and this repo) have the **same two remotes** (names are NOT swapped):
- `origin`   → `github.com:alawnjr/CityOSNode.git` (the Pis pull from this one)
- `personal` → `gitlab.orbit-lab.org:alawnjr/smartroom.git`

Always push to **both** so GitHub and the GitLab mirror stay in sync:

```bash
# Dev machine (this repo): push the same commit to both remotes
git push origin master
git push personal master

# Each Pi (~/CityOS): pulls from origin (GitHub)
ssh smartroom@smartroom1.local 'cd ~/CityOS && git pull origin master'
ssh smartroom@smartroom2.local 'cd ~/CityOS && git pull origin master'
```

On a Pi you may freely **read and run** code (and run read-only git commands), but
never modify tracked files there — editing happens only on the dev machine and
arrives via `git pull`. SSH key auth is already configured (no password needed).
If `.local` mDNS resolution is flaky, SSH by IP instead.

Run capture with the venv Python (`~/CityOS/.venv/bin/python`), never system
python:

```bash
ssh smartroom@smartroom2.local '~/CityOS/.venv/bin/python ~/CityOS/capture.py'
```

There is **no build/lint/test suite** — these are standalone hardware scripts
verified by running them against the live cameras.

Install dependencies (on each Pi): `pip install -r requirements.txt`. Camera
capture also needs the system tools `ffmpeg` and `v4l-utils` (apt). Fresh nodes
can be bootstrapped with `setup_pi.sh`.

## Hardware and how the camera is accessed

| Node | Camera | Access |
|---|---|---|
| `smartroom1` (Pi 3) | Logitech Webcam Pro 9000 | ffmpeg `-f v4l2`, MJPG; **pinned to 800×600** — the only mode where it sustains ~27–30fps (its full 1280×800 only delivers ~21fps) |
| `smartroom2` (Pi 4) | Logitech (C920-class) USB camera | ffmpeg `-f v4l2`, MJPG 1280×720@30; wide 16:9 FOV (4:3 modes are center-cropped/narrower) |

The camera device is **auto-detected** at runtime (first `/dev/v4l/by-id/*-video-index0`
symlink), so the one shared codebase works on both nodes despite the different
cameras. Override per node with env vars: `SMARTROOM_CAMERA` (device),
`SMARTROOM_CAMERA_SIZE` (e.g. `1280x720`), `SMARTROOM_CAMERA_FPS`.

**Per-node overrides live in `node.env`** (gitignored, at the repo root on the
Pi) — `KEY=VALUE` lines loaded at startup by both `capture.py` and the web page;
the real environment wins over the file. This is how smartroom1 pins
`SMARTROOM_CAMERA_SIZE=800x600` without diverging the shared code. Recordings
are expected to be ~30fps — the laptop-side validator flags clips outside
27–33fps, so when swapping a camera, measure which mode actually *delivers*
30fps (cheap UVC cams often under-deliver at their max resolution, and drop
further in low light) and pin it in `node.env`.

**Audio and the I²C/PCB sensors were removed from the capture pipeline** (video
only for now). Their drivers/wiring still exist on the boards and the per-device
`test/` scripts are retained, so re-adding any of them later is straightforward.

## capture.py — the main pipeline

Records video for `DURATION_SECONDS` (default 30, set with `--duration`/`-d`) into
a recording folder mirroring `sample_dataset/`:

```
data/day_NN_YYYY-MM-DD/rec_YYYYMMDD_NNN/
  metadata.json          # recording_id, node (hostname), streams{}
  streams/
    camera_main.mp4 + camera_main_timestamps.csv
```

It's a single blocking `ffmpeg -f v4l2` call for the full duration, then it writes
the per-frame timestamps and `metadata.json`. **Depth cameras are recorded too**:
capture.py asks the RealSense page (port 8001, `POST /record/start`) to record
every connected depth camera into the same `streams/` folder — color as
`camera_d4xx_color.mp4` (h264, Pi hardware encoder) and depth as
`camera_d4xx_depth.mkv` (**lossless FFV1 gray16le, raw z16 units ×
`depth_scale_m` = meters**, aligned to color), each with a real timestamps
CSV, merged into `metadata.json`'s `streams{}` with factory intrinsics +
room-frame extrinsics. Recording runs at the camera's pipeline rate
(**both cameras 640x480@30** — `SMARTROOM_DEPTH_PROFILE_D4XX` in node.env
overrides): depth is captured RAW (to /dev/shm, or the SD card for long
clips) and FFV1-encoded *after* the recording ends (~1-2x the clip length;
`/record/status` stays running until done), and both containers are timed to
the measured rate so playback matches wall clock. **Frame sync**: every
timestamps CSV has a `hw_timestamp_ms` column — librealsense global time
(sensor mid-exposure mapped to the host clock, ms since epoch) — match
frames across cameras on it (~1-2ms; the sensors free-run, so pair nearest
frames, up to ±17ms at 30fps) and use each stream's embedded room-frame
extrinsics to fuse 3D points into the one tag-1 frame. If the page is
down or no depth camera is plugged in, recordings are webcam-only as before.
**On smartroom2 the webcam is EXCLUDED from recordings** (`SMARTROOM_SKIP_WEBCAM=1`
in node.env): its mjpeg-decode + overlay + encode pipeline cost the D455 its
30fps, so recordings are depth-cameras-only (`camera_d455_color.mp4` is the RGB
record; no `camera_main`) — the C920 still serves the live view. Unset the flag
to bring the webcam back into recordings. **The timestamps CSV is real, not
synthetic**: after recording, `probe_frame_times()` ffprobes the finished mp4 for
each frame's actual presentation time (the USB cams are variable-rate, so the
nominal-fps grid would be wrong), one CSV row per actual encoded frame;
`metadata.json`'s `frame_count` matches the video. `metadata.json` records `node`
(`socket.gethostname()`) so recordings from `smartroom1` vs `smartroom2` are
distinguishable when merged, and **embeds the camera's calibration** (see below)
when one exists. Folder/recording numbers auto-increment — note the two Pis'
counters must stay in step for the laptop dashboard to pair same-session
recordings (failed/aborted attempts still consume a number).

When `SMARTROOM_PREVIEW` is set in the environment, the camera ffmpeg gains a
second output writing the latest frame to a jpg — the web UI uses this to show the
camera while recording. Without it, behaviour is unchanged.

Cross-node sync: there's no shared clock between the two Pis, so align recordings
using the **clap at t=0** marker in `test/scenarios.md` (keep the Pis' clocks close
with NTP).

## Camera calibration (intrinsics)

`calibrate_camera.py` computes checkerboard intrinsics (camera matrix, distortion)
**on the Pi**, with the venv python. Two modes; photos mode is the recommended flow
(you can frame the board on the live view, and the camera is never opened):

```bash
# 1. On the node's web page: Snap photo of the checkerboard at 10+ positions
#    (center, corners, near, far, tilted), then press "Calibrate from photos" —
#    or run it by hand:
~/CityOS/.venv/bin/python ~/CityOS/calibrate_camera.py --photos   # from data/photos/*.jpg
~/CityOS/.venv/bin/python ~/CityOS/calibrate_camera.py            # live capture (camera must be free)
```

Board: the **DFvision Q18-100-4.5 glass plate** by default — 18×18 squares on
100×100mm, 4.5mm squares → **17×17 inner corners**. It's small: hold it
~15–40cm from the lens. Other boards via `--cols/--rows/--square-mm` (the
printed paper board is `--cols 9 --rows 6 --square-mm 25`).
Output: `calibration/<usb-serial>.json` (gitignored — machine-generated on the Pi,
like `data/`), **keyed by the camera's USB serial** so a swapped camera never
inherits stale intrinsics. Also writes corner-overlay debug JPGs and a
`before.jpg`/`after.jpg`/`compare.jpg` (same frame raw vs undistorted) under
`calibration/debug/<camera-id>/`. RMS < 1.0 px is good.

`capture.py` embeds the values into every recording's `metadata.json`
(`streams.camera_main.calibration`, resolution-scaled if needed). Videos stay
**raw** — the laptop undistorts downstream. Requires `opencv-python-headless`
(in `requirements.txt`).

## smartroom_video_page.py — web UI

A stdlib `http.server` (`ThreadingHTTPServer`) serving `http://<node>.local:8000`
(runs independently on each node — `smartroom1`/`smartroom2`), via the
`smartroom-video-page.service` systemd unit (system `python3`, working dir
`~/CityOS`). It auto-detects the node's camera the same way `capture.py` does. It
lives in this repo and is synced via GitHub like everything else — **after editing,
push and `git pull` on each Pi, then `sudo systemctl restart
smartroom-video-page.service`** there. (Without sudo: `fuser -k -TERM 8000/tcp`,
then relaunch with `nohup python3 ~/CityOS/smartroom_video_page.py &`. Never
`pkill -f` the script name over ssh — the pattern matches your own remote shell
and kills the connection; also note systemd treats that SIGTERM as a clean exit,
so `Restart=on-failure` will NOT respawn it.) Routes:
- `/` — live MJPEG view (`/stream.mjpg`), a Record panel, Photos (snap/calibrate),
  and a list of recordings.
- `POST /record` (duration) → runs `run_smartroom_capture.sh`; `POST /record/cancel`
  kills the recording's process group; `/record/status` returns countdown/elapsed JSON.
- `POST /photo` — full-resolution still to `data/photos/` (borrows the camera from
  the preview with a busy-retry); `/photo/<name>` serves it; `POST /photo/delete`
  removes one. Photos are page-only (not in the `/recordings` listing).
- `POST /calibrate` + `/calibrate/status` — runs `calibrate_camera.py --photos` in
  the background and reports the RMS/result.
- `/recordings` — JSON listing consumed by the laptop's Save All; includes each
  clip's `metadata.json` **and `*_timestamps.csv`** so frame timing reaches the laptop.
- `/dataset/<rec>` zips a whole recording folder; `/video/`, `/download/` serve single files.

The camera is **single-access**, so the page releases its own preview ffmpeg
before a recording starts and instead relays the recorder's `SMARTROOM_PREVIEW`
feed, then resumes its own camera capture when the recording ends.

## realsense_depth_page.py — D455 depth/RGB view (port 8001)

A second stdlib web page for the Intel RealSense cameras on `smartroom2`
(D455 + D435, one section per connected device, keyed by USB serial) — live RGB
and colorized depth side by side (depth aligned to color), click any pixel to
read its distance in meters. Runs with the **venv** python via the
`smartroom-depth-page.service` systemd unit (needs `pyrealsense2`, which has no
aarch64 wheel — built from source on the Pi by `setup_realsense_pi.sh`, ~1–2h
compile with the RSUSB backend). The RealSense cameras are separate USB devices
from the node's main webcam, so this page and `smartroom_video_page.py` (port
8000) run side by side without conflict; the port-8000 page also **embeds this
page's cameras** (its JSON endpoints send CORS headers for that). RealSense
cameras must be in the **blue USB 3 ports with USB 3 cables** (on USB 2 they
only reach reduced profiles, and two RealSense cannot share the USB 2 bus; the
pages show the negotiated USB speed). **One worker subprocess per camera**
(multiprocessing spawn): two 30fps pipelines in one Python process starve each
other on the GIL — the HTTP front end supervises workers over command pipes; a
watchdog respawns dead workers and exits the page (systemd respawns it) when
the plugged-camera set changes. Gotchas learned the hard way: enumerate ONCE
before any worker/pipeline exists (a context probing the bus while another
process holds the devices enumerates empty); per-serial 180° flips via
`SMARTROOM_DEPTH_FLIP` in node.env are applied OFF the capture path (ffmpeg /
view encoder / coordinate transform — in-loop rotation cost the flipped
camera 5fps).

`realsense_extrinsics.py` — AprilTag extrinsic calibration for a RealSense
camera using its **factory intrinsics** (no checkerboard needed) plus a
depth-vs-PnP cross-check; same tag (36h11 id 1) and same
`calibration/<serial>.extrinsics.json` schema as `calibrate_extrinsics.py`.
Run from the "Calibrate extrinsic" button on either web page (executes
in-process on live frames) or standalone with the venv python. **Tag
chaining**: other 36h11 tags seen in the same frame as tag 1 get their
room-frame pose saved to `calibration/tags.json`, and `capture.py` embeds
that map into every recording's `metadata.json` (top-level `room_tags`).

## `sample_dataset/` is the schema reference

`run_smartroom_capture.py` is a thin wrapper that calls `capture.py` (kept for the
`.sh`/desktop launchers). `sample_dataset/` is the canonical reference for the
`metadata.json` schema — match its `streams{}` structure when changing capture
output. (Older flat recordings under `data/`, `env_sensor_01.csv` etc., came from a
previous generation of the pipeline.)

## test/ directory

Standalone per-device scripts, run individually on a Pi. **These are retained for
all sensors — including audio (MCP3008) and the I²C/PCB chips — even though those
are no longer in the capture pipeline**, so any of them can be re-added later.
- `test/<device>/read.py` — live continuous monitor for that sensor (Ctrl-C to stop).
- `test/<device>/*_scanner.py` — one-off wiring/channel discovery tools.
- `test/camera/record_camera.py` — grab a single camera snapshot.
- `test/scenarios.md` — **human acting scripts** to perform in front of the node
  for realistic occupancy data; each opens with a clap at t=0 (`clap_at_t0`) as the
  cross-stream / cross-node sync marker.

## The laptop counterpart: smartroom-control

The analysis side lives in a **separate repo** on the dev machine,
`~/Code/smartroom-control` (Next.js dashboard on `localhost:4000`, its own git —
single `origin` on GitHub, commits stay local unless asked to push). It pulls
recordings from both Pis via the pages' `/recordings` listings ("Save All"),
merges the two cameras into one session tree keyed by matching `day/rec` names,
validates data integrity, writes undistorted copies of calibrated clips, runs
YOLO/RTMPose/action-recognition analyses, and serves a read-only LAN API
(`/api/v1`, documented in its `API.md`). When changing this repo's recording
layout, `metadata.json` schema, or the `/recordings` listing, check that repo's
`app/api/save-all/`, `lib/detections.ts`, and `detect/` for the consuming side.

## Conventions

- `data/` (recordings + `data/photos/` snaps), `calibration/`, `node.env`,
  `*.mp4`, `*.wav`, `*.pdf`, and `PRIVATE.md` are gitignored — recordings and
  machine-generated per-node state are not version-controlled.
- Keep raw sensor values raw — capture writes uncorrected readings; any
  post-processing happens downstream, not baked into the CSVs.
- The camera device/format are auto-detected with `SMARTROOM_CAMERA*` env
  overrides (the one shared codebase runs on both differently-cameraed nodes).
  Other one-off `test/` scripts may still hardcode their device path.
