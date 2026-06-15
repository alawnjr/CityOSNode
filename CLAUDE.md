# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multi-camera capture system for a "smart room" research node — cameras placed
in different parts of a room, recording into structured folders for downstream
analysis (e.g. occupancy detection). **Video-only for now**: audio and the custom
I²C/PCB sensors were removed from the capture pipeline (their `test/` scripts are
kept — see below).

## Critical: code runs on the Raspberry Pis, not the dev machine

There are **two camera nodes**, each a Pi running the same checkout of this repo:

| Host | Board | Camera |
|---|---|---|
| `smartroom1.local` | Raspberry Pi 3 | generic ("random Chinese") USB camera |
| `smartroom2.local` | Raspberry Pi 4 | Logitech (C920-class) USB camera |

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
| `smartroom1` (Pi 3) | generic USB camera | ffmpeg `-f v4l2`, MJPG; device auto-detected |
| `smartroom2` (Pi 4) | Logitech (C920-class) USB camera | ffmpeg `-f v4l2`, MJPG 1280×720@30; wide 16:9 FOV (4:3 modes are center-cropped/narrower) |

The camera device is **auto-detected** at runtime (first `/dev/v4l/by-id/*-video-index0`
symlink), so the one shared codebase works on both nodes despite the different
cameras. Override per node with env vars: `SMARTROOM_CAMERA` (device),
`SMARTROOM_CAMERA_SIZE` (e.g. `1280x720`), `SMARTROOM_CAMERA_FPS`. Set
`SMARTROOM_CAMERA_SIZE` on a node whose camera doesn't support the 1280×720 default.

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
the per-frame timestamps and `metadata.json`. `metadata.json` records `node`
(`socket.gethostname()`) so recordings from `smartroom1` vs `smartroom2` are
distinguishable when merged. Folder/recording numbers auto-increment.

When `SMARTROOM_PREVIEW` is set in the environment, the camera ffmpeg gains a
second output writing the latest frame to a jpg — the web UI uses this to show the
camera while recording. Without it, behaviour is unchanged.

Cross-node sync: there's no shared clock between the two Pis, so align recordings
using the **clap at t=0** marker in `test/scenarios.md` (keep the Pis' clocks close
with NTP).

## smartroom_video_page.py — web UI

A stdlib `http.server` (`ThreadingHTTPServer`) serving `http://<node>.local:8000`
(runs independently on each node — `smartroom1`/`smartroom2`), via the
`smartroom-video-page.service` systemd unit (system `python3`, working dir
`~/CityOS`). It auto-detects the node's camera the same way `capture.py` does. It
lives in this repo and is synced via GitHub like everything else — **after editing,
push and `git pull` on each Pi, then `sudo systemctl restart
smartroom-video-page.service`** there. Routes:
- `/` — live MJPEG view (`/stream.mjpg`), a Record panel, and a list of recordings.
- `POST /record` (duration) → runs `run_smartroom_capture.sh`; `POST /record/cancel`
  kills the recording's process group; `/record/status` returns countdown/elapsed JSON.
- `/dataset/<rec>` zips a whole recording folder; `/video/`, `/download/` serve single files.

The camera is **single-access**, so the page releases its own preview ffmpeg
before a recording starts and instead relays the recorder's `SMARTROOM_PREVIEW`
feed, then resumes its own camera capture when the recording ends.

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

## Conventions

- `data/`, `*.mp4`, `*.wav`, and `PRIVATE.md` are gitignored — recordings are not
  version-controlled.
- Keep raw sensor values raw — capture writes uncorrected readings; any
  post-processing happens downstream, not baked into the CSVs.
- The camera device/format are auto-detected with `SMARTROOM_CAMERA*` env
  overrides (the one shared codebase runs on both differently-cameraed nodes).
  Other one-off `test/` scripts may still hardcode their device path.
