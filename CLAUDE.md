# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multimodal sensor-capture system for a "smart room" research node. It records
synchronized camera, audio, radar, and environmental/motion data into structured
recording folders for downstream analysis (e.g. occupancy detection).

## Critical: code runs on the Raspberry Pi, not the dev machine

The sensors are physically wired to a **Raspberry Pi 5** (`smartroom.local`, user
`smartroom` — credentials in `PRIVATE.md`, gitignored). The Adafruit Blinka /
CircuitPython libraries (`board`, `busio`, `adafruit_*`) only function there
against real GPIO/SPI/I²C — they will not import or run on a normal dev machine.

### Sync workflow — GitHub only, never edit code on the Pi

**Do NOT edit code on the Pi directly, and do NOT `scp`/copy files onto it.** All
code changes flow through GitHub so the two checkouts never diverge:

1. On the dev machine: make changes, commit, and push to GitHub.
2. On the Pi: `git pull`, then run.

GitHub is `github.com:alawnjr/CityOSNode.git`. The remote names are **swapped**
between the machines:

```bash
# Dev machine (this repo): GitHub is the "personal" remote
git push personal master

# Pi (~/CityOS): GitHub is the "origin" remote
ssh smartroom@smartroom.local 'cd ~/CityOS && git pull origin master'
```

On the Pi you may freely **read and run** code (and run read-only git commands),
but never modify tracked files there — editing happens only on the dev machine and
arrives via `git pull`. SSH key auth is already configured (no password needed).

Run sensor code with the venv Python (`~/CityOS/.venv/bin/python`), never system
python:

```bash
ssh smartroom@smartroom.local '~/CityOS/.venv/bin/python ~/CityOS/capture.py'
```

There is **no build/lint/test suite** — these are standalone hardware scripts
verified by running them against the live sensors.

Install dependencies (on the Pi): `pip install -r requirements.txt`. Camera
capture also needs the system tools `ffmpeg` and `v4l-utils` (apt).

## Hardware and how each sensor is accessed

| Sensor | Access | Notes |
|---|---|---|
| USB camera (lihappe8) | ffmpeg `-f v4l2`, device `/dev/v4l/by-id/usb-lihappe8_...index0`, MJPG 640×480@30 | also has a built-in mic |
| Camera mic | `arecord -D plughw:CARD=Camera,DEV=0`, 48 kHz stereo | addressed by card **name** so it survives card-number reordering |
| OPS243-C radar | pyserial `/dev/ttyACM0` @ **19200**, send `OD` to enable output | factory-calibrated; speed+distance FMCW. Dirs/comments say "OPS243-A" but the hardware is a **-C** |
| MCP3008 ADC (SPI, CE0=`board.D8`) | `adafruit_mcp3xxx`, mic (MAX9814) on channel P0 | **level sensing only**, not real audio; MAX9814 has hardware AGC |
| MAESTRO 2.1 I²C board | `board.I2C()` + `adafruit_*` | BME680 `0x76`, TCS34725 `0x29`, ADXL345 `0x53`, MLX90393 `0x0C` |

The PIR motion sensor was removed; `test/GPIO/` scripts are legacy wiring-discovery tools.

## capture.py — the main pipeline

Records all sensors concurrently for `DURATION_SECONDS` (hardcoded 30, no CLI
flags by design) into a recording folder mirroring `sample_dataset/`:

```
data/day_NN_YYYY-MM-DD/rec_YYYYMMDD_NNN/
  metadata.json          # recording_id, streams{}, calibration block
  streams/
    camera_main.mp4 + camera_main_timestamps.csv
    mic_array.wav          # camera mic (real audio)
    mcp3008_mic.csv        # mic level/voltage (separate from mic_array.wav)
    custom_board_i2c.csv   # all 4 i2c chips, one row/sec
    radar_ops243.csv       # raw serial lines
```

Concurrency model: the camera runs via a blocking `ffmpeg` call for the full
duration; `arecord` runs as a parallel subprocess; the i2c/mic/radar loggers each
run in a thread writing CSV against a **shared `time.monotonic()` clock** (`start`),
and a `stop_event` ends them when the camera finishes. Every sensor's init is
wrapped in try/except so missing hardware prints `<sensor>: unavailable` and the
rest of the capture still proceeds. Folder/recording numbers auto-increment.

## Two generations of capture (important context)

- `run_smartroom_capture.py` lives **only on the Pi** (not in this repo) — the
  original full pipeline with CLI flags. It wrote the **flat** layout
  `data/<day>/smartroom_<timestamp>/` (note `env_sensor_01.csv`,
  `radar_sensor_01.csv` names). The existing recordings under `data/` came from it.
- `capture.py` (this repo) is the simpler reimplementation using the **nested**
  `sample_dataset` layout above. New recordings use this layout.

`sample_dataset/` is the canonical reference for the `metadata.json` schema — match
its `streams{}` structure when changing capture output.

## test/ directory

Standalone per-device scripts, run individually on the Pi:
- `test/<device>/read.py` — live continuous monitor for that sensor (Ctrl-C to stop).
- `test/<device>/*_scanner.py` — one-off wiring/channel discovery tools.
- `test/camera/record_camera.py` — grab a single camera snapshot.
- `test/scenarios.md` — **human acting scripts** to perform in front of the node
  for realistic occupancy data; each opens with a clap at t=0 (`clap_at_t0`) as the
  cross-stream sync marker.

## Conventions

- `data/`, `*.mp4`, `*.wav`, and `PRIVATE.md` are gitignored — recordings are not
  version-controlled.
- Keep raw sensor values raw — capture writes uncorrected readings; any
  post-processing happens downstream, not baked into the CSVs.
- New device-driving scripts hardcode the device path/port (matching `capture.py`)
  rather than adding flags.
