#!/usr/bin/env python3
"""
Standalone health check for the camera and microphones — no recording folder,
no web UI, no streaming. Run it on the Pi to see at a glance whether each
audio/video sensor is actually working.

It tests three things, each independently (a failure in one does not stop the
others), and prints a PASS / WARN / FAIL summary:

    1. USB camera        -- grabs a short clip, confirms frames decode and the
                            image is not fully black.
    2. Camera USB mic    -- records a few seconds and checks the signal is real
                            bipolar audio, not a frozen DC value. (This mic was
                            found dead: it emits a constant ~3145 and never
                            responds to sound, so we test the *shape*, not just
                            the level.)
    3. MCP3008/MAX9814   -- samples the ADC mic over SPI and reports its rate
                            and how much the level varies.

Device paths/format match capture.py. Run on the Pi:

    ~/CityOS/.venv/bin/python ~/CityOS/check_av.py
"""

import array
import math
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

# --- device config (matches capture.py) ------------------------------------
CAMERA = "/dev/v4l/by-id/usb-lihappe8_Corp._USB_2.0_Camera-video-index0"
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 640, 480, 30

MIC_DEVICE = "plughw:CARD=Camera,DEV=0"  # camera's built-in mic (by card name)
AUDIO_RATE, AUDIO_CHANNELS = 48000, 2

# --- test durations (seconds) -----------------------------------------------
CAMERA_TEST_SECONDS = 2
CAMERA_MIC_TEST_SECONDS = 3
ADC_MIC_TEST_SECONDS = 2

FULL_SCALE = 32768.0  # 16-bit signed


def _dbfs(amplitude):
    """Amplitude (0..32768) as dB relative to full scale."""
    if amplitude <= 0:
        return float("-inf")
    return 20 * math.log10(amplitude / FULL_SCALE)


# --- 1. camera --------------------------------------------------------------
def check_camera():
    print("\n[1] USB camera")
    print(f"    device: {CAMERA}")
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "cam.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "v4l2", "-framerate", str(CAMERA_FPS),
            "-input_format", "mjpeg",
            "-video_size", f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
            "-i", CAMERA, "-t", str(CAMERA_TEST_SECONDS),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(clip),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except FileNotFoundError:
            print("    FAIL: ffmpeg not found")
            return "FAIL"
        except subprocess.TimeoutExpired:
            print("    FAIL: camera capture timed out (device busy or unplugged?)")
            return "FAIL"
        if proc.returncode != 0 or not clip.exists() or clip.stat().st_size == 0:
            tail = proc.stderr.strip().splitlines()[-3:]
            print("    FAIL: ffmpeg could not capture from the camera")
            for line in tail:
                print(f"      {line}")
            return "FAIL"

        # how many frames did we actually decode?
        frames = _ffprobe_frames(clip)
        size_kb = clip.stat().st_size / 1024
        print(f"    captured {frames} frames, {size_kb:.0f} KB in {CAMERA_TEST_SECONDS}s")

        # is the whole clip black (lens cap / dead sensor)?
        black = _is_all_black(clip)
        if frames == 0:
            print("    FAIL: no frames decoded")
            return "FAIL"
        if black:
            print("    WARN: image is fully black -- lens covered or sensor not exposing")
            return "WARN"
        print("    PASS: camera is producing video")
        return "PASS"


def _ffprobe_frames(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames",
             "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
             "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return int(out.stdout.strip() or 0)
    except (ValueError, subprocess.SubprocessError):
        return -1


def _is_all_black(path):
    """True if ffmpeg's blackdetect flags ~the entire clip as black."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-i", str(path), "-vf", "blackdetect=d=0.5:pic_th=0.98",
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.SubprocessError:
        return False
    for line in out.stderr.splitlines():
        if "black_duration" in line:
            for token in line.split():
                if token.startswith("black_duration:"):
                    dur = float(token.split(":")[1])
                    return dur >= CAMERA_TEST_SECONDS * 0.9
    return False


# --- 2. camera USB mic ------------------------------------------------------
def check_camera_mic():
    print("\n[2] Camera USB mic")
    print(f"    device: {MIC_DEVICE}  ({AUDIO_RATE} Hz, {AUDIO_CHANNELS}ch)")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "mic.wav"
        cmd = [
            "arecord", "-q", "-D", MIC_DEVICE,
            "-f", "S16_LE", "-r", str(AUDIO_RATE), "-c", str(AUDIO_CHANNELS),
            "-d", str(CAMERA_MIC_TEST_SECONDS), str(wav),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except FileNotFoundError:
            print("    FAIL: arecord not found")
            return "FAIL"
        except subprocess.TimeoutExpired:
            print("    FAIL: arecord timed out")
            return "FAIL"
        if proc.returncode != 0 or not wav.exists():
            print(f"    FAIL: arecord failed -- {proc.stderr.strip()}")
            return "FAIL"
        return _analyze_wav(wav)


def _analyze_wav(path):
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        samples = array.array("h")
        samples.frombytes(w.readframes(w.getnframes()))
    mono = samples[0::channels] if channels > 1 else samples
    n = len(mono)
    if n == 0:
        print("    FAIL: empty recording")
        return "FAIL"

    mean = sum(mono) / n
    peak = max(abs(mn := min(mono)), abs(mx := max(mono)))
    ac_rms = math.sqrt(sum((s - mean) ** 2 for s in mono) / n)

    # A healthy AC mic swings BOTH sides of its DC bias. The dead camera mic
    # sits pinned at its rest value (~3145) and only ever dips downward, so it
    # almost never rises above its own mean.
    above = sum(1 for s in mono if s > mean)
    frac_above = above / n

    print(f"    samples={n}  dc_mean={mean:.0f}  min={mn}  max={mx}")
    print(f"    ac_rms={ac_rms:.1f} ({_dbfs(ac_rms):.1f} dBFS)  "
          f"peak={peak} ({_dbfs(peak):.1f} dBFS)")
    print(f"    samples above DC mean: {frac_above*100:.1f}%")

    if frac_above < 0.02:
        print("    FAIL: signal is one-sided / pinned at a DC value -- "
              "mic is not producing real audio (likely dead)")
        return "FAIL"
    if ac_rms < FULL_SCALE * 10 ** (-50 / 20):  # below ~-50 dBFS
        print("    WARN: audio present but very quiet -- silent room, or low "
              "sensitivity. Re-run while making noise to confirm.")
        return "WARN"
    print("    PASS: real bipolar audio detected")
    return "PASS"


# --- 3. MCP3008 / MAX9814 mic ----------------------------------------------
def check_adc_mic():
    print("\n[3] MCP3008/MAX9814 mic (SPI ADC, channel P0)")
    try:
        import board
        import busio
        import digitalio
        import adafruit_mcp3xxx.mcp3008 as MCP
        from adafruit_mcp3xxx.analog_in import AnalogIn
        spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
        cs = digitalio.DigitalInOut(board.D8)
        mcp = MCP.MCP3008(spi, cs, ref_voltage=3.3)
        mic = AnalogIn(mcp, MCP.P0)
    except Exception as error:  # noqa: BLE001 -- mirror capture.py's broad guard
        print(f"    FAIL: ADC unavailable ({error})")
        return "FAIL"

    vals = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < ADC_MIC_TEST_SECONDS:
        vals.append(mic.voltage)
    elapsed = time.monotonic() - t0
    rate = len(vals) / elapsed

    vmin, vmax = min(vals), max(vals)
    vmean = sum(vals) / len(vals)
    vstd = math.sqrt(sum((v - vmean) ** 2 for v in vals) / len(vals))

    print(f"    sampled {len(vals)} reads in {elapsed:.2f}s  (~{rate:.0f} Hz)")
    print(f"    voltage  min={vmin:.3f}  max={vmax:.3f}  mean={vmean:.3f}  std={vstd:.3f}")

    if vstd < 0.01:
        print("    FAIL: level is flat -- ADC reading nothing (check wiring/power)")
        return "FAIL"
    print("    PASS: ADC mic is responding (loudness/activity level)")
    print("    note: ~20 Hz in capture.py = level only, not listenable audio")
    return "PASS"


def main():
    print("=" * 60)
    print("camera + microphone health check")
    print("=" * 60)
    results = {
        "camera": check_camera(),
        "camera_mic": check_camera_mic(),
        "adc_mic": check_adc_mic(),
    }
    print("\n" + "=" * 60)
    print("summary")
    print("=" * 60)
    for name, verdict in results.items():
        print(f"    {name:12s} {verdict}")
    # exit non-zero if anything failed, for scripting
    sys.exit(1 if "FAIL" in results.values() else 0)


if __name__ == "__main__":
    main()
