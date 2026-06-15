#!/usr/bin/env python3
"""
Capture raw data from all smartroom sensors into one recording.

Records for DURATION_SECONDS and writes a recording folder under data/ in the
same layout as sample_dataset/:

    data/day_NN_YYYY-MM-DD/rec_YYYYMMDD_NNN/
        metadata.json
        streams/
            camera_main.mp4              (USB camera, h264)
            camera_main_timestamps.csv   (frame_index, timestamp_seconds)
            mic_array.wav                (C920 camera mic, 48 kHz stereo)
            custom_board_i2c.csv         (BME680 / TCS34725 / ADXL345 / MLX90393)
            mcp3008_mic.csv              (mic level via MCP3008 ADC, ~20 Hz)
            mcp3008_audio.wav            (MAX9814 mic waveform via MCP3008, ~28 kHz)

Run on the Pi:  python capture.py            # 30s (default)
                python capture.py -d 60      # 60s
"""

import argparse
import array
import csv
import json
import os
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

DEFAULT_DURATION = 30
DURATION_SECONDS = DEFAULT_DURATION  # overridden by --duration in main()
DATA_DIR = Path(__file__).resolve().parent / "data"

# Logitech C920 (wide 16:9 FOV); addressed by stable by-id path. The 16:9 MJPG
# modes use the full sensor width — the 4:3 640x480 mode is center-cropped/narrower.
CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_6E6D65AF-video-index0"
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 1280, 720, 30
TIMESTAMP_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

MIC_DEVICE = "plughw:CARD=C920,DEV=0"  # C920's built-in mic, by card name (a real, working mic)
AUDIO_RATE, AUDIO_CHANNELS = 48000, 2

# MCP3008 ADC mic (MAX9814 on channel P0, SPI0 CE0 = board.D8). One thread owns
# the bus and reads it flat-out, producing both the ~20 Hz level CSV and a
# real audio waveform. This is now a second working mic alongside the C920's
# (mic_array.wav); the old lihappe8 camera mic was dead.
ADC_REF_VOLTAGE = 3.3
MCP3008_SPI_HZ = 1_350_000        # MCP3008 rated max clock at ~3.3V -> ~28 kHz sampling
MCP3008_LEVEL_HZ = 20             # mcp3008_mic.csv level cadence (unchanged)
MCP3008_AUDIO_GAIN = 32           # 10-bit (±512) -> 16-bit PCM, with headroom
MCP3008_AUDIO_RATE_NOMINAL = 28000  # metadata fallback if the measured rate is unknown
# The mic is software-timed (no hardware clock), so sample spacing jitters and
# the bulk of the noise energy sits above the voice band. The audio wav is
# resampled onto a uniform grid (jitter correction) using the per-sample
# timestamps, then band-passed to the voice range before encoding. Needs numpy;
# without it we fall back to the plain AC-coupled encode.
MCP3008_AUDIO_HP_HZ = 80          # high-pass: drop DC/rumble
MCP3008_AUDIO_LP_HZ = 5000        # low-pass: drop out-of-band hiss/aliasing

I2C_FIELDS = [
    "temperature_c", "humidity_pct", "pressure_hpa", "gas_ohm",
    "light_lux", "color_temperature_k", "red", "green", "blue",
    "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
    "mag_x_ut", "mag_y_ut", "mag_z_ut",
]


def make_recording_dir():
    """Pick the day_NN_DATE / rec_DATE_NNN folder, matching sample_dataset."""
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    compact = now.strftime("%Y%m%d")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing_days = sorted(DATA_DIR.glob("day_*"))
    day_dir = next((d for d in existing_days if d.name.endswith(date)), None)
    if day_dir is None:
        day_dir = DATA_DIR / f"day_{len(existing_days) + 1:02d}_{date}"

    rec_num = len(list(day_dir.glob("rec_*"))) + 1
    rec_dir = day_dir / f"rec_{compact}_{rec_num:03d}"
    (rec_dir / "streams").mkdir(parents=True, exist_ok=True)
    return rec_dir


# --------------------------------------------------------------------------- #
# Sensor loggers (each runs in its own thread until stop_event is set)
# --------------------------------------------------------------------------- #
def log_i2c(path, stop_event, start):
    try:
        import board
        import adafruit_bme680
        import adafruit_tcs34725
        import adafruit_adxl34x
        import adafruit_mlx90393
        i2c = board.I2C()
        bme = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=0x76)
        tcs = adafruit_tcs34725.TCS34725(i2c)
        adxl = adafruit_adxl34x.ADXL345(i2c, address=0x53)
        mlx = adafruit_mlx90393.MLX90393(i2c, address=0x0C)
    except Exception as error:
        print(f"i2c: unavailable ({error})", file=sys.stderr)
        return

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_seconds"] + I2C_FIELDS)
        while not stop_event.is_set():
            t = time.monotonic() - start
            r, g, b = tcs.color_rgb_bytes
            ax, ay, az = adxl.acceleration
            mx, my, mz = mlx.magnetic
            color_temp = tcs.color_temperature
            writer.writerow([
                f"{t:.3f}",
                f"{bme.temperature:.2f}", f"{bme.relative_humidity:.2f}",
                f"{bme.pressure:.2f}", bme.gas,
                f"{tcs.lux:.2f}", f"{color_temp:.0f}" if color_temp else "",
                r, g, b,
                f"{ax:.3f}", f"{ay:.3f}", f"{az:.3f}",
                f"{mx:.3f}", f"{my:.3f}", f"{mz:.3f}",
            ])
            handle.flush()
            time.sleep(1.0)


def log_mcp3008(audio_path, level_path, stop_event, start, result):
    """Read the MAX9814 mic on the MCP3008 (SPI channel P0) as fast as the bus
    allows and produce both ADC outputs from the one stream of samples:

      - level_path  raw voltage at ~MCP3008_LEVEL_HZ (loudness/activity, as before)
      - audio_path  the full waveform as a real wav (a working audio mic)

    A single thread owns the chip select so the two outputs never contend for
    the SPI bus. The high-rate samples are buffered and written to the wav once
    recording stops (the measured rate is reported back via `result`)."""
    try:
        import spidev
        spi = spidev.SpiDev()
        spi.open(0, 0)                       # SPI0, CE0 (board.D8) -- matches wiring
        spi.max_speed_hz = MCP3008_SPI_HZ
    except Exception as error:
        print(f"mcp3008: unavailable ({error})", file=sys.stderr)
        return

    def read_raw():
        # MCP3008 single-ended channel 0 -> 10-bit reading.
        reply = spi.xfer2([1, 8 << 4, 0])
        return ((reply[1] & 3) << 8) | reply[2]

    raw = array.array("H")
    times = array.array("d")             # per-sample timestamps, for jitter correction
    next_level = 0.0
    try:
        with open(level_path, "w", newline="") as level_file:
            writer = csv.writer(level_file)
            writer.writerow(["timestamp_seconds", "voltage"])
            while not stop_event.is_set():
                value = read_raw()
                t = time.monotonic() - start
                raw.append(value)
                times.append(t)
                if t >= next_level:
                    writer.writerow([f"{t:.3f}", f"{value / 1023 * ADC_REF_VOLTAGE:.4f}"])
                    level_file.flush()
                    next_level += 1.0 / MCP3008_LEVEL_HZ
    finally:
        spi.close()

    rate = _encode_adc_audio(raw, times, audio_path)
    if rate:
        result["audio_rate"] = rate


def _encode_adc_audio(raw, times, audio_path):
    """Encode the buffered MCP3008 samples to a 16-bit wav and return its rate.

    The samples are software-timed, so before encoding we resample them onto a
    uniform grid using the captured timestamps (this undoes the ~20% timing
    jitter) and band-pass to the voice range to drop out-of-band hiss. This
    needs numpy; without it we fall back to a plain AC-coupled encode at the
    average rate. Raw, uncorrected readings remain in the level CSV -- only this
    listenable audio artifact is filtered."""
    n = len(raw)
    if n < 2:
        return None
    duration = times[-1] - times[0]
    if duration <= 0:
        return None
    rate = int(round(n / duration)) or MCP3008_AUDIO_RATE_NOMINAL

    try:
        import numpy as np
        signal = np.asarray(raw, dtype=np.float64)
        signal -= signal.mean()                      # AC-couple
        stamps = np.asarray(times, dtype=np.float64) - times[0]
        # 1) jitter correction: resample the unevenly-spaced samples onto a
        #    uniform time grid so playback timing is regular.
        uniform = np.linspace(0.0, duration, n)
        signal = np.interp(uniform, stamps, signal)
        # 2) voice band-pass via an FFT mask.
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=duration / n)
        spectrum[(freqs < MCP3008_AUDIO_HP_HZ) | (freqs > MCP3008_AUDIO_LP_HZ)] = 0.0
        signal = np.fft.irfft(spectrum, n=n)
        pcm = np.clip(signal * MCP3008_AUDIO_GAIN, -32768, 32767).astype("<i2").tobytes()
    except Exception as error:
        # numpy missing (or any failure): plain AC-coupled encode, no filtering.
        print(f"mcp3008: writing unfiltered audio ({error})", file=sys.stderr)
        mean = sum(raw) / n
        pcm = array.array("h", (
            max(-32768, min(32767, int((value - mean) * MCP3008_AUDIO_GAIN)))
            for value in raw
        )).tobytes()

    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return rate


def start_audio(path):
    """Record the camera mic to a wav for DURATION_SECONDS (non-blocking)."""
    try:
        return subprocess.Popen([
            "arecord", "-q",
            "-D", MIC_DEVICE,
            "-f", "S16_LE",
            "-r", str(AUDIO_RATE),
            "-c", str(AUDIO_CHANNELS),
            "-d", str(DURATION_SECONDS),
            str(path),
        ])
    except Exception as error:
        print(f"audio: unavailable ({error})", file=sys.stderr)
        return None


def record_camera(path):
    timestamp_filter = (
        f"drawtext=fontfile={TIMESTAMP_FONT}:"
        "text=%{localtime\\\\:%Y-%m-%d %H-%M-%S}:"
        "x=12:y=12:fontsize=22:fontcolor=white:"
        "box=1:boxcolor=black@0.55:boxborderw=8"
    )
    source = [
        "ffmpeg", "-y",
        "-f", "v4l2", "-framerate", str(CAMERA_FPS),
        "-input_format", "mjpeg", "-video_size", f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
    ]
    preview_path = os.environ.get("SMARTROOM_PREVIEW")
    if preview_path:
        # Record to file (with overlay) AND write the latest frame (~5 fps) to a
        # single jpg so the web page can show the camera while recording. The
        # duration limit goes on the INPUT so BOTH outputs stop together (an
        # unbounded second output would keep ffmpeg running forever).
        command = source + [
            "-t", str(DURATION_SECONDS), "-i", CAMERA,
            "-map", "0:v:0", "-vf", timestamp_filter,
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(path),
            "-map", "0:v:0", "-r", "5", "-update", "1", "-y", preview_path,
        ]
    else:
        command = source + [
            "-i", CAMERA, "-t", str(DURATION_SECONDS),
            "-vf", timestamp_filter,
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(path),
        ]
    subprocess.run(command, check=True)


def write_camera_timestamps(path, fps):
    frame_count = DURATION_SECONDS * fps
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_seconds"])
        for i in range(frame_count):
            writer.writerow([i, f"{i / fps:.6f}"])
    return frame_count


def write_metadata(rec_dir, start_time, end_time, frame_count, mcp3008_audio_rate=None):
    metadata = {
        "recording_id": rec_dir.name,
        "space": "smart_room_1",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": DURATION_SECONDS,
        "schema_version": "0.1",
        "streams": {
            "camera_main": {
                "modality": "video",
                "path": "streams/camera_main.mp4",
                "codec": "h264",
                "resolution": [CAMERA_WIDTH, CAMERA_HEIGHT],
                "fps": CAMERA_FPS,
                "frame_count": frame_count,
                "timestamps_path": "streams/camera_main_timestamps.csv",
            },
            "custom_board_i2c": {
                "modality": "environmental",
                "path": "streams/custom_board_i2c.csv",
                "sample_rate_hz": 1,
                "fields": I2C_FIELDS,
            },
            "mic_array": {
                "modality": "audio",
                "path": "streams/mic_array.wav",
                "sample_rate_hz": AUDIO_RATE,
                "channels": AUDIO_CHANNELS,
            },
            "mcp3008_mic": {
                "modality": "audio_level",
                "path": "streams/mcp3008_mic.csv",
                "sample_rate_hz": MCP3008_LEVEL_HZ,
                "fields": ["voltage"],
            },
            "mcp3008_audio": {
                "modality": "audio",
                "path": "streams/mcp3008_audio.wav",
                "sample_rate_hz": mcp3008_audio_rate or MCP3008_AUDIO_RATE_NOMINAL,
                "channels": 1,
                "note": (
                    "MAX9814 mic via MCP3008 ADC, raw spidev; AC-coupled, "
                    "jitter-corrected (resampled from per-sample timestamps) and "
                    f"band-passed {MCP3008_AUDIO_HP_HZ}-{MCP3008_AUDIO_LP_HZ} Hz. "
                    "Software-timed, so the rate is approximate."
                ),
            },
        },
    }
    (rec_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main():
    global DURATION_SECONDS
    parser = argparse.ArgumentParser(description="Capture raw data from all smartroom sensors.")
    parser.add_argument("-d", "--duration", type=int, default=DEFAULT_DURATION,
                        help=f"Recording length in seconds (default: {DEFAULT_DURATION}).")
    DURATION_SECONDS = parser.parse_args().duration

    rec_dir = make_recording_dir()
    streams = rec_dir / "streams"
    print(f"Recording {DURATION_SECONDS}s -> {rec_dir}", file=sys.stderr)

    stop_event = threading.Event()
    start = time.monotonic()
    start_time = datetime.now().astimezone()

    mcp_result = {}
    threads = [
        threading.Thread(target=log_i2c, args=(streams / "custom_board_i2c.csv", stop_event, start), daemon=True),
        threading.Thread(target=log_mcp3008, args=(streams / "mcp3008_audio.wav", streams / "mcp3008_mic.csv", stop_event, start, mcp_result), daemon=True),
    ]
    for thread in threads:
        thread.start()
    audio_proc = start_audio(streams / "mic_array.wav")

    # The camera drives the recording length; whatever happens, stop the sensor
    # threads and audio cleanly so this process always exits (a camera failure
    # must not leave it hanging).
    try:
        record_camera(streams / "camera_main.mp4")  # blocks for DURATION_SECONDS
    finally:
        stop_event.set()
        if audio_proc:
            try:
                audio_proc.wait(timeout=10)
            except Exception:
                audio_proc.terminate()
        for thread in threads:
            # Generous timeout: the mcp3008 thread encodes its buffered audio to
            # the wav after stop, which takes a moment on long recordings.
            thread.join(timeout=30)

    end_time = datetime.now().astimezone()
    frame_count = write_camera_timestamps(streams / "camera_main_timestamps.csv", CAMERA_FPS)
    write_metadata(rec_dir, start_time, end_time, frame_count, mcp_result.get("audio_rate"))
    print(f"Done -> {rec_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
