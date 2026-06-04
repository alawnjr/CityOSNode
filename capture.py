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
            custom_board_i2c.csv         (BME680 / TCS34725 / ADXL345 / MLX90393)
            mcp3008_mic.csv              (mic level via MCP3008 ADC)
            radar_ops243.csv             (OPS243-A radar, raw serial lines)

Run on the Pi:  python capture.py
"""

import csv
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

DURATION_SECONDS = 30
DATA_DIR = Path(__file__).resolve().parent / "data"

CAMERA = "/dev/v4l/by-id/usb-lihappe8_Corp._USB_2.0_Camera-video-index0"
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 640, 480, 30
TIMESTAMP_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

RADAR_PORT, RADAR_BAUD = "/dev/ttyACM0", 19200

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
        print(f"i2c: unavailable ({error})")
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


def log_mcp3008_mic(path, stop_event, start):
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
    except Exception as error:
        print(f"mcp3008: unavailable ({error})")
        return

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_seconds", "voltage"])
        while not stop_event.is_set():
            t = time.monotonic() - start
            writer.writerow([f"{t:.3f}", f"{mic.voltage:.4f}"])
            handle.flush()
            time.sleep(0.05)


def log_radar(path, stop_event, start):
    try:
        import serial
        port = serial.Serial(RADAR_PORT, RADAR_BAUD, timeout=0.5)
        port.write(b"OD")  # OPS243-A: enable data output
    except Exception as error:
        print(f"radar: unavailable ({error})")
        return

    with path.open("w", newline="") as handle, port:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_seconds", "raw"])
        while not stop_event.is_set():
            line = port.readline().decode("utf-8", errors="ignore").strip()
            if line:
                writer.writerow([f"{time.monotonic() - start:.3f}", line])
                handle.flush()


def record_camera(path):
    timestamp_filter = (
        f"drawtext=fontfile={TIMESTAMP_FONT}:"
        "text=%{localtime\\\\:%Y-%m-%d %H-%M-%S}:"
        "x=12:y=12:fontsize=22:fontcolor=white:"
        "box=1:boxcolor=black@0.55:boxborderw=8"
    )
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "v4l2", "-framerate", str(CAMERA_FPS),
        "-input_format", "mjpeg", "-video_size", f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
        "-i", CAMERA, "-t", str(DURATION_SECONDS),
        "-vf", timestamp_filter,
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(path),
    ], check=True)


def write_camera_timestamps(path, fps):
    frame_count = DURATION_SECONDS * fps
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_seconds"])
        for i in range(frame_count):
            writer.writerow([i, f"{i / fps:.6f}"])
    return frame_count


def write_metadata(rec_dir, start_time, end_time, frame_count):
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
            "mcp3008_mic": {
                "modality": "audio_level",
                "path": "streams/mcp3008_mic.csv",
                "sample_rate_hz": 20,
                "fields": ["voltage"],
            },
            "radar_ops243": {
                "modality": "motion",
                "path": "streams/radar_ops243.csv",
                "fields": ["raw"],
            },
        },
    }
    (rec_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main():
    rec_dir = make_recording_dir()
    streams = rec_dir / "streams"
    print(f"Recording {DURATION_SECONDS}s -> {rec_dir}")

    stop_event = threading.Event()
    start = time.monotonic()
    start_time = datetime.now().astimezone()

    threads = [
        threading.Thread(target=log_i2c, args=(streams / "custom_board_i2c.csv", stop_event, start)),
        threading.Thread(target=log_mcp3008_mic, args=(streams / "mcp3008_mic.csv", stop_event, start)),
        threading.Thread(target=log_radar, args=(streams / "radar_ops243.csv", stop_event, start)),
    ]
    for thread in threads:
        thread.start()

    record_camera(streams / "camera_main.mp4")  # blocks for DURATION_SECONDS

    stop_event.set()
    for thread in threads:
        thread.join()

    end_time = datetime.now().astimezone()
    frame_count = write_camera_timestamps(streams / "camera_main_timestamps.csv", CAMERA_FPS)
    write_metadata(rec_dir, start_time, end_time, frame_count)
    print(f"Done -> {rec_dir}")


if __name__ == "__main__":
    main()
