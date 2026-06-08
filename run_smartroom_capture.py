#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path.home() / "CityOS" / "data"
TIMESTAMP_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
RADAR_PORT = "/dev/serial/by-id/usb-Infineon_IFX_CDC-if00"
RADAR_BAUD = 115200


def run_text(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except Exception:
        return ""


def natural_key(text):
    chunks = []
    current = ""
    current_is_digit = text[:1].isdigit()
    for char in text:
        if char.isdigit() == current_is_digit:
            current += char
        else:
            chunks.append(int(current) if current_is_digit else current)
            current = char
            current_is_digit = char.isdigit()
    if current:
        chunks.append(int(current) if current_is_digit else current)
    return chunks


def is_camera_capture_device(path):
    formats = run_text(["v4l2-ctl", "-d", path, "--list-formats-ext"])
    return "Video Capture" in formats and "'MJPG'" in formats


def detect_camera():
    by_id = Path("/dev/v4l/by-id")
    if by_id.exists():
        stable_paths = sorted(by_id.glob("*video-index0*"), key=lambda p: natural_key(str(p)))
        for path in stable_paths:
            resolved = str(path.resolve())
            if is_camera_capture_device(resolved):
                return str(path)

    output = run_text(["v4l2-ctl", "--list-devices"])
    devices = []
    current_name = ""
    for raw_line in output.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            current_name = ""
            continue
        if not raw_line.startswith("\t") and stripped.endswith(":"):
            current_name = stripped[:-1]
            continue
        if stripped.startswith("/dev/video"):
            devices.append((stripped, current_name))

    usb_devices = [path for path, name in devices if "usb" in name.lower()]
    candidates = usb_devices or [path for path, _ in devices]
    for path in candidates:
        if is_camera_capture_device(path):
            return path

    for path in sorted(Path("/dev").glob("video*"), key=lambda p: natural_key(str(p))):
        if is_camera_capture_device(str(path)):
            return str(path)

    raise RuntimeError("No USB camera capture device found.")


def write_env_header(path):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_seconds", "temperature_c", "humidity_pct"])


def append_env_row(path, timestamp_seconds, temperature_c="", humidity_pct=""):
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"{timestamp_seconds:.3f}", temperature_c, humidity_pct])


def write_custom_i2c_header(path):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "timestamp_seconds",
            "temperature_c",
            "humidity_pct",
            "pressure_hpa",
            "gas_ohm",
            "light_lux",
            "color_temperature_k",
            "red",
            "green",
            "blue",
            "accel_x_mps2",
            "accel_y_mps2",
            "accel_z_mps2",
            "mag_x_ut",
            "mag_y_ut",
            "mag_z_ut",
        ])


class CustomI2CBoard:
    def __init__(self):
        self.i2c = None
        self.bme680 = None
        self.tcs = None
        self.adxl = None
        self.mlx = None
        self.available = False
        try:
            import board
            self.i2c = board.I2C()
            self.available = True
        except Exception:
            return
        try:
            import adafruit_bme680
            self.bme680 = adafruit_bme680.Adafruit_BME680_I2C(self.i2c, address=0x76)
            self.bme680.sea_level_pressure = 1013.25
        except Exception:
            self.bme680 = None
        try:
            import adafruit_tcs34725
            self.tcs = adafruit_tcs34725.TCS34725(self.i2c)
            self.tcs.integration_time = 100
            self.tcs.gain = 4
        except Exception:
            self.tcs = None
        try:
            import adafruit_adxl34x
            self.adxl = adafruit_adxl34x.ADXL345(self.i2c, address=0x53)
        except Exception:
            self.adxl = None
        try:
            import adafruit_mlx90393
            self.mlx = adafruit_mlx90393.MLX90393(self.i2c, address=0x0C)
        except Exception:
            self.mlx = None

    def read(self):
        data = {
            "temperature_c": "",
            "humidity_pct": "",
            "pressure_hpa": "",
            "gas_ohm": "",
            "light_lux": "",
            "color_temperature_k": "",
            "red": "",
            "green": "",
            "blue": "",
            "accel_x_mps2": "",
            "accel_y_mps2": "",
            "accel_z_mps2": "",
            "mag_x_ut": "",
            "mag_y_ut": "",
            "mag_z_ut": "",
        }
        if self.bme680:
            try:
                data["temperature_c"] = f"{self.bme680.temperature:.2f}"
                data["humidity_pct"] = f"{self.bme680.relative_humidity:.2f}"
                data["pressure_hpa"] = f"{self.bme680.pressure:.2f}"
                data["gas_ohm"] = f"{self.bme680.gas}"
            except Exception:
                pass
        if self.tcs:
            try:
                red, green, blue = self.tcs.color_rgb_bytes
                data["red"] = red
                data["green"] = green
                data["blue"] = blue
                data["light_lux"] = f"{self.tcs.lux:.2f}"
                color_temp = self.tcs.color_temperature
                data["color_temperature_k"] = f"{color_temp:.0f}" if color_temp else ""
            except Exception:
                pass
        if self.adxl:
            try:
                x, y, z = self.adxl.acceleration
                data["accel_x_mps2"] = f"{x:.3f}"
                data["accel_y_mps2"] = f"{y:.3f}"
                data["accel_z_mps2"] = f"{z:.3f}"
            except Exception:
                pass
        if self.mlx:
            try:
                x, y, z = self.mlx.magnetic
                data["mag_x_ut"] = f"{x:.3f}"
                data["mag_y_ut"] = f"{y:.3f}"
                data["mag_z_ut"] = f"{z:.3f}"
            except Exception:
                pass
        return data


def append_sensor_rows(env_path, custom_i2c_path, timestamp_seconds, board_reader=None):
    data = board_reader.read() if board_reader and board_reader.available else {}
    append_env_row(
        env_path,
        timestamp_seconds,
        data.get("temperature_c", ""),
        data.get("humidity_pct", ""),
    )
    with custom_i2c_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            f"{timestamp_seconds:.3f}",
            data.get("temperature_c", ""),
            data.get("humidity_pct", ""),
            data.get("pressure_hpa", ""),
            data.get("gas_ohm", ""),
            data.get("light_lux", ""),
            data.get("color_temperature_k", ""),
            data.get("red", ""),
            data.get("green", ""),
            data.get("blue", ""),
            data.get("accel_x_mps2", ""),
            data.get("accel_y_mps2", ""),
            data.get("accel_z_mps2", ""),
            data.get("mag_x_ut", ""),
            data.get("mag_y_ut", ""),
            data.get("mag_z_ut", ""),
        ])


def write_radar_header(path):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_seconds", "sensor_time_seconds", "measurement", "unit", "value", "raw"])


def parse_radar_line(raw_line):
    line = raw_line.strip()
    if not line:
        return "", "", "", ""
    if line.startswith("{") or line.startswith("["):
        return "", "", "", line
    try:
        row = next(csv.reader([line]))
    except Exception:
        return "", "", "", line
    if len(row) >= 3:
        sensor_time = row[0].strip()
        unit = row[1].strip().strip('"')
        value = row[2].strip()
        measurement = {
            "cm": "distance_cm",
            "cmps": "velocity_cmps",
            "dB": "signal_db",
            "db": "signal_db",
        }.get(unit, unit or "unknown")
        return sensor_time, measurement, unit, value
    if len(row) == 1:
        return "", "unlabeled", "", row[0].strip()
    return "", "", "", line


def radar_logger(path, stop_event, start_monotonic, port=RADAR_PORT, baud=RADAR_BAUD):
    write_radar_header(path)
    try:
        import serial
    except Exception:
        return
    if not Path(port).exists():
        return
    try:
        serial_port = serial.Serial(port, baudrate=baud, timeout=0.25)
    except Exception:
        return
    with serial_port:
        while not stop_event.is_set():
            try:
                raw = serial_port.readline()
            except Exception:
                break
            if not raw:
                continue
            decoded = raw.decode("utf-8", errors="replace").strip()
            sensor_time, measurement, unit, value = parse_radar_line(decoded)
            timestamp_seconds = time.monotonic() - start_monotonic
            with path.open("a", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    f"{timestamp_seconds:.3f}",
                    sensor_time,
                    measurement,
                    unit,
                    value,
                    decoded,
                ])


def write_camera_timestamps(path, duration_seconds, fps):
    frame_count = max(1, int(round(duration_seconds * fps)))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_seconds"])
        for frame_index in range(frame_count):
            writer.writerow([frame_index, f"{frame_index / fps:.6f}"])


def create_empty_wav(path, duration_seconds, sample_rate=16000, channels=1):
    frame_count = max(1, int(duration_seconds * sample_rate))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        chunk = b"\x00\x00" * min(frame_count, sample_rate)
        remaining = frame_count
        while remaining > 0:
            frames = min(remaining, sample_rate)
            handle.writeframes(chunk[: frames * 2])
            remaining -= frames


def detect_audio_device():
    output = run_text(["arecord", "-l"])
    for line in output.splitlines():
        if "USB" not in line and "Camera" not in line:
            continue
        match = re.search(r"card\s+(\d+):.*device\s+(\d+):", line)
        if match:
            return f"plughw:{match.group(1)},{match.group(2)}"
    return ""


def start_audio_capture(path, duration_seconds, sample_rate):
    if not shutil_which("arecord"):
        return None
    audio_device = detect_audio_device()
    command = [
        "arecord",
        "-q",
    ]
    if audio_device:
        command.extend(["-D", audio_device])
    command.extend([
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        "1",
        "-d",
        str(int(duration_seconds)),
        str(path),
    ])
    try:
        return subprocess.Popen(command, preexec_fn=os.setsid)
    except Exception:
        return None


def shutil_which(name):
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(folder) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return ""


def build_video_command(args, camera, video_path):
    timestamp_filter = (
        f"drawtext=fontfile={TIMESTAMP_FONT}:"
        "text=%{localtime\\\\:%Y-%m-%d %H-%M-%S}:"
        "x=12:y=12:fontsize=22:fontcolor=white:"
        "box=1:boxcolor=black@0.55:boxborderw=8"
    )
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "v4l2",
        "-framerate",
        str(args.fps),
        "-input_format",
        "mjpeg",
        "-video_size",
        f"{args.width}x{args.height}",
        "-i",
        camera,
    ]
    if args.duration > 0:
        command.extend(["-t", str(args.duration)])
    command.extend(
        [
            "-vf",
            timestamp_filter,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
    )
    return command


def stop_stale_camera_streams(camera):
    targets = {camera}
    try:
        targets.add(str(Path(camera).resolve()))
    except Exception:
        pass
    try:
        output = subprocess.check_output(["pgrep", "-af", "python3"], text=True)
    except Exception:
        output = ""
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        if "/home/smartroom/smartroom_recorder.py" not in command:
            continue
        try:
            os.kill(int(pid_text), signal.SIGTERM)
        except Exception:
            pass
    time.sleep(0.5)
    try:
        output = subprocess.check_output(["pgrep", "-af", "ffmpeg|ffplay"], text=True)
    except Exception:
        output = ""
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        if "v4l2" not in command:
            continue
        if not any(target in command for target in targets) and "pipe:1" not in command:
            continue
        try:
            os.kill(int(pid_text), signal.SIGTERM)
        except Exception:
            pass
    time.sleep(0.7)


def parse_args():
    parser = argparse.ArgumentParser(description="Capture Smartroom video and metadata.")
    parser.add_argument("--duration", type=float, default=60.0, help="Recording duration in seconds.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--camera", default="", help="Camera device path. Auto-detected when omitted.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--no-audio", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    start_wall = dt.datetime.now()
    day_dir = args.output_dir / start_wall.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    session_id = f"smartroom_{start_wall.strftime('%Y-%m-%d_%H-%M-%S')}"
    session_dir = day_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    camera = args.camera or detect_camera()
    stop_stale_camera_streams(camera)
    video_path = session_dir / "camera_main.mp4"
    camera_timestamps_path = session_dir / "camera_main_timestamps.csv"
    env_sensor_path = session_dir / "env_sensor_01.csv"
    custom_i2c_path = session_dir / "custom_board_i2c.csv"
    radar_sensor_path = session_dir / "radar_sensor_01.csv"
    mic_path = session_dir / "mic_array.wav"
    manifest_path = session_dir / "metadata.json"
    log_path = session_dir / "capture.log"

    write_env_header(env_sensor_path)
    write_custom_i2c_header(custom_i2c_path)
    write_radar_header(radar_sensor_path)
    board_reader = CustomI2CBoard()
    audio_process = None
    if not args.no_audio:
        audio_process = start_audio_capture(mic_path, args.duration, args.audio_sample_rate)

    command = build_video_command(args, camera, video_path)
    started = time.monotonic()
    radar_stop_event = threading.Event()
    radar_thread = threading.Thread(
        target=radar_logger,
        args=(radar_sensor_path, radar_stop_event, started),
        daemon=True,
    )
    radar_thread.start()
    with log_path.open("w") as log:
        log.write(" ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=log, text=True, preexec_fn=os.setsid)
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                append_sensor_rows(env_sensor_path, custom_i2c_path, elapsed, board_reader)
                time.sleep(1.0)
        except KeyboardInterrupt:
            if process.stdin:
                process.stdin.write("q\n")
                process.stdin.flush()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait()

    radar_stop_event.set()
    radar_thread.join(timeout=2)

    if audio_process:
        try:
            audio_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(audio_process.pid), signal.SIGTERM)
            audio_process.wait()
    elif not args.no_audio:
        create_empty_wav(mic_path, args.duration, args.audio_sample_rate)

    elapsed_seconds = max(0.0, time.monotonic() - started)
    write_camera_timestamps(camera_timestamps_path, elapsed_seconds, args.fps)

    manifest = {
        "session_id": session_id,
        "started_at_local": start_wall.isoformat(timespec="seconds"),
        "day_folder": day_dir.name,
        "duration_seconds": round(elapsed_seconds, 3),
        "camera_device": camera,
        "video_file": "camera_main.mp4",
        "camera_timestamps_file": "camera_main_timestamps.csv",
        "environment_sensor_file": "env_sensor_01.csv",
        "custom_i2c_file": "custom_board_i2c.csv",
        "radar_sensor_file": "radar_sensor_01.csv",
        "microphone_file": "mic_array.wav" if mic_path.exists() else None,
        "metadata_format": {
            "camera_main_timestamps.csv": ["frame_index", "timestamp_seconds"],
            "env_sensor_01.csv": ["timestamp_seconds", "temperature_c", "humidity_pct"],
            "custom_board_i2c.csv": [
                "timestamp_seconds",
                "temperature_c",
                "humidity_pct",
                "pressure_hpa",
                "gas_ohm",
                "light_lux",
                "color_temperature_k",
                "red",
                "green",
                "blue",
                "accel_x_mps2",
                "accel_y_mps2",
                "accel_z_mps2",
                "mag_x_ut",
                "mag_y_ut",
                "mag_z_ut",
            ],
            "radar_sensor_01.csv": [
                "timestamp_seconds",
                "sensor_time_seconds",
                "measurement",
                "unit",
                "value",
                "raw",
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if process.returncode != 0:
        print(f"Recording failed. Check {log_path}", file=sys.stderr)
        return process.returncode

    print(f"Saved session: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
