#!/usr/bin/env python3
"""
Record the Reolink NVR's cameras into the smartroom recording layout.

capture.py records the camera attached to the Pi it runs on. This records the
four Reolink cameras, which are not attached to anything: they live on an NVR
and are reached over RTSP. That difference drives every choice here.

WHERE THIS RUNS. Not on the quad server -- it cannot reach the NVR (no route
from 172.16.60.0/24 to the camera network; ping and TCP/554 both fail). The
host that can see the NVR runs this, and --upload pushes finished recordings to
the analysis volume. Measured on the lab laptop: RTSP/554 open, 1-4ms away.

LAYOUT. capture.py writes one camera into rec/streams/ and lets
upload_recording.sh fold it into rec/streams/<node>/ on the way up. Four cameras
do not fit that shape, so this writes the FINAL layout directly, following the
convention a RealSense node already uses -- ONE node directory holding several
clips, one metadata.json describing them all:

    data/day_NN_YYYY-MM-DD/rec_YYYYMMDD_HHMMSS/streams/reolink/
        camera_cam1_color.mp4  + camera_cam1_color_timestamps.csv
        ...                      (one pair per channel)
        metadata.json

The stems must be UNIQUE per camera, which is why none of them is camera_main:
live_infer's find_calib_clips() locates a camera's calibration by globbing
<cam_key>.mp4 across the whole archive, so four cameras sharing one stem would
each adopt whichever was calibrated last. camera_main also means "this node's
main camera", which four cameras with four different poses are not. Naming them
camera_cam<N>_color puts them in the same shape as camera_d455_color /
camera_d435_color, so they address as reolink-cam1 ... reolink-cam4.

Recording ids are timestamps (rec_YYYYMMDD_HHMMSS) rather than capture.py's
per-day counter. The counter is only unique among recordings made by the same
node, and these clips are uploaded into a tree the Pi is also writing into --
two nodes independently picking rec_..._001 would merge two unrelated sessions
into one recording directory.

TIMEBASE. The CSV is (frame_index, timestamp_seconds), matching what capture.py
writes for camera_main. There is no hw_timestamp_ms column: that is a real
hardware clock the RealSense sensors provide, and RTSP does not. Inventing one
from the wall clock would produce a column that downstream trusts for
cross-camera alignment and that is wrong by the network's jitter.

Config (environment / node.env, loaded via calibration_config):
    SMARTROOM_REOLINK_HOST      NVR address
    SMARTROOM_REOLINK_USER      NVR username
    SMARTROOM_REOLINK_PASS      NVR password
    SMARTROOM_REOLINK_CHANNELS  channels to record (default "1,2,3,4")
    SMARTROOM_REOLINK_STREAM    "main" (default) or "sub"
    SMARTROOM_REOLINK_PATH      URL path template (default Reolink's own)
    SMARTROOM_REOLINK_AUDIO_CH  channel whose mic to keep (default 1; 0 = none)
    SMARTROOM_UPLOAD_DEST       user@host:/abs/recordings/root for --upload

AUDIO. Every camera offers an aac 16kHz mono track, but only channel 1's mic
actually carries signal -- measured over a real recording, ch1 is -40.9 dB mean
with -14.9 dB peaks while 2, 3 and 4 sit at a flat -91.0 dB with mean equal to
peak, which is digital silence rather than a quiet room. Only the configured
channel keeps its track: copying the others would leave three clips that look
like they have sound to anything checking for an audio stream rather than for
signal. Point SMARTROOM_REOLINK_AUDIO_CH somewhere else if a different camera's
mic is enabled later.

The password is read from the environment and never printed; URLs are redacted
in every log line. It does still reach ffmpeg's argv, which is visible in the
process list on a shared machine -- run this somewhere you trust.

    python reolink_capture.py --probe                # auth + stream check only
    python reolink_capture.py --duration 30
    python reolink_capture.py --duration 30 --upload
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.parse
from pathlib import Path

import calibration_config as cfg

PROJECT_ROOT = cfg.PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
CALIBRATION_DIR = cfg.DEFAULT_OUT

# Reolink's own scheme, verified against this NVR: zero-padded channel, NO "ch"
# prefix (h264Preview_ch01_main is a 404 here). The "h264" in the name is part of
# the path, not a promise -- this NVR answers it with HEVC on the main stream.
DEFAULT_PATH_TEMPLATE = "h264Preview_{ch:02d}_{stream}"
DEFAULT_CHANNELS = "1,2,3,4"
DEFAULT_DEST = "intern26@172.16.60.239:/mnt/data4/intern26/recordings"


def redact_text(text: str) -> str:
    """Strip credentials from any RTSP URL inside free text.

    ffmpeg and ffprobe echo the URL they were given -- password included -- in
    their own error messages, so anything relayed from their stderr has to go
    through this or the first failed connection prints the NVR password.
    """
    return re.sub(r"(rtsp://[^:/@\s]+:)[^@\s]*@", r"\1***@", text or "")


def redact(url: str) -> str:
    """rtsp://user:pass@host/... -> rtsp://user:***@host/... for logging."""
    try:
        p = urllib.parse.urlsplit(url)
        if p.password is None:
            return url
        host = p.hostname or ""
        if p.port:
            host = f"{host}:{p.port}"
        return urllib.parse.urlunsplit(
            (p.scheme, f"{p.username}:***@{host}", p.path, p.query, p.fragment))
    except ValueError:
        return "<unparseable url>"


def rtsp_url(channel: int, stream: str) -> str:
    host = os.environ.get("SMARTROOM_REOLINK_HOST", "").strip()
    user = os.environ.get("SMARTROOM_REOLINK_USER", "").strip()
    password = os.environ.get("SMARTROOM_REOLINK_PASS", "")
    if not host or not user:
        raise SystemExit(
            "ERROR: set SMARTROOM_REOLINK_HOST and SMARTROOM_REOLINK_USER (node.env or the "
            "environment). The password goes in SMARTROOM_REOLINK_PASS; node.env is gitignored.")
    template = os.environ.get("SMARTROOM_REOLINK_PATH", DEFAULT_PATH_TEMPLATE)
    path = template.format(ch=channel, stream=stream)
    # Credentials are quoted: a password with @ or / silently corrupts the URL.
    cred = f"{urllib.parse.quote(user, safe='')}:{urllib.parse.quote(password, safe='')}"
    return f"rtsp://{cred}@{host}:554/{path}"


def channels_from_env(explicit=None):
    raw = explicit or os.environ.get("SMARTROOM_REOLINK_CHANNELS", DEFAULT_CHANNELS)
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


NODE_DIR = "reolink"


def camera_id(channel: int) -> str:
    """Calibration key for a channel: reolink_calibrate.py's --id-prefix + folder."""
    return f"reolink-camera{channel}"


def stream_stem(channel: int) -> str:
    """Clip stem, following the RealSense convention: one node directory holding
    several camera_<sensor>_color clips, addressed as <node>-<sensor>.

    The stem must be UNIQUE per camera, which is why these are not all
    camera_main: live_infer's find_calib_clips() locates a camera's calibration
    by globbing <cam_key>.mp4 across the archive, so four cameras sharing one
    stem would each adopt whichever of the four was calibrated last.
    """
    return f"camera_cam{channel}_color"


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def load_calibration(channel: int):
    """Intrinsics for embedding, mirroring capture.py's shape (no bookkeeping keys)."""
    cal = load_json(CALIBRATION_DIR / f"{camera_id(channel)}.json")
    if not cal:
        return None
    keys = ("camera_matrix", "dist_coeffs", "image_size", "rms", "pattern", "calibrated_at")
    return {k: cal[k] for k in keys if k in cal}


def load_extrinsics(channel: int):
    ext = load_json(CALIBRATION_DIR / f"{camera_id(channel)}.extrinsics.json")
    if not ext:
        return None
    keys = ("camera_id", "frame", "tag", "rotation_cam_to_room", "camera_position_mm",
            "reprojection_error_px", "levelled", "calibrated_at")
    return {k: ext[k] for k in keys if k in ext}


def room_frame_info():
    """Same room-frame facts capture.py embeds, from the shared config."""
    ref_id = cfg.tag_id()
    height = cfg.tag_height_mm(ref_id)
    return {
        "reference_tag": {
            "family": cfg.TAG_FAMILY,
            "id": ref_id,
            "size_mm": cfg.tag_sizes().get(ref_id, cfg.tag_size_mm()),
        },
        "definition": "origin=tag center, X=tag right, Y=DOWN (up is -Y), Z=out of tag; units mm",
        "tag_center_above_floor_mm": height,
        "floor_plane": (f"y = {-height:.0f} mm" if height is not None else None),
    }


def make_recording_dir(now: dt.datetime) -> Path:
    """day_NN_YYYY-MM-DD/rec_YYYYMMDD_HHMMSS — see the docstring on why it is a
    timestamp and not capture.py's per-day counter."""
    date = now.strftime("%Y-%m-%d")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_days = sorted(DATA_DIR.glob("day_*"))
    day_dir = next((d for d in existing_days if d.name.endswith(date)), None)
    if day_dir is None:
        day_dir = DATA_DIR / f"day_{len(existing_days) + 1:02d}_{date}"
    rec_dir = day_dir / f"rec_{now.strftime('%Y%m%d_%H%M%S')}"
    rec_dir.mkdir(parents=True, exist_ok=True)
    return rec_dir


def probe_audio_props(mp4_path: Path):
    """(codec, sample_rate, channels) of the first audio track, or None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=codec_name,sample_rate,channels", "-of", "default=nw=1:nk=1",
             str(mp4_path)], capture_output=True, text=True, timeout=60).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    if len(out) < 3:
        return None
    try:
        return {"codec": out[0], "sample_rate": int(out[1]), "channels": int(out[2])}
    except ValueError:
        return None


def probe_video_props(mp4_path: Path):
    """(codec, width, height) of a finished file, or (None, None, None).

    Read, never assumed: this NVR serves HEVC on a path called h264Preview_*,
    so a hardcoded "h264" in metadata.json would be a lie that downstream has no
    way to catch.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,width,height", "-of", "default=nw=1:nk=1", str(mp4_path)],
            capture_output=True, text=True, timeout=60).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None, None, None
    if len(out) < 3:
        return None, None, None
    try:
        return out[0], int(out[1]), int(out[2])
    except ValueError:
        return out[0], None, None


def ffprobe_frame_times(mp4_path: Path):
    """Each encoded frame's real presentation time, from the finished file.

    RTSP delivers what the network delivers, so a nominal-fps grid would be
    fiction; this reads what actually landed (same approach as capture.py)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(mp4_path)],
            capture_output=True, text=True, timeout=300).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    times = []
    for line in out.splitlines():
        line = line.strip().rstrip(",")
        if not line or line == "N/A":
            continue
        try:
            times.append(float(line))
        except ValueError:
            continue
    return times


def write_timestamps(csv_path: Path, times) -> int:
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_seconds"])
        for i, t in enumerate(times):
            writer.writerow([i, f"{t:.6f}"])
    return len(times)


def probe_stream(channel: int, stream: str, timeout_s: int = 20):
    """(ok, note) — does this channel authenticate and carry video?"""
    url = rtsp_url(channel, stream)
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-rtsp_transport", "tcp",
             "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,width,height,avg_frame_rate",
             "-of", "default=nw=1:nk=1", "-i", url],
            capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, "timed out"
    except OSError as exc:
        return False, f"ffprobe not runnable: {exc}"
    if proc.returncode != 0:
        first = (proc.stderr or "").strip().splitlines()
        return False, redact_text(first[-1] if first else f"exit {proc.returncode}")
    return True, " ".join((proc.stdout or "").split())


def record_channels(channels, stream, duration, rec_dir: Path, transport="tcp",
                    audio_ch=None):
    """Start one ffmpeg per channel, run them concurrently, wait for all.

    Stream-copy, not re-encode: the NVR already sends h264 and four 4K decodes
    would not keep up on a laptop.

    Only `audio_ch` keeps its audio track. Every camera on this NVR offers one,
    but measured over a real recording only channel 1 carries signal (-40.9 dB
    mean, -14.9 dB peak); 2, 3 and 4 are flat digital silence at -91.0 dB, mean
    equal to peak. Copying those is worse than dropping them: the clip then
    looks like it has sound, and anything downstream that checks for an audio
    stream rather than for signal will believe it.
    """
    procs = {}
    cam_dir = rec_dir / "streams" / NODE_DIR
    cam_dir.mkdir(parents=True, exist_ok=True)
    for ch in channels:
        out = cam_dir / f"{stream_stem(ch)}.mp4"
        url = rtsp_url(ch, stream)
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
               "-rtsp_transport", transport, "-i", url, "-t", str(duration)]
        cmd += ["-c", "copy"] if ch == audio_ch else ["-an", "-c:v", "copy"]
        cmd += ["-movflags", "+faststart", str(out)]
        print(f"  ch{ch:02d} -> {out.relative_to(rec_dir)}  ({redact(url)})", file=sys.stderr)
        procs[ch] = (subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE), out)

    results = {}
    for ch, (proc, out) in procs.items():
        # Generous margin over -t: ffmpeg still has to flush and finalise the mp4.
        try:
            _, err = proc.communicate(timeout=duration + 120)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, err = proc.communicate()
            results[ch] = (out, False, "ffmpeg overran its deadline")
            continue
        ok = proc.returncode == 0 and out.exists() and out.stat().st_size > 0
        note = "" if ok else redact_text(
            ((err or b"").decode(errors="replace").strip().splitlines() or [""])[-1])
        results[ch] = (out, ok, note)
    return results


def upload(rec_dir: Path, dest: str):
    """Copy the finished recording up. The layout is already final, so this is a
    straight copy of the rec dir into <dest>/<day>/ -- no restructuring like
    upload_recording.sh does (that script folds ONE camera into a node dir).

    scp, not rsync: the host that can reach the NVR is a Windows laptop and rsync
    is not there.
    """
    if ":" not in dest:
        print(f"ERROR: --upload dest wants user@host:/path, got {dest!r}", file=sys.stderr)
        return False
    host, root = dest.split(":", 1)
    day = rec_dir.parent.name
    # day_NN is a per-node counter, and this uploads into a tree other nodes also
    # write to. Our NN reflects how many days THIS host has recorded, which for a
    # new capture host is 01 while the shared tree is already on 20 -- and that
    # created a second folder for the same date, splitting one day in two. The
    # date is the real key, so adopt whatever the destination already calls it.
    date = day.split("_", 2)[-1] if day.startswith("day_") else ""
    if date:
        probe = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host,
             f"ls -d '{root}'/day_*_{date} 2>/dev/null | head -1"],
            capture_output=True, text=True)
        existing = probe.stdout.strip().rsplit("/", 1)[-1]
        if existing and existing != day:
            print(f"note: destination already calls {date} '{existing}' — using that",
                  file=sys.stderr)
            day = existing
    if not shutil.which("scp"):
        print("ERROR: scp not on PATH", file=sys.stderr)
        return False
    mk = subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"mkdir -p '{root}/{day}'"],
                        capture_output=True, text=True)
    if mk.returncode != 0:
        print(f"ERROR: could not create {root}/{day}: {mk.stderr.strip()}", file=sys.stderr)
        return False
    cp = subprocess.run(["scp", "-o", "BatchMode=yes", "-r", str(rec_dir),
                         f"{host}:{root}/{day}/"], capture_output=True, text=True)
    if cp.returncode != 0:
        print(f"ERROR: upload failed: {cp.stderr.strip()}", file=sys.stderr)
        return False
    print(f"uploaded -> {host}:{root}/{day}/{rec_dir.name}", file=sys.stderr)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--duration", type=int, default=30, help="seconds to record (default 30)")
    ap.add_argument("--channels", default=None, help="NVR channels, e.g. '1,2,3,4'")
    ap.add_argument("--stream", default=None, choices=["main", "sub"], help="NVR stream")
    ap.add_argument("--transport", default="tcp", choices=["tcp", "udp"], help="RTSP transport")
    ap.add_argument("--audio-from", type=int, default=None,
                    help="channel whose mic to keep (0 = no audio at all; "
                         "default: SMARTROOM_REOLINK_AUDIO_CH, else 1)")
    ap.add_argument("--probe", action="store_true", help="check auth/streams and exit")
    ap.add_argument("--upload", action="store_true", help="copy the recording to the analysis volume")
    ap.add_argument("--dest", default=None, help=f"upload target (default {DEFAULT_DEST})")
    args = ap.parse_args(argv)

    cfg.load_node_env()
    channels = channels_from_env(args.channels)
    stream = args.stream or os.environ.get("SMARTROOM_REOLINK_STREAM", "main")
    audio_ch = (args.audio_from if args.audio_from is not None
                else int(os.environ.get("SMARTROOM_REOLINK_AUDIO_CH", "1") or 0))
    audio_ch = audio_ch or None
    if audio_ch and audio_ch not in channels:
        print(f"NOTE: audio channel {audio_ch} is not being recorded, so this clip has no sound",
              file=sys.stderr)
        audio_ch = None
    if not channels:
        print("ERROR: no channels selected", file=sys.stderr)
        return 1

    if args.probe:
        bad = 0
        for ch in channels:
            ok, note = probe_stream(ch, stream)
            print(f"ch{ch:02d} [{camera_id(ch)}]: {'OK  ' + note if ok else 'FAIL ' + note}",
                  file=sys.stderr)
            bad += 0 if ok else 1
        missing = [c for c in channels if load_calibration(c) is None]
        if missing:
            print(f"NOTE: no calibration for channel(s) {missing} — run reolink_calibrate.py, "
                  f"or those clips upload uncalibrated and the mirror cannot place them.",
                  file=sys.stderr)
        return 1 if bad else 0

    start = dt.datetime.now().astimezone()
    rec_dir = make_recording_dir(start)
    print(f"Recording {args.duration}s from {len(channels)} camera(s) -> {rec_dir}", file=sys.stderr)
    results = record_channels(channels, stream, args.duration, rec_dir, args.transport,
                              audio_ch=audio_ch)
    end = dt.datetime.now().astimezone()

    cam_dir = rec_dir / "streams" / NODE_DIR
    streams = {}
    for ch in channels:
        out, ok, note = results[ch]
        stem = stream_stem(ch)
        if not ok:
            print(f"  ch{ch:02d}: FAILED — {note}", file=sys.stderr)
            continue
        times = ffprobe_frame_times(out)
        frame_count = write_timestamps(cam_dir / f"{stem}_timestamps.csv", times)
        fps = (frame_count / args.duration) if args.duration else 0.0

        codec, width, height = probe_video_props(out)
        entry = {
            "modality": "video",
            "path": f"{stem}.mp4",
            "codec": codec,
            "device": f"reolink-nvr:ch{ch:02d}:{stream}",
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "timestamps_path": f"{stem}_timestamps.csv",
            # No hardware clock over RTSP — see the docstring.
            "hw_timestamp_domain": None,
        }
        if width and height:
            entry["resolution"] = [width, height]
        audio = probe_audio_props(out) if ch == audio_ch else None
        if audio:
            # Recorded, not assumed: only this channel's mic carries signal, and
            # metadata that claims audio on a silent track is worse than none.
            entry["audio"] = audio
        cal = load_calibration(ch)
        if cal:
            entry["calibration"] = cal
            # Calibration is at the still's resolution; the recorded stream may
            # differ (sub is 640x360 against a 3840x2160 calibration). Same 16:9,
            # so a uniform scale applies — downstream scales by resolution, which
            # is why the REAL one is recorded above rather than the calibration's.
            entry.setdefault("resolution", cal.get("image_size"))
        ext = load_extrinsics(ch)
        if ext:
            entry["extrinsics"] = ext
        streams[stem] = entry
        flag = "" if cal and ext else "  (UNCALIBRATED — mirror cannot place it)"
        snd = f", audio {audio['codec']} {audio['sample_rate']}Hz" if audio else ""
        print(f"  ch{ch:02d}: {frame_count} frames, {fps:.1f} fps{snd}{flag}", file=sys.stderr)

    if not streams:
        print("no camera recorded successfully", file=sys.stderr)
        return 1

    # One metadata.json for the node directory carrying every camera, exactly as
    # a RealSense node writes camera_d455_color + camera_d435_color side by side.
    metadata = {
        "recording_id": rec_dir.name,
        "node": socket.gethostname(),
        "space": "smart_room_1",
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_seconds": args.duration,
        "schema_version": "0.1",
        "streams": streams,
        "room_frame": room_frame_info(),
    }
    tags = load_json(CALIBRATION_DIR / cfg.TAGS_FILENAME)
    if tags is not None:
        metadata["room_tags"] = tags
    # metadata.json LAST: its presence is what marks a recording finished.
    (cam_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"wrote {len(streams)}/{len(channels)} camera(s) -> {rec_dir}", file=sys.stderr)

    if args.upload:
        dest = args.dest or os.environ.get("SMARTROOM_UPLOAD_DEST", DEFAULT_DEST)
        if not upload(rec_dir, dest):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
