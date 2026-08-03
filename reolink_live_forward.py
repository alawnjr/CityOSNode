#!/usr/bin/env python3
"""
Live frame forwarder for the Reolink NVR cameras (RTSP -> live_infer ingest).

live_forward.py does this for a RealSense on a Pi, reading the depth page's
MJPEG. Same destination and same wire protocol here, different source: these
cameras speak RTSP, and the host that can reach them is not the host running
the inference.

WHY A FORWARDER AT ALL. The quad server cannot reach the NVR -- no route from
172.16.60.0/24 to the camera network, ping and TCP/554 both fail -- exactly the
one-way network live_forward.py was written for. So frames are PUSHED: this runs
where the cameras are visible and opens an outbound connection to the server.

    ffmpeg (RTSP, sub stream) -> JPEG frames -> TCP -> server:8010 /ingest?cam=...

    POST /ingest?cam=<key> HTTP/1.1        (once, then the socket stays open)
    [4B big-endian length][8B big-endian double hw_ts_ms][JPEG]   repeated

THE SUB STREAM, not main. Live inference wants small frames at a steady rate,
not 4K: the recorder already keeps full resolution on disk, and four extra 4K
decodes would land on a box already running pose for the depth cameras. Override
with --stream main if you specifically want it.

TIMESTAMPS ARE RECEIVE TIMES, and that is a real limitation. The RealSense path
sends librealsense global time -- sensor mid-exposure, mapped to the host clock
-- which is what makes ±17ms cross-camera pairing meaningful. RTSP carries no
such clock, so what goes on the wire here is when THIS host finished decoding
the frame: the same units (ms since the epoch, so nothing downstream has to
special-case it) but late by the NVR's encode and buffer latency plus the
network, which for a network camera is tens to hundreds of ms and not constant.
Good enough to order one camera's own frames; NOT good enough to fuse a Reolink
person with a RealSense person by timestamp. Sending 0.0 instead would be more
honest still, but it reads as "no clock at all" and loses the per-camera
ordering that does work.

CROSS-CAMERA FUSION DOES NOT USE THIS COLUMN. The server stamps its own arrival
time for every camera -- one clock instead of this host's, the Pi's and
librealsense's three -- and subtracts a per-camera delay measured by its lights
on/off calibration (smartroom-control's detect/timing_sync.py). That is what
puts a Reolink detection and a RealSense detection on a comparable timeline; the
hw_ts sent here is carried through to recordings as a raw value and nothing
compares it across cameras. Nothing on this side needs to change for it, but do
re-run that calibration after anything that alters the path: a different NVR
stream, a change of forwarding host, or a new --fps.

A camera the server has no calibration for is SKIPPED by live_infer entirely
(it needs an uploaded clip carrying calibration+extrinsics to build room
geometry), so run reolink_capture.py --upload once before expecting a camera to
appear live.

Config (environment / node.env), shared with reolink_capture.py:
    SMARTROOM_REOLINK_HOST / _USER / _PASS / _CHANNELS
    SMARTROOM_REOLINK_LIVE_STREAM   "sub" (default) or "main"
    SMARTROOM_LIVE_SERVER           host:port of live_infer (default 172.16.60.239:8010)

    python reolink_live_forward.py                 # all configured channels
    python reolink_live_forward.py --channels 1,2  # a subset
"""

import argparse
import os
import socket
import struct
import subprocess
import sys
import threading
import time

import calibration_config as cfg
from reolink_capture import (channels_from_env, drain_stderr, redact, redact_text,
                             rtsp_url, stream_stem)

DEFAULT_SERVER = "172.16.60.239:8010"
RECONNECT_S = 3.0
SOI = b"\xff\xd8"  # JPEG start of image
EOI = b"\xff\xd9"  # JPEG end of image
READ_CHUNK = 65536
MAX_BUFFER = 24 * 1024 * 1024  # a stalled reader must not grow without bound


def jpeg_stream(proc, stop):
    """Yield complete JPEGs from ffmpeg's mjpeg stdout.

    ffmpeg emits frames back to back with no container, so frames are split on
    the JPEG markers themselves rather than by any length prefix.
    """
    buf = bytearray()
    while not stop.is_set():
        chunk = proc.stdout.read(READ_CHUNK)
        if not chunk:
            return
        buf.extend(chunk)
        while True:
            start = buf.find(SOI)
            if start < 0:
                # No frame started yet; drop everything but a possible split marker.
                if len(buf) > 1:
                    del buf[:-1]
                break
            end = buf.find(EOI, start + 2)
            if end < 0:
                if start:
                    del buf[:start]
                break
            frame = bytes(buf[start:end + 2])
            del buf[:end + 2]
            yield frame
        if len(buf) > MAX_BUFFER:
            # Never seen a valid frame in 24MB — the source is not mjpeg.
            raise ValueError("no JPEG found in the stream; is ffmpeg producing mjpeg?")


def ffmpeg_for(channel, stream, fps, quality):
    url = rtsp_url(channel, stream)
    return url, [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        # Prefer freshness over completeness: this is a live view, so a late
        # frame is worth less than a current one.
        "-fflags", "nobuffer", "-flags", "low_delay",
        "-i", url,
        "-an", "-f", "image2pipe", "-vcodec", "mjpeg",
        "-q:v", str(quality), "-r", str(fps), "pipe:1",
    ]


def forward_once(channel, cam_key, stream, srv_host, srv_port, fps, quality, stop):
    url, cmd = ffmpeg_for(channel, stream, fps, quality)
    print(f"[ch{channel:02d}] opening {redact(url)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=0)
    # Quiet on this source today, but the same undrained-pipe deadlock that
    # stalled the audio forwarder applies here. See drain_stderr.
    errlines = drain_stderr(proc)
    sock = None
    try:
        sock = socket.create_connection((srv_host, srv_port), timeout=15)
        req = (f"POST /ingest?cam={cam_key} HTTP/1.1\r\n"
               f"Host: {srv_host}:{srv_port}\r\n"
               f"Content-Type: application/octet-stream\r\n"
               f"Connection: keep-alive\r\n\r\n")
        sock.sendall(req.encode())
        print(f"[ch{channel:02d}] connected to {srv_host}:{srv_port} (cam={cam_key})", flush=True)

        n, t0 = 0, time.time()
        for jpeg in jpeg_stream(proc, stop):
            if stop.is_set():
                break
            # Receive time, not exposure time — see the module docstring.
            hw_ts = time.time() * 1000.0
            sock.sendall(struct.pack(">Id", len(jpeg), hw_ts) + jpeg)
            n += 1
            if n % 120 == 0:
                print(f"[ch{channel:02d}] forwarded {n} frames "
                      f"({n / max(time.time() - t0, 1e-6):.1f} fps)", flush=True)
        if not stop.is_set():
            last = errlines[-1] if errlines else ""
            raise ConnectionError(redact_text(last) or "source stream ended")
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def forward_forever(channel, stream, srv_host, srv_port, fps, quality, stop):
    cam_key = stream_stem(channel)
    while not stop.is_set():
        try:
            forward_once(channel, cam_key, stream, srv_host, srv_port, fps, quality, stop)
        except Exception as exc:  # noqa: BLE001 — any failure is a reconnect
            if stop.is_set():
                break
            print(f"[ch{channel:02d}] {redact_text(str(exc))} — retrying in {RECONNECT_S:.0f}s",
                  flush=True)
        if not stop.is_set():
            time.sleep(RECONNECT_S)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", default=None, help="NVR channels, e.g. '1,2,3,4'")
    ap.add_argument("--stream", default=None, choices=["sub", "main"],
                    help="NVR stream (default: sub — see the docstring)")
    ap.add_argument("--server", default=None, help=f"live_infer host:port (default {DEFAULT_SERVER})")
    ap.add_argument("--fps", type=float, default=10.0, help="frames per second to forward")
    ap.add_argument("--quality", type=int, default=6, help="mjpeg quality, 2=best 31=worst")
    args = ap.parse_args(argv)

    cfg.load_node_env()
    channels = channels_from_env(args.channels)
    stream = args.stream or os.environ.get("SMARTROOM_REOLINK_LIVE_STREAM", "sub")
    server = args.server or os.environ.get("SMARTROOM_LIVE_SERVER", DEFAULT_SERVER)
    if ":" not in server:
        print(f"ERROR: --server wants host:port, got {server!r}", file=sys.stderr)
        return 1
    srv_host, srv_port = server.rsplit(":", 1)
    srv_port = int(srv_port)
    if not channels:
        print("ERROR: no channels selected", file=sys.stderr)
        return 1

    print(f"forwarding ch{channels} ({stream} stream) -> {srv_host}:{srv_port} "
          f"as {', '.join(stream_stem(c) for c in channels)}", flush=True)

    stop = threading.Event()
    threads = [threading.Thread(target=forward_forever,
                                args=(c, stream, srv_host, srv_port, args.fps, args.quality, stop),
                                daemon=True, name=f"ch{c:02d}")
               for c in channels]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("stopping...", flush=True)
        stop.set()
        for t in threads:
            t.join(timeout=8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
