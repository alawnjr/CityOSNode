#!/usr/bin/env python3
"""
Live AUDIO forwarder for the Reolink NVR (RTSP -> live_infer's audio relay).

reolink_live_forward.py pushes this NVR's pictures to the inference server. This
pushes its sound, from ONE camera, over the same kind of one-way outbound
connection and for the same reason: the quad server has no route to the NVR, so
frames and samples have to be PUSHED from the host that can see it.

    ffmpeg (RTSP, audio only) -> mp3 -> TCP -> server:8010 /audio

    POST /audio HTTP/1.1                    (once, then the socket stays open)
    [4B big-endian length][8B big-endian double ts_ms][encoded audio]   repeated

ONE CAMERA, NOT A MIX. Every camera on this NVR offers an aac 16kHz mono track,
but only channel 1's microphone is actually enabled: measured over a real
recording, ch1 is -40.9 dB mean with -14.9 dB peaks while 2, 3 and 4 sit at a flat
-91.0 dB with mean equal to peak, which is digital silence rather than a quiet
room. Mixing four sources to add three silences would only cost bandwidth. The
RealSense cameras have no microphone at all, so channel 1 is the room's only
voice. --channel moves it if a different mic is enabled later.

MP3 AT 32kHz, transcoded from the NVR's aac. Two deliberate choices:
  * mp3, because a continuous mp3 stream is the one thing every browser will play
    from a plain <audio src=...> with no player library, no container timing and no
    MSE plumbing.
  * 32kHz rather than the source's 16kHz, because 32/44.1/48 are the MPEG-1 rates
    and are handled everywhere, while 16kHz falls into MPEG-2 LSF, which some
    decoders treat as an oddity. Upsampling adds no information and a few kbit/s;
    it buys not having to debug a browser that plays nothing.
Transcoding costs almost nothing at one mono channel, and it happens HERE rather
than on the server so the server only ever relays bytes.

TIMING is not this script's problem. The server holds the audio back to match the
video it is already delaying, using the same measured per-camera offsets, so the
only knob is SMARTROOM_AUDIO_TRIM_MS on the server -- set by ear, because a light
switch makes no sound and the calibration cannot measure an audio path.

The password is read from the environment and never printed; URLs are redacted in
every log line. It does still reach ffmpeg's argv, which is visible in the process
list on a shared machine -- run this somewhere you trust.

Config (environment / node.env), shared with the other reolink scripts:
    SMARTROOM_REOLINK_HOST / _USER / _PASS
    SMARTROOM_REOLINK_AUDIO_CH      channel whose mic to forward (default 1)
    SMARTROOM_REOLINK_LIVE_STREAM   "sub" (default) or "main"
    SMARTROOM_LIVE_SERVER           host:port of live_infer (default 172.16.60.239:8010)

    python reolink_audio_forward.py                 # channel from the environment
    python reolink_audio_forward.py --channel 1
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
from reolink_capture import redact, redact_text, rtsp_url

DEFAULT_SERVER = "172.16.60.239:8010"
RECONNECT_S = 3.0
# ~170ms of audio at 48kbps. Small enough that the server's delay buffer can place
# it accurately, large enough not to send thousands of tiny writes per minute.
CHUNK = 1024
CONTENT_TYPE = "audio/mpeg"


def ffmpeg_for(channel, stream, bitrate, rate):
    url = rtsp_url(channel, stream)
    return url, [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        # Freshness over completeness: this is live sound, and a late sample is
        # worth less than a current one.
        "-fflags", "nobuffer", "-flags", "low_delay",
        "-i", url,
        "-vn",                       # audio only; the pictures have their own forwarder
        "-c:a", "libmp3lame", "-b:a", bitrate, "-ar", str(rate), "-ac", "1",
        # Write a bare mp3 elementary stream: no container, so the server can cut it
        # anywhere and a browser can start listening mid-stream.
        "-f", "mp3", "pipe:1",
    ]


def forward_once(channel, stream, srv_host, srv_port, bitrate, rate, stop):
    url, cmd = ffmpeg_for(channel, stream, bitrate, rate)
    print(f"[audio ch{channel:02d}] opening {redact(url)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=0)
    sock = None
    try:
        sock = socket.create_connection((srv_host, srv_port), timeout=15)
        req = (f"POST /audio HTTP/1.1\r\n"
               f"Host: {srv_host}:{srv_port}\r\n"
               f"Content-Type: application/octet-stream\r\n"
               f"X-Audio-Content-Type: {CONTENT_TYPE}\r\n"
               f"Connection: keep-alive\r\n\r\n")
        sock.sendall(req.encode())
        print(f"[audio ch{channel:02d}] connected to {srv_host}:{srv_port}", flush=True)

        sent, t0 = 0, time.time()
        while not stop.is_set():
            data = proc.stdout.read(CHUNK)
            if not data:
                break
            sock.sendall(struct.pack(">Id", len(data), time.time() * 1000.0) + data)
            sent += len(data)
            if sent // (256 * 1024) != (sent - len(data)) // (256 * 1024):
                kbps = sent * 8 / 1000.0 / max(time.time() - t0, 1e-6)
                print(f"[audio ch{channel:02d}] forwarded {sent / 1024:.0f} KB "
                      f"({kbps:.0f} kbps)", flush=True)
        if not stop.is_set():
            err = (proc.stderr.read() or b"").decode(errors="replace").strip()
            last = (err.splitlines() or [""])[-1]
            raise ConnectionError(redact_text(last) or "audio stream ended")
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", type=int, default=None,
                    help="NVR channel whose mic to forward "
                         "(default: SMARTROOM_REOLINK_AUDIO_CH, else 1)")
    ap.add_argument("--stream", default=None, choices=["sub", "main"],
                    help="NVR stream carrying the audio (default sub)")
    ap.add_argument("--server", default=None,
                    help=f"live_infer host:port (default {DEFAULT_SERVER})")
    ap.add_argument("--bitrate", default="48k", help="mp3 bitrate (default 48k)")
    ap.add_argument("--rate", type=int, default=32000,
                    help="mp3 sample rate; an MPEG-1 rate for browser support")
    args = ap.parse_args(argv)

    cfg.load_node_env()
    channel = (args.channel if args.channel is not None
               else int(os.environ.get("SMARTROOM_REOLINK_AUDIO_CH", "1") or 1))
    if not channel:
        print("ERROR: no audio channel configured (SMARTROOM_REOLINK_AUDIO_CH=0)",
              file=sys.stderr)
        return 1
    stream = args.stream or os.environ.get("SMARTROOM_REOLINK_LIVE_STREAM", "sub")
    server = args.server or os.environ.get("SMARTROOM_LIVE_SERVER", DEFAULT_SERVER)
    if ":" not in server:
        print(f"ERROR: --server wants host:port, got {server!r}", file=sys.stderr)
        return 1
    srv_host, srv_port = server.rsplit(":", 1)
    srv_port = int(srv_port)

    print(f"forwarding ch{channel:02d} audio ({stream} stream, mp3 {args.bitrate} "
          f"@ {args.rate}Hz mono) -> {srv_host}:{srv_port}", flush=True)
    stop = threading.Event()
    try:
        while not stop.is_set():
            try:
                forward_once(channel, stream, srv_host, srv_port,
                             args.bitrate, args.rate, stop)
            except Exception as exc:  # noqa: BLE001 — any failure is a reconnect
                if stop.is_set():
                    break
                print(f"[audio ch{channel:02d}] {redact_text(str(exc))} — "
                      f"retrying in {RECONNECT_S:.0f}s", flush=True)
            if not stop.is_set():
                time.sleep(RECONNECT_S)
    except KeyboardInterrupt:
        print("stopping...", flush=True)
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
