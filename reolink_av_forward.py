#!/usr/bin/env python3
"""Combined A/V forwarder for the Reolink NVR cameras — ONE RTSP session.

WHY THIS EXISTS. reolink_live_forward.py and reolink_audio_forward.py each open
their own RTSP session to the SAME camera and each drop the other's stream
(`-an` / `-vn`). The camera hands audio and video over together on one timebase;
splitting them into two processes throws that away, and the server then stamps
each on arrival, so the only thing relating a sound to a picture is how long each
pipeline happened to take. Those pipelines are not comparable — the video side
decodes h264 and re-encodes mjpeg, the audio side just encodes mp3 — which is why
the sound ran ~2.8s ahead of the picture and needed a constant set by ear. A
constant that, being a race between two pipelines, drifts with load.

Here the split happens INSIDE one ffmpeg, so both outputs descend from one input
clock, and each chunk carries a MEDIA time measured from its own position in the
stream rather than from when it happened to arrive.

    ffmpeg (one RTSP session)
      ├── -map 0:v -> mjpeg frames -> pipe:1  -> /ingest?cam=...&media=1
      └── -map 0:a -> mp3          -> pipe:3  -> /audio?media=1

Measured with a synthetic flash+beep source: the split itself introduces 0.000s
of A/V skew and does not drift, against the 2.79s the two-session arrangement
needed. What remains is ~90ms of constant mp3 encoder priming (see below).

TWO THINGS THAT LOOK LIKE DETAILS AND ARE NOT.

  * Video rate control uses the `fps` FILTER, never `-r`. With `-r 10` ffmpeg
    emitted 2 more frames than duration x 10 predicted, so frame_index/fps
    mislabelled every frame by 0.2s. `-vf fps=10` emits exactly the expected
    count. Since the frames go out as a bare JPEG stream with nowhere to put a
    timestamp, index IS the clock, and it has to be an honest one.

  * Audio time comes from counting mp3 FRAMES, not bytes. A frame at 32kHz is
    exactly 1152 samples = 36ms, and counting them gives chunk boundaries that
    land on frame boundaries — which is what lets a chunk be tagged with a time
    at all. (A bytes/second estimate was NOT the drift disaster first assumed:
    measured at 6s/12s/30s/60s/120s, both estimates sit at a CONSTANT +84..104ms
    and neither drifts. Frames are ~8ms closer and exact at the boundaries; the
    claim that bytes drift was wrong.)

  * There is a residual ~90ms of audio lead from libmp3lame's own priming delay,
    constant at every length tested. It is one video frame at 10fps and below the
    ~100ms where lip-sync becomes noticeable, against the 2790ms the two-session
    arrangement needed. SMARTROOM_AUDIO_TRIM_MS can still absorb it if anyone
    finds it audible; it is no longer load-dependent, which is what made the old
    constant untrustworthy.

Config (environment / node.env), shared with the other Reolink tools:
    SMARTROOM_REOLINK_HOST / _USER / _PASS / _CHANNELS
    SMARTROOM_REOLINK_LIVE_STREAM   "sub" (default) or "main"
    SMARTROOM_LIVE_SERVER           host:port of live_infer (default 172.16.60.239:8010)

    python reolink_av_forward.py                    # audio+video, all channels
    python reolink_av_forward.py --channels 1       # one channel
    python reolink_av_forward.py --no-audio         # video only (per channel)

Only ONE channel should carry audio — the room has one microphone — so audio is
sent for --audio-channel (default 1) and suppressed on the rest.
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
SOI = b"\xff\xd8"           # JPEG start of image
EOI = b"\xff\xd9"           # JPEG end of image
READ_CHUNK = 65536
MAX_BUFFER = 24 * 1024 * 1024
AUDIO_FD = 3                # ffmpeg writes the second output to pipe:3

# MPEG-1 Layer III at 32kHz: 1152 samples per frame, so 36ms exactly. This is the
# audio clock — see the module docstring on why bytes are not.
MP3_SAMPLES_PER_FRAME = 1152
MP3_BITRATES = [None, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
MP3_RATES = {0: 44100, 1: 48000, 2: 32000}


def ffmpeg_for(channel, stream, fps, quality, bitrate, rate, with_audio):
    """One input, one or two outputs. See the docstring on `fps` vs `-r`."""
    url = rtsp_url(channel, stream)
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        # Freshness over completeness: this is a live view, so a late frame is
        # worth less than a current one. Applies to BOTH streams now, which is
        # part of the point — the audio used to get different buffering.
        "-fflags", "nobuffer", "-flags", "low_delay",
        "-i", url,
        # --- video output ---
        "-map", "0:v", "-an",
        "-vf", f"fps={fps}",          # the FILTER, not -r; see the docstring
        "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", str(quality), "pipe:1",
    ]
    if with_audio:
        cmd += [
            # --- audio output, same input, same clock ---
            "-map", "0:a", "-vn",
            "-c:a", "libmp3lame", "-b:a", bitrate, "-ar", str(rate), "-ac", "1",
            "-f", "mp3", f"pipe:{AUDIO_FD}",
        ]
    return url, cmd


def jpeg_stream(read, stop):
    """Yield complete JPEGs. Frames are split on the JPEG markers themselves,
    since image2pipe emits them back to back with no container."""
    buf = bytearray()
    while not stop.is_set():
        chunk = read(READ_CHUNK)
        if not chunk:
            return
        buf.extend(chunk)
        while True:
            start = buf.find(SOI)
            if start < 0:
                if len(buf) > 1:
                    del buf[:-1]
                break
            end = buf.find(EOI, start + 2)
            if end < 0:
                if start:
                    del buf[:start]
                break
            yield bytes(buf[start:end + 2])
            del buf[:end + 2]
        if len(buf) > MAX_BUFFER:
            raise ValueError("no JPEG found in 24MB; is ffmpeg producing mjpeg?")


def mp3_frame_count(buf: bytes, start: int = 0):
    """(frames, consumed) — count whole MPEG audio frames from `start`.

    Walks the sync words rather than dividing by a nominal byte rate. The reason is
    chunk BOUNDARIES: a chunk can only be tagged with a time if it ends on a frame
    boundary, and the padding bit makes frames 216 or 217 bytes, so byte
    arithmetic cannot find them. (It is not about drift — measured, the byte
    estimate does not drift either; see the module docstring.)
    """
    n, i = 0, start
    end = len(buf)
    while i + 4 <= end:
        if buf[i] != 0xFF or (buf[i + 1] & 0xE0) != 0xE0:
            i += 1                      # not a sync word; resync
            continue
        ver = (buf[i + 1] >> 3) & 0x03  # 3 = MPEG-1
        bri = (buf[i + 2] >> 4) & 0x0F
        sri = (buf[i + 2] >> 2) & 0x03
        pad = (buf[i + 2] >> 1) & 0x01
        if ver != 3 or bri in (0, 15) or sri == 3:
            i += 1
            continue
        rate = MP3_RATES[sri]
        size = 144 * MP3_BITRATES[bri] * 1000 // rate + pad
        if size <= 4 or i + size > end:
            break                       # partial frame; wait for more bytes
        n += 1
        i += size
    return n, i


class Session:
    """One camera: one ffmpeg, one socket per stream, one shared media clock."""

    def __init__(self, channel, stream, srv, fps, quality, bitrate, rate, with_audio):
        self.channel, self.stream = channel, stream
        self.cam_key = stream_stem(channel)
        self.srv_host, self.srv_port = srv
        self.fps, self.quality = fps, quality
        self.bitrate, self.rate = bitrate, rate
        self.with_audio = with_audio
        self.mp3_frame_ms = MP3_SAMPLES_PER_FRAME * 1000.0 / rate

    def _connect(self, path):
        sock = socket.create_connection((self.srv_host, self.srv_port), timeout=15)
        req = (f"POST {path} HTTP/1.1\r\n"
               f"Host: {self.srv_host}:{self.srv_port}\r\n"
               f"Content-Type: application/octet-stream\r\n"
               f"Connection: keep-alive\r\n\r\n")
        sock.sendall(req.encode())
        return sock

    def _pump_audio(self, rfd, stop, errors):
        """mp3 bytes -> /audio, each chunk tagged with its MEDIA time.

        The tag is the media time of the chunk's FIRST frame, so the server can
        place the sound against the pictures without knowing anything about how
        long either took to arrive.
        """
        sock = None
        try:
            sock = self._connect(f"/audio?cam={self.cam_key}&media=1")
            print(f"[ch{self.channel:02d}] audio connected (media clock)", flush=True)
            pending = bytearray()
            frames = 0
            with os.fdopen(rfd, "rb", buffering=0) as fh:
                while not stop.is_set():
                    data = fh.read(READ_CHUNK)
                    if not data:
                        return
                    pending.extend(data)
                    n, used = mp3_frame_count(bytes(pending))
                    if not n:
                        continue          # not a whole frame yet
                    media_ms = frames * self.mp3_frame_ms
                    payload = bytes(pending[:used])
                    del pending[:used]
                    frames += n
                    sock.sendall(struct.pack(">Idd", len(payload),
                                             time.time() * 1000.0, media_ms) + payload)
        except Exception as exc:  # noqa: BLE001 — surfaced by the supervisor
            if not stop.is_set():
                errors.append(f"audio: {redact_text(str(exc))}")
        finally:
            stop.set()
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    def run_once(self, stop):
        url, cmd = ffmpeg_for(self.channel, self.stream, self.fps, self.quality,
                              self.bitrate, self.rate, self.with_audio)
        print(f"[ch{self.channel:02d}] opening {redact(url)}"
              f"{' +audio' if self.with_audio else ''}", flush=True)

        arfd = awfd = None
        pass_fds = ()
        if self.with_audio:
            arfd, awfd = os.pipe()
            # ffmpeg writes its second output to fd 3, so hand it that fd.
            os.set_inheritable(awfd, True)
            pass_fds = (awfd,)

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                bufsize=0, pass_fds=pass_fds,
                                preexec_fn=(None if awfd is None else
                                            _dup_to(awfd, AUDIO_FD)))
        if awfd is not None:
            os.close(awfd)      # the child holds it now
        # An undrained stderr pipe DEADLOCKS ffmpeg once the OS buffer fills; this
        # source warns continuously. See drain_stderr's own note.
        errlines = drain_stderr(proc)
        errors: list[str] = []
        athread = None
        if arfd is not None:
            athread = threading.Thread(target=self._pump_audio, args=(arfd, stop, errors),
                                       daemon=True, name=f"a{self.channel:02d}")
            athread.start()

        sock = None
        try:
            sock = self._connect(f"/ingest?cam={self.cam_key}&media=1")
            print(f"[ch{self.channel:02d}] video connected to "
                  f"{self.srv_host}:{self.srv_port} (cam={self.cam_key})", flush=True)
            n, t0 = 0, time.time()
            for jpeg in jpeg_stream(proc.stdout.read, stop):
                if stop.is_set():
                    break
                # Media time from the frame INDEX — honest because the fps filter
                # emits exactly one frame per output slot.
                media_ms = n * 1000.0 / self.fps
                sock.sendall(struct.pack(">Idd", len(jpeg),
                                         time.time() * 1000.0, media_ms) + jpeg)
                n += 1
                if n % 120 == 0:
                    print(f"[ch{self.channel:02d}] {n} frames "
                          f"({n / max(time.time() - t0, 1e-6):.1f} fps), "
                          f"media {media_ms / 1000.0:.1f}s", flush=True)
            if not stop.is_set():
                last = errors[0] if errors else (errlines[-1] if errlines else "")
                raise ConnectionError(redact_text(last) or "source stream ended")
        finally:
            stop.set()
            for s in (sock,):
                if s is not None:
                    try:
                        s.close()
                    except OSError:
                        pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            if athread is not None:
                athread.join(timeout=5)

    def run_forever(self, stop_all):
        while not stop_all.is_set():
            stop = threading.Event()
            # A failure on either stream restarts BOTH: they only share a clock
            # because they share an ffmpeg, so reconnecting one alone would put
            # them back on two clocks — the exact thing this file exists to avoid.
            watch = threading.Thread(target=lambda: (stop_all.wait(), stop.set()),
                                     daemon=True)
            watch.start()
            try:
                self.run_once(stop)
            except Exception as exc:  # noqa: BLE001 — any failure is a reconnect
                if stop_all.is_set():
                    break
                print(f"[ch{self.channel:02d}] {redact_text(str(exc))} — "
                      f"retrying in {RECONNECT_S:.0f}s", flush=True)
            if not stop_all.is_set():
                time.sleep(RECONNECT_S)


def _dup_to(src_fd, dst_fd):
    """preexec hook: make the inherited pipe appear as fd `dst_fd` in the child."""
    def hook():
        if src_fd != dst_fd:
            os.dup2(src_fd, dst_fd)
    return hook


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", default=None, help="NVR channels, e.g. '1,2,3,4'")
    ap.add_argument("--stream", default=None, choices=["sub", "main"],
                    help="NVR stream (default: sub)")
    ap.add_argument("--server", default=None, help=f"live_infer host:port (default {DEFAULT_SERVER})")
    ap.add_argument("--fps", type=float, default=10.0, help="frames per second to forward")
    ap.add_argument("--quality", type=int, default=6, help="mjpeg quality, 2=best 31=worst")
    ap.add_argument("--audio-channel", type=int, default=1,
                    help="which channel's microphone to forward (0 = none)")
    ap.add_argument("--no-audio", action="store_true", help="video only")
    ap.add_argument("--bitrate", default="48k", help="mp3 bitrate")
    ap.add_argument("--audio-rate", type=int, default=32000,
                    help="mp3 sample rate; an MPEG-1 rate for browser support")
    args = ap.parse_args(argv)

    cfg.load_node_env()
    channels = channels_from_env(args.channels)
    stream = args.stream or os.environ.get("SMARTROOM_REOLINK_LIVE_STREAM", "sub")
    server = args.server or os.environ.get("SMARTROOM_LIVE_SERVER", DEFAULT_SERVER)
    if ":" not in server:
        print(f"ERROR: --server wants host:port, got {server!r}", file=sys.stderr)
        return 1
    host, port = server.rsplit(":", 1)
    if not channels:
        print("ERROR: no channels selected", file=sys.stderr)
        return 1
    if args.audio_rate not in (32000, 44100, 48000):
        print("ERROR: --audio-rate must be an MPEG-1 rate (32000/44100/48000)",
              file=sys.stderr)
        return 1

    audio_ch = 0 if args.no_audio else args.audio_channel
    if audio_ch and audio_ch not in channels:
        print(f"note: channel {audio_ch} carries the microphone but is not being "
              f"forwarded; no audio will be sent", file=sys.stderr)

    print(f"forwarding ch{channels} ({stream}) -> {host}:{port} as "
          f"{', '.join(stream_stem(c) for c in channels)}; "
          f"audio from {('ch%02d' % audio_ch) if audio_ch else 'nothing'}", flush=True)

    stop_all = threading.Event()
    threads = []
    for c in channels:
        s = Session(c, stream, (host, int(port)), args.fps, args.quality,
                    args.bitrate, args.audio_rate, with_audio=(c == audio_ch))
        t = threading.Thread(target=s.run_forever, args=(stop_all,), daemon=True,
                             name=f"ch{c:02d}")
        threads.append(t)
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("stopping...", flush=True)
        stop_all.set()
        for t in threads:
            t.join(timeout=8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
