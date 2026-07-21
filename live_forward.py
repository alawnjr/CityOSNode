#!/usr/bin/env python3
"""
Live frame forwarder (Pi side of the live-stream feature).

The quad server can't reach the Pi (one-way network), so we PUSH frames to it.
This reads the depth page's existing local MJPEG for one camera
(`http://127.0.0.1:8001/rgb.mjpg?s=<serial>` — already 180-rotated by the view
encoder) and streams each JPEG to the server's live_infer ingest over a single
persistent TCP connection, length-prefixed:
    [4-byte big-endian uint32 length][JPEG bytes]  repeated.

No camera access and no depth-page changes: it just re-serves frames the page
is already producing. Auto-reconnects both ends on failure.

Env / flags:
  --serial     RealSense USB serial to forward (default D455 243122300173)
  --source     local MJPEG url (default http://127.0.0.1:8001/rgb.mjpg)
  --server     server host:port for /ingest (default 172.16.60.239:8010)
  --cam        stream key the server localizes as (default camera_d455_color)

Usage (on the Pi, system python is fine — stdlib only):
  python3 ~/CityOS/live_forward.py
"""

import argparse
import json
import socket
import struct
import sys
import threading
import time
import urllib.request
from urllib.request import urlopen

RECONNECT_S = 3.0
DEPTH_POLL_S = 0.12


def mjpeg_frames(resp):
    """Yield (jpeg_bytes, hw_ts_ms) from a multipart/x-mixed-replace response.
    hw_ts_ms is the sensor timestamp from the depth page's X-Hw-Timestamp-Ms
    part header (librealsense global clock) — the cross-camera sync key; 0.0 if
    the page is an older build that doesn't send it."""
    hw_ts = 0.0
    while True:
        line = resp.readline()
        if not line:
            return
        low = line.strip().lower()
        if low.startswith(b"x-hw-timestamp-ms:"):
            try:
                hw_ts = float(line.split(b":", 1)[1])
            except ValueError:
                hw_ts = 0.0
        elif low.startswith(b"content-length:"):
            clen = int(line.split(b":", 1)[1])
            # consume up to and including the blank line after the part headers
            while True:
                l2 = resp.readline()
                if l2 in (b"\r\n", b"\n", b""):
                    break
            buf = b""
            while len(buf) < clen:
                chunk = resp.read(clen - len(buf))
                if not chunk:
                    return
                buf += chunk
            yield buf, hw_ts
            hw_ts = 0.0


def detect_serial(source_base, cam_key):
    """Serial of the camera behind a stream key (camera_d455_color -> the D455s
    serial), by asking the depth page. Lets a systemd template start one unit
    per camera without hard-coding serials that change when hardware is swapped."""
    want = cam_key.replace("camera_", "").replace("_color", "").lower()   # d455
    with urllib.request.urlopen(source_base + "/devices", timeout=10) as r:
        devices = json.load(r).get("devices", [])
    for d in devices:
        if want in (d.get("name") or "").lower().replace(" ", ""):
            return d["serial"]
    raise SystemExit(f"[fwd] no camera matching {want!r}; saw "
                     + ", ".join(d.get("name", "?") for d in devices))


def depth_channel(value_base, serial, server_base, cam_key):
    """Depth back-channel (the server can't reach us). Poll the server for the
    hip pixels it wants ranged, sample our own /value there (depth aligned to
    color), and POST the metres back. Sparse: only a few points per frame.
    Per-camera: /hips and /depths are keyed by ?cam= so two cameras don't mix."""
    q = f"?cam={cam_key}"
    while True:
        try:
            with urllib.request.urlopen(server_base + "/hips" + q, timeout=5) as r:
                hips = json.load(r).get("hips", [])
            out = []
            for u, v in hips:
                try:
                    url = f"{value_base}/value?s={serial}&x={u:.4f}&y={v:.4f}"
                    with urllib.request.urlopen(url, timeout=3) as r:
                        d = json.load(r)
                    if d.get("ok") and d.get("m"):
                        out.append({"u": u, "v": v, "m": d["m"]})
                except Exception:  # noqa: BLE001
                    pass
            if out:
                req = urllib.request.Request(
                    server_base + "/depths" + q,
                    data=json.dumps(out).encode(),
                    headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5).read()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(DEPTH_POLL_S)


def run_once(source_url, srv_host, srv_port, cam_key):
    print(f"[fwd] opening {source_url}", flush=True)
    resp = urlopen(source_url, timeout=15)
    sock = socket.create_connection((srv_host, srv_port), timeout=15)
    req = (f"POST /ingest?cam={cam_key} HTTP/1.1\r\n"
           f"Host: {srv_host}:{srv_port}\r\n"
           f"Content-Type: application/octet-stream\r\n"
           f"Connection: keep-alive\r\n\r\n")
    sock.sendall(req.encode())
    print(f"[fwd] connected to {srv_host}:{srv_port} (cam={cam_key})", flush=True)
    n = 0
    t0 = time.time()
    for jpeg, hw_ts in mjpeg_frames(resp):
        # wire frame: [4B len][8B double hw_ts_ms][jpeg]
        sock.sendall(struct.pack(">Id", len(jpeg), hw_ts) + jpeg)
        n += 1
        if n % 60 == 0:
            fps = n / (time.time() - t0)
            print(f"[fwd] forwarded {n} frames ({fps:.1f} fps)", flush=True)
    raise ConnectionError("source stream ended")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", default=None,
                    help="camera USB serial; auto-detected from --cam if omitted")
    ap.add_argument("--source", default="http://127.0.0.1:8001/rgb.mjpg")
    ap.add_argument("--server", default="172.16.60.239:8010")
    ap.add_argument("--cam", default="camera_d455_color")
    args = ap.parse_args()

    value_base_early = args.source.rsplit("/", 1)[0]
    serial = args.serial or detect_serial(value_base_early, args.cam)
    print(f"[fwd] {args.cam} -> serial {serial}", flush=True)
    source_url = f"{args.source}?s={serial}&t={int(time.time())}"
    srv_host, srv_port = args.server.split(":")
    srv_port = int(srv_port)

    # depth back-channel: sample /value on the local depth page for hips the
    # server asks about. value_base is the depth page (derived from --source).
    value_base = args.source.rsplit("/", 1)[0]
    server_base = f"http://{args.server}"
    threading.Thread(target=depth_channel,
                     args=(value_base, serial, server_base, args.cam),
                     daemon=True).start()

    while True:
        try:
            run_once(source_url, srv_host, srv_port, args.cam)
        except KeyboardInterrupt:
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[fwd] disconnected: {exc}; retry in {RECONNECT_S}s", flush=True)
            time.sleep(RECONNECT_S)


if __name__ == "__main__":
    sys.exit(main())
