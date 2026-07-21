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
    """Yield JPEG byte blobs from a multipart/x-mixed-replace response that
    sends a Content-Length per part (the depth page does)."""
    while True:
        line = resp.readline()
        if not line:
            return
        low = line.strip().lower()
        if low.startswith(b"content-length:"):
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
            yield buf


def depth_channel(value_base, serial, server_base):
    """Depth back-channel (the server can't reach us). Poll the server for the
    hip pixels it wants ranged, sample our own /value there (D455 depth aligned
    to color), and POST the metres back. Sparse: only a few points per frame."""
    while True:
        try:
            with urllib.request.urlopen(server_base + "/hips", timeout=5) as r:
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
                    server_base + "/depths",
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
    for jpeg in mjpeg_frames(resp):
        sock.sendall(struct.pack(">I", len(jpeg)) + jpeg)
        n += 1
        if n % 60 == 0:
            fps = n / (time.time() - t0)
            print(f"[fwd] forwarded {n} frames ({fps:.1f} fps)", flush=True)
    raise ConnectionError("source stream ended")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", default="243122300173")
    ap.add_argument("--source", default="http://127.0.0.1:8001/rgb.mjpg")
    ap.add_argument("--server", default="172.16.60.239:8010")
    ap.add_argument("--cam", default="camera_d455_color")
    args = ap.parse_args()

    source_url = f"{args.source}?s={args.serial}&t={int(time.time())}"
    srv_host, srv_port = args.server.split(":")
    srv_port = int(srv_port)

    # depth back-channel: sample /value on the local depth page for hips the
    # server asks about. value_base is the depth page (derived from --source).
    value_base = args.source.rsplit("/", 1)[0]
    server_base = f"http://{args.server}"
    threading.Thread(target=depth_channel,
                     args=(value_base, args.serial, server_base),
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
