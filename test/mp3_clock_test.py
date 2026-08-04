#!/usr/bin/env python3
"""The mp3 frame clock used by reolink_av_forward.py.

Audio position is turned into time by counting MPEG frames (1152 samples = 36ms
at 32kHz). What this pins down: whole frames only, a partial tail left for the
next read, and agreement with the file's real duration to within the encoder's
constant priming delay.

    python test/mp3_clock_test.py
"""
import subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, "/home/alawn/Code/CityOSNode")
from reolink_av_forward import MP3_SAMPLES_PER_FRAME, mp3_frame_count

fails=[]
def check(n, c, d=""):
    print(("ok  " if c else "FAIL ")+n+("  "+str(d) if d else ""))
    if not c: fails.append(n)

with tempfile.TemporaryDirectory() as td:
    p = Path(td)/"a.mp3"
    RATE, DUR = 32000, 12.0
    subprocess.run(["ffmpeg","-v","error","-y","-f","lavfi","-i",f"sine=f=440:r={RATE}:d={DUR}",
                    "-c:a","libmp3lame","-b:a","48k","-ar",str(RATE),"-ac","1",str(p)],check=True)
    raw = p.read_bytes()
    n, used = mp3_frame_count(raw)
    ms = n * MP3_SAMPLES_PER_FRAME * 1000.0 / RATE
    dur = float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                                "-of","csv=p=0",str(p)],capture_output=True,text=True,check=True).stdout)
    print(f"  {n} frames, {used}/{len(raw)} bytes consumed, media {ms/1000:.3f}s, ffprobe {dur:.3f}s")
    check("counts whole frames and consumes nearly the whole file",
          used > len(raw) - 400, f"{len(raw)-used} bytes left")
    # libmp3lame primes its encoder, so the frame clock reads ~90ms long. Measured
    # at 6/12/30/60/120s it is CONSTANT, which is the property that matters: a
    # fixed offset is correctable, a drifting one is not. 150ms of headroom here
    # so the check is about "constant and small", not about a magic 96.
    check("frame count is within a constant ~90ms of the real duration",
          0.0 < (ms/1000 - dur) < 0.15, f"{ms/1000:.3f}s vs {dur:.3f}s")
    # the byte-rate shortcut it replaces, for the record
    byte_est = len(raw) / (48000/8)
    print(f"  bytes/nominal-rate would say {byte_est:.3f}s — off by {abs(byte_est-dur)*1000:.0f}ms")
    check("and it beats the byte-rate estimate it replaced",
          abs(ms/1000 - dur) < abs(byte_est - dur), f"{abs(ms/1000-dur)*1000:.0f}ms vs {abs(byte_est-dur)*1000:.0f}ms")
    # partial frame at the tail must not be miscounted
    n2, used2 = mp3_frame_count(raw[:used-100])
    check("a partial trailing frame is left for the next read", n2 == n-1, f"{n2} vs {n}")

print()
print("FAILED: "+", ".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
