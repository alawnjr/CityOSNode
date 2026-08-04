#!/usr/bin/env python3
"""Does ONE ffmpeg with TWO outputs preserve audio/video alignment?

The experiment that justifies reolink_av_forward.py. Run it with:
    python test/av_split_probe.py


This is the premise of the combined forwarder: instead of two RTSP sessions whose
relative timing is only "however long each pipeline took", open one session and
split it inside a single ffmpeg. The question this answers is whether the two
output streams can still be placed on a COMMON clock afterwards, given that
image2pipe carries no timestamps and mp3 carries none either.

The claim under test: with a fixed output frame rate and CBR mp3, position in each
stream IS time on the shared input clock — video frame index / fps, and audio byte
offset / bytes-per-second. If ffmpeg normalises the two outputs' start times
independently, that claim is false and the whole approach fails.

Method: build a source that flashes and beeps at exactly the same instants, run it
through one ffmpeg with two outputs, then ask where each marker landed.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

MARKS = [2.0, 5.0, 8.0]      # seconds — flash AND beep at each
DUR = 11.0
FPS = 10                      # the rate the real forwarder uses
RATE = 32000
BITRATE = "48k"

fails = []


def check(name, cond, detail=""):
    print(("ok  " if cond else "FAIL ") + name + ("  " + str(detail) if detail else ""))
    if not cond:
        fails.append(name)


def build_source(path: Path):
    """Black video that flashes white, silence that beeps — at the same instants."""
    on = "+".join(f"between(t,{m},{m + 0.2})" for m in MARKS)
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=320x240:r=30:d={DUR}",
        "-f", "lavfi", "-i", f"sine=f=1000:r={RATE}:d={DUR}",
        # drawbox rather than geq: geq's cb/cr expressions refused the yuv input
        # here, and a full-frame filled box is the same "flash" with one filter.
        "-filter_complex",
        f"[0:v]drawbox=x=0:y=0:w=iw:h=ih:color=white@1:t=fill:enable='{on}'[v];"
        f"[1:a]volume=volume=0:eval=frame:enable='not({on})'[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", str(RATE),
        str(path),
    ], check=True)


def split(src: Path, vdir: Path, apath: Path, vopts=None, fps=None):
    """ONE ffmpeg, ONE input, TWO outputs — the shape the forwarder would use."""
    vdir.mkdir(parents=True, exist_ok=True)
    vopts = ["-r", str(FPS)] if vopts is None else vopts
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(src),
        "-map", "0:v", "-an", "-f", "image2", "-vcodec", "mjpeg", "-q:v", "6",
        *vopts, str(vdir / "%05d.jpg"),
        "-map", "0:a", "-vn", "-c:a", "libmp3lame", "-b:a", BITRATE,
        "-ar", str(RATE), "-ac", "1", "-f", "mp3", str(apath),
    ], check=True)


def flash_frames(vdir: Path):
    """Indices of the mostly-white frames, in ONE pass over the jpg sequence.

    Two things this got wrong first: it ran ffmpeg per file (112 processes), and it
    passed `-v error`, which suppresses the very `metadata=print` lines it parses —
    so it reported "no flashes" on a source that demonstrably flashes.
    """
    r = subprocess.run(
        ["ffmpeg", "-v", "info", "-f", "image2", "-i", str(vdir / "%05d.jpg"),
         "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YAVG",
         "-f", "null", "-"], capture_output=True, text=True)
    out, idx = [], -1
    for line in r.stderr.splitlines():
        # ffmpeg prefixes every filter line with "[Parsed_metadata_1 @ 0x...]", so
        # the frame counter is INSIDE the line, not at its start.
        if " frame:" in line or line.startswith("frame:"):
            idx = int(line.split("frame:")[1].split()[0])
        elif "YAVG" in line and idx >= 0:
            if float(line.split("=")[-1]) > 128:
                out.append(idx)
    return out


def beep_times(apath: Path):
    """Where each beep starts in the mp3, from silencedetect."""
    r = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(apath),
         "-af", "silencedetect=noise=-30dB:d=0.1", "-f", "null", "-"],
        capture_output=True, text=True)
    ends = [float(l.split("silence_end:")[1].split("|")[0].strip())
            for l in r.stderr.splitlines() if "silence_end" in l]
    # silencedetect reports the trailing silence ending at EOF; that is not a beep.
    return [t for t in ends if t < DUR - 0.4]



def source_marks(src: Path):
    """The source's own flash and beep instants, measured with the same tooling.

    The split must be judged against ITS INPUT, not against the nominal numbers:
    any quirk in how the synthetic source was built (which frame a drawbox first
    covers, where an encoder puts a keyframe) would otherwise be charged to the
    split.
    """
    r = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(src), "-map", "0:v",
         "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YAVG",
         "-f", "null", "-"], capture_output=True, text=True)
    times, t = [], None
    for line in r.stderr.splitlines():
        if "pts_time:" in line:
            t = float(line.split("pts_time:")[1].split()[0])
        elif "YAVG" in line and t is not None:
            if float(line.split("=")[-1]) > 128:
                times.append(t)
    firsts = [x for i, x in enumerate(times) if i == 0 or x - times[i - 1] > 0.5]
    return firsts, beep_times(src)


def mp3_bytes_per_s(apath: Path) -> float:
    """CBR mp3: a byte offset IS a time. Derived from the file, not assumed."""
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(apath)],
        capture_output=True, text=True, check=True).stdout.strip())
    return apath.stat().st_size / dur


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src, vdir, apath = root / "src.mp4", root / "frames", root / "audio.mp3"

        build_source(src)
        split(src, vdir, apath)

        frames = sorted(vdir.glob("*.jpg"))
        check("the single ffmpeg produced both outputs",
              len(frames) > 0 and apath.exists() and apath.stat().st_size > 0,
              f"{len(frames)} frames, {apath.stat().st_size} audio bytes")

        fl = flash_frames(vdir)
        bt = beep_times(apath)
        events = [f for i, f in enumerate(fl) if i == 0 or f - fl[i - 1] > 1]
        v_times = [f / FPS for f in events]

        print(f"\n  flash frames -> {events}  => video t = {[round(t, 3) for t in v_times]}")
        print(f"  beep starts  -> {[round(t, 3) for t in bt]}")
        print(f"  truth        -> {MARKS}")
        print(f"  mp3 bytes/s  -> {mp3_bytes_per_s(apath):.1f}\n")

        sv, sa = source_marks(src)
        print(f"  SOURCE  flashes {[round(x, 3) for x in sv]}  beeps {[round(x, 3) for x in sa]}")
        print(f"  SPLIT   flashes {[round(x, 3) for x in v_times]}  beeps {[round(x, 3) for x in bt]}\n")

        check("every marker survived the split",
              len(v_times) == len(MARKS) and len(bt) == len(MARKS),
              f"{len(v_times)} flashes, {len(bt)} beeps, expected {len(MARKS)}")

        if len(sv) == len(sa) == len(v_times) == len(bt) == len(MARKS):
            before = [v - a for v, a in zip(sv, sa)]
            after = [v - a for v, a in zip(v_times, bt)]
            delta = [b - a for b, a in zip(after, before)]
            print(f"  A/V skew in the SOURCE : {[round(x, 3) for x in before]}")
            print(f"  A/V skew after the SPLIT: {[round(x, 3) for x in after]}")
            print(f"  introduced by the split : {[round(x, 3) for x in delta]}\n")

            # THE test. One video frame at 10fps is 0.1s, and the video output is
            # resampled 30->10fps, so half a frame of quantisation is unavoidable and
            # is not skew. What must not happen is a systematic offset beyond that.
            check("the split introduces no A/V offset beyond frame quantisation",
                  max(abs(d) for d in delta) < 0.11,
                  f"worst {max(abs(d) for d in delta):.3f}s")
            check("and it does not DRIFT across the clip",
                  abs(delta[-1] - delta[0]) < 0.06,
                  f"first {delta[0]:.3f}s last {delta[-1]:.3f}s")
            check("audio position maps to time by a constant byte rate (CBR)",
                  abs(mp3_bytes_per_s(apath) - 6000) < 400, f"{mp3_bytes_per_s(apath):.0f} B/s")

    print("FAILED: " + ", ".join(fails) if fails else "all checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
