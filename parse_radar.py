#!/usr/bin/env python3
"""
Downstream parser: split the OPS243-C raw radar log into typed columns.

capture.py writes radar_ops243.csv as a verbatim dump of the sensor's serial
lines (per the "keep raw values raw" convention), so speed and distance reports
land interleaved in a single `raw` column. The sensor emits two data shapes:

    12875.710,"cmps",54.6     -> speed report  (cm/s)
    12876.048,"cm",332.2      -> distance report (cm)

plus the odd bare number and JSON config echo (e.g. {"SpeedOutputFeature":"D"}),
which carry no reading and are skipped.

This reads that raw CSV and writes radar_ops243_parsed.csv alongside it with
clean, separated columns:

    timestamp_seconds,sensor_time_s,speed_cmps,distance_cm

one row per data line (the un-used measurement column is left blank). The raw
log is left untouched; this only adds the derived file. If metadata.json is
present, a `radar_ops243_parsed` stream entry is registered in it.

Usage (run on the Pi after capture.py, which the .sh wrapper does automatically):

    python parse_radar.py                 # parse the newest recording under data/
    python parse_radar.py <recording_dir> # parse a specific rec_* folder
"""

import csv
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# unit token (lower-cased) -> (kind, factor-to-canonical). Speed is canonicalised
# to cm/s and distance to cm so the output columns have fixed units regardless of
# how the sensor was configured.
SPEED_UNITS = {"cmps": 1.0, "mps": 100.0}
DISTANCE_UNITS = {"cm": 1.0, "m": 100.0}


def parse_raw(raw):
    """Parse one raw serial line into (kind, sensor_time_s, value_canonical).

    Returns None for lines that aren't a unit-tagged reading (bare numbers,
    JSON config echoes, blanks).
    """
    raw = raw.strip()
    if not raw or raw.startswith("{"):
        return None
    try:
        fields = [f.strip() for f in next(csv.reader([raw]))]
    except StopIteration:
        return None

    unit = next((f.lower() for f in fields
                 if f.lower() in SPEED_UNITS or f.lower() in DISTANCE_UNITS), None)
    if unit is None:
        return None

    nums = []
    for f in fields:
        if f.lower() == unit:
            continue
        try:
            nums.append(float(f))
        except ValueError:
            pass
    if not nums:
        return None

    value = nums[-1]                              # the reading is the last number
    sensor_time = nums[0] if len(nums) > 1 else None  # leading number is the sensor clock
    if unit in SPEED_UNITS:
        return "speed", sensor_time, value * SPEED_UNITS[unit]
    return "distance", sensor_time, value * DISTANCE_UNITS[unit]


def parse_file(raw_path, out_path):
    counts = {"speed": 0, "distance": 0, "skipped": 0}
    with raw_path.open(newline="") as src, out_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.writer(dst)
        writer.writerow(["timestamp_seconds", "sensor_time_s", "speed_cmps", "distance_cm"])
        for row in reader:
            parsed = parse_raw(row.get("raw", ""))
            if parsed is None:
                counts["skipped"] += 1
                continue
            kind, sensor_time, value = parsed
            counts[kind] += 1
            st = f"{sensor_time:.3f}" if sensor_time is not None else ""
            if kind == "speed":
                writer.writerow([row["timestamp_seconds"], st, f"{value:.1f}", ""])
            else:
                writer.writerow([row["timestamp_seconds"], st, "", f"{value:.1f}"])
    return counts


def register_in_metadata(rec_dir):
    """Add the parsed stream to metadata.json if it exists (best-effort)."""
    meta_path = rec_dir / "metadata.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text())
        meta.setdefault("streams", {})["radar_ops243_parsed"] = {
            "modality": "motion",
            "path": "streams/radar_ops243_parsed.csv",
            "derived_from": "radar_ops243",
            "fields": ["sensor_time_s", "speed_cmps", "distance_cm"],
        }
        meta_path.write_text(json.dumps(meta, indent=2))
    except (OSError, ValueError) as error:
        print(f"parse_radar: could not update metadata.json ({error})", file=sys.stderr)


def newest_recording():
    rec_dirs = [p for p in DATA_DIR.glob("day_*/rec_*") if p.is_dir()]
    if not rec_dirs:
        return None
    return max(rec_dirs, key=lambda p: p.stat().st_mtime)


def main():
    if len(sys.argv) > 1:
        rec_dir = Path(sys.argv[1]).resolve()
    else:
        rec_dir = newest_recording()
        if rec_dir is None:
            print("parse_radar: no recordings found under data/", file=sys.stderr)
            return 0  # nothing to do is not a failure

    raw_path = rec_dir / "streams" / "radar_ops243.csv"
    if not raw_path.exists():
        print(f"parse_radar: no radar log at {raw_path}", file=sys.stderr)
        return 0

    out_path = rec_dir / "streams" / "radar_ops243_parsed.csv"
    counts = parse_file(raw_path, out_path)
    register_in_metadata(rec_dir)
    print(f"parse_radar: {out_path} "
          f"({counts['speed']} speed, {counts['distance']} distance, "
          f"{counts['skipped']} skipped)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
