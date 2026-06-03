#!/usr/bin/env python3
"""
Sound-level monitor for the MAESTRO 2.1 onboard mic.

Signal path:  electret mic -> MAX9814 (amp + AGC) -> MCP3008 ch -> SPI -> Pi

What it does:
  - Auto-calibrates the quiet-room baseline at startup
  - Continuously reports loudness (peak-to-peak amplitude + a rolling level)
  - Detects loud events (claps, doors, raised voices) above an adaptive threshold

What it does NOT do:
  - Record audio / transcribe / do real FFT. The MCP3008-over-SPI path from
    Python is only fast enough for level/envelope sensing, not audio capture.

  >>> SET MIC_CH AND CS_PIN below to match your schematic/wiring <<<

Install driver (in venv):
  pip install adafruit-circuitpython-mcp3xxx

Run:  python mic_monitor.py
Stop: Ctrl-C
"""

import time
import board
import busio
import digitalio
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn

# ---------------------------------------------------------------------------
# CONFIG  -- edit to match your board
# ---------------------------------------------------------------------------
CS_PIN = board.D8        # MCP3008 chip-select: CE0 = D8, CE1 = D7
MIC_CH = MCP.P0          # MCP3008 channel the MAX9814 output is wired to
VREF   = 3.3             # MCP3008 reference / board supply voltage

# How each loudness reading is taken: sample fast over a short window and
# measure peak-to-peak swing. Bigger swing = louder.
WINDOW_S   = 0.05        # sampling window length (seconds) per reading
MAX_SAMPLES = 300        # cap on samples per window
REPORT_HZ  = 5           # how many readings per second to print

# Event detection: an event fires when the current loudness exceeds the
# rolling baseline by this multiplier. Tune EVENT_FACTOR for sensitivity.
EVENT_FACTOR   = 3.0     # current pp must exceed baseline_pp * this
EVENT_COOLDOWN = 0.4     # seconds to wait before another event can fire
BASELINE_DECAY = 0.95    # rolling baseline smoothing (closer to 1 = slower)
# ---------------------------------------------------------------------------

# --- set up MCP3008 ---
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
cs = digitalio.DigitalInOut(CS_PIN)
mcp = MCP.MCP3008(spi, cs, ref_voltage=VREF)
mic = AnalogIn(mcp, MIC_CH)


def sample_pp():
    """Sample the mic fast over WINDOW_S and return peak-to-peak voltage."""
    lo = VREF
    hi = 0.0
    n = 0
    end = time.monotonic() + WINDOW_S
    while n < MAX_SAMPLES and time.monotonic() < end:
        v = mic.voltage
        if v < lo:
            lo = v
        if v > hi:
            hi = v
        n += 1
    return hi - lo


def calibrate(seconds=2.0):
    """Measure the quiet-room baseline peak-to-peak so we can detect spikes."""
    print(f"Calibrating quiet baseline for {seconds:.0f}s... stay quiet.")
    samples = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        samples.append(sample_pp())
        time.sleep(0.02)
    base = sum(samples) / len(samples) if samples else 0.01
    base = max(base, 0.01)  # avoid zero baseline
    print(f"Baseline pp = {base:.3f} V\n")
    return base


print("Mic sound-level monitor")
print(f"  MIC on channel {MIC_CH}, CS {CS_PIN}")
baseline = calibrate()

interval = 1.0 / REPORT_HZ
last_event = 0.0

print("Monitoring (Ctrl-C to stop)")
print("-" * 60)

try:
    while True:
        pp = sample_pp()

        # adaptive baseline: only let quiet periods pull it down
        if pp < baseline:
            baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * pp
            baseline = max(baseline, 0.005)

        # loudness 0..1 relative to full scale, and a text bar
        level = min(pp / VREF, 1.0)
        bar_len = int(level * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)

        # event detection
        event = ""
        now = time.monotonic()
        if pp > baseline * EVENT_FACTOR and (now - last_event) > EVENT_COOLDOWN:
            last_event = now
            event = "  <<< LOUD EVENT"

        print(f"[{time.strftime('%H:%M:%S')}] pp {pp:4.2f} V  "
              f"|{bar}| {level*100:3.0f}%{event}")
        time.sleep(interval)
except KeyboardInterrupt:
    print("\nStopped.")
