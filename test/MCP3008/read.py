#!/usr/bin/env python3
"""
Sound-level monitor for the MAESTRO 2.1 board.

Signal path (analog -> MCP3008 ADC -> SPI -> Pi):
  electret mic -> MAX9814 (amp + AGC) -> MCP3008 MIC_CH

What it does:
  - Mic: auto-calibrates quiet baseline, reports loudness, flags loud events

What it does NOT do:
  - Record audio / FFT. SPI-from-Python is only fast enough for level sensing.

Install driver (in venv):
  pip install adafruit-circuitpython-mcp3xxx

Run:  python sensor_monitor.py
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
CS_PIN  = board.D8       # MCP3008 chip-select: CE0 = D8, CE1 = D7
MIC_CH  = MCP.P0         # confirmed: mic (MAX9814) on P0
VREF    = 3.3            # MCP3008 reference / board supply voltage

# --- mic settings ---
WINDOW_S    = 0.05       # sampling window per loudness reading (seconds)
MAX_SAMPLES = 300        # cap on samples per window
REPORT_HZ   = 5          # readings per second
EVENT_FACTOR   = 3.0     # loud event when pp > baseline * this
EVENT_COOLDOWN = 0.4     # min seconds between loud events
BASELINE_DECAY = 0.95    # mic baseline smoothing (closer to 1 = slower)
# ---------------------------------------------------------------------------

# --- set up MCP3008 ---
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
cs = digitalio.DigitalInOut(CS_PIN)
mcp = MCP.MCP3008(spi, cs, ref_voltage=VREF)
mic = AnalogIn(mcp, MIC_CH)


def sample_pp():
    """Sample the mic fast over WINDOW_S, return peak-to-peak voltage."""
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


def calibrate_mic(seconds=2.0):
    print(f"Calibrating mic baseline for {seconds:.0f}s... stay quiet.")
    samples = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        samples.append(sample_pp())
        time.sleep(0.02)
    base = max(sum(samples) / len(samples) if samples else 0.01, 0.01)
    print(f"  mic baseline pp = {base:.3f} V")
    return base


print("Smart-room monitor: mic")
print(f"  MIC ch {MIC_CH}   CS {CS_PIN}")
mic_baseline = calibrate_mic()
print()

interval = 1.0 / REPORT_HZ
last_event = 0.0

print("Monitoring (Ctrl-C to stop)")
print("-" * 72)

try:
    while True:
        now = time.monotonic()

        pp = sample_pp()
        if pp < mic_baseline:
            mic_baseline = BASELINE_DECAY * mic_baseline + (1 - BASELINE_DECAY) * pp
            mic_baseline = max(mic_baseline, 0.005)

        level = min(pp / VREF, 1.0)
        bar_len = int(level * 20)
        bar = "#" * bar_len + "-" * (20 - bar_len)

        loud = ""
        if pp > mic_baseline * EVENT_FACTOR and (now - last_event) > EVENT_COOLDOWN:
            last_event = now
            loud = " LOUD"

        print(f"[{time.strftime('%H:%M:%S')}]  "
              f"MIC pp {pp:4.2f}V |{bar}|{loud:<5}")
        time.sleep(interval)
except KeyboardInterrupt:
    print("\nStopped.")
