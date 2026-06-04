#!/usr/bin/env python3
"""
GPIO scanner -- find which Raspberry Pi GPIO the PIR (NCS36000) is wired to.

The PIR motion signal is NOT on the MCP3008 (confirmed by ADC swing test),
so it's almost certainly a digital signal on a GPIO pin. This watches all
general-purpose GPIOs, records their idle state, then reports any pin that
CHANGES while you move in front of the board.

Pins already used by other buses are skipped:
  I2C : GPIO2 (SDA), GPIO3 (SCL)
  SPI : GPIO7,8 (CE1/CE0), GPIO9 (MISO), GPIO10 (MOSI), GPIO11 (SCLK)
  UART: GPIO14 (TXD), GPIO15 (RXD)

Uses gpiozero (preinstalled on Raspberry Pi OS). If missing:
  pip install gpiozero lgpio

Run:  python gpio_scan.py
Walk in front of the board the whole time. Stop: Ctrl-C
"""

import time
from gpiozero import DigitalInputDevice

# BCM GPIO numbers that are general-purpose and free to probe here.
# Skipping I2C (2,3), SPI (7,8,9,10,11), UART (14,15).
CANDIDATE_PINS = [4, 5, 6, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

SCAN_SECONDS = 25     # how long to watch
POLL_S = 0.01         # polling interval

# Set up each candidate pin as a plain digital input.
# pull_up=None leaves the pin floating so we read whatever the board drives.
devices = {}
for pin in CANDIDATE_PINS:
    try:
        devices[pin] = DigitalInputDevice(pin, pull_up=None)
    except Exception as e:
        print(f"  (skipping GPIO{pin}: {e})")

print(f"Watching {len(devices)} GPIO pins for {SCAN_SECONDS}s.")
print("Walk in front of the board / wave repeatedly...\n")

# Record the initial (idle) state of every pin.
idle = {pin: dev.value for pin, dev in devices.items()}

# Count how many times each pin changes, and track high/low time.
changes = {pin: 0 for pin in devices}
high_count = {pin: 0 for pin in devices}
samples = 0
last = dict(idle)

end = time.monotonic() + SCAN_SECONDS
try:
    while time.monotonic() < end:
        samples += 1
        for pin, dev in devices.items():
            v = dev.value
            if v:
                high_count[pin] += 1
            if v != last[pin]:
                changes[pin] += 1
                last[pin] = v
        time.sleep(POLL_S)
except KeyboardInterrupt:
    pass

print("Results (idle state -> activity):")
print("-" * 55)
candidates = []
for pin in devices:
    pct_high = 100.0 * high_count[pin] / samples if samples else 0
    flag = ""
    if changes[pin] > 0:
        flag = "  <<< CHANGED (likely PIR)"
        candidates.append((pin, changes[pin]))
    print(f"  GPIO{pin:<2}  idle={idle[pin]}  toggled {changes[pin]:4d}x  "
          f"high {pct_high:5.1f}% of time{flag}")

print("-" * 55)
if candidates:
    candidates.sort(key=lambda x: -x[1])
    best = candidates[0][0]
    print(f"\nMost active pin: GPIO{best}  ->  that's almost certainly the PIR.")
    print(f"Use PIR_PIN = {best} in the motion code.")
else:
    print("\nNo pin changed. Try again moving more (PIR needs MOVING heat),")
    print("or the PIR may need a jumper/power enable on the board.")
