import time, board, busio, digitalio
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn

spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
cs = digitalio.DigitalInOut(board.D8)
mcp = MCP.MCP3008(spi, cs, ref_voltage=3.3)
chans = [AnalogIn(mcp, getattr(MCP, f"P{i}")) for i in range(8)]

print("Make noise. Watching all 8 channels for ~15s...")
peak = [0.0]*8
end = time.monotonic() + 15
while time.monotonic() < end:
    for i, c in enumerate(chans):
        v = c.voltage
        if v > peak[i]:
            peak[i] = v
    time.sleep(0.002)

for i, p in enumerate(peak):
    print(f"  P{i}: max {p:.3f} V")
