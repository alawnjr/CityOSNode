import time, board, busio, digitalio
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn

spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
cs = digitalio.DigitalInOut(board.D8)
mcp = MCP.MCP3008(spi, cs, ref_voltage=3.3)
chans = [AnalogIn(mcp, getattr(MCP, f"P{i}")) for i in range(8)]

print("Walk in front of the board for ~20s...")
lo = [3.3]*8
hi = [0.0]*8
end = time.monotonic() + 20
while time.monotonic() < end:
    for i, c in enumerate(chans):
        v = c.voltage
        if v < lo[i]: lo[i] = v
        if v > hi[i]: hi[i] = v
    time.sleep(0.01)

for i in range(8):
    print(f"  P{i}: min {lo[i]:.3f}  max {hi[i]:.3f}  swing {hi[i]-lo[i]:.3f} V")
