#!/usr/bin/env python3
"""
Continuous read of all I2C sensors on the MAESTRO 2.1 board.
 
Bus scan showed:
  0x0c  MLX90393  magnetometer
  0x29  TCS34725  color / light
  0x53  ADXL345   accelerometer
  0x76  BME680    temp / humidity / pressure / gas
 
Install drivers (in your venv):
  pip install adafruit-circuitpython-bme680 \
              adafruit-circuitpython-tcs34725 \
              adafruit-circuitpython-adxl34x \
              adafruit-circuitpython-mlx90393
 
Run:  python i2c_test.py
Stop: Ctrl-C
"""
 
import time
import board
 
# Each sensor is wrapped in try/except so one missing/failed chip
# doesn't kill the whole loop.
 
i2c = board.I2C()
 
# ---- BME680 (0x76) ----
bme680 = None
try:
    import adafruit_bme680
    bme680 = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=0x76)
    bme680.sea_level_pressure = 1013.25  # set to your local value for altitude
    print("BME680   : ok")
except Exception as e:
    print(f"BME680   : FAILED ({e})")
 
# ---- TCS34725 (0x29) ----
tcs = None
try:
    import adafruit_tcs34725
    tcs = adafruit_tcs34725.TCS34725(i2c)
    tcs.integration_time = 100  # ms
    tcs.gain = 4
    print("TCS34725 : ok")
except Exception as e:
    print(f"TCS34725 : FAILED ({e})")
 
# ---- ADXL345 (0x53) ----
adxl = None
try:
    import adafruit_adxl34x
    adxl = adafruit_adxl34x.ADXL345(i2c, address=0x53)
    print("ADXL345  : ok")
except Exception as e:
    print(f"ADXL345  : FAILED ({e})")
 
# ---- MLX90393 (0x0c) ----
mlx = None
try:
    import adafruit_mlx90393
    mlx = adafruit_mlx90393.MLX90393(i2c, address=0x0C)
    print("MLX90393 : ok")
except Exception as e:
    print(f"MLX90393 : FAILED ({e})")
 
print("\nStarting continuous read (Ctrl-C to stop)\n" + "-" * 60)
 
 
def read_bme680():
    if not bme680:
        return "BME680   : ---"
    return (f"BME680   : {bme680.temperature:5.1f} C  "
            f"{bme680.relative_humidity:4.1f} %RH  "
            f"{bme680.pressure:7.1f} hPa  "
            f"{bme680.gas:6d} ohm")
 
 
def read_tcs():
    if not tcs:
        return "TCS34725 : ---"
    r, g, b = tcs.color_rgb_bytes
    lux = tcs.lux
    temp = tcs.color_temperature
    cct = f"{temp:.0f}K" if temp else "n/a"
    return f"TCS34725 : R{r:3d} G{g:3d} B{b:3d}  {lux:6.1f} lux  {cct}"
 
 
def read_adxl():
    if not adxl:
        return "ADXL345  : ---"
    x, y, z = adxl.acceleration
    return f"ADXL345  : x{x:+6.2f}  y{y:+6.2f}  z{z:+6.2f}  m/s^2"
 
 
def read_mlx():
    if not mlx:
        return "MLX90393 : ---"
    mx, my, mz = mlx.magnetic
    return f"MLX90393 : x{mx:+8.1f} y{my:+8.1f} z{mz:+8.1f}  uT"
 
 
readers = [read_bme680, read_tcs, read_adxl, read_mlx]
 
try:
    while True:
        print(time.strftime("[%H:%M:%S]"))
        for r in readers:
            try:
                print("  " + r())
            except Exception as e:
                print(f"  read error: {e}")
        print()
        time.sleep(1.0)
except KeyboardInterrupt:
    print("\nStopped.")
 
