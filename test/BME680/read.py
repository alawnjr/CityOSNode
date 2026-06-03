import board, adafruit_bme680
i2c = board.I2C()
sensor = adafruit_bme680.Adafruit_BME680_I2C(i2c,address=0x76)
print(sensor.temperature, sensor.humidity, sensor.pressure, sensor.gas)
