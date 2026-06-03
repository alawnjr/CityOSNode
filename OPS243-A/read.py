import serial

ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1)

ser.write(b'UC')

while True:
    ser.write(b'OD')
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if line:
        print(line)
