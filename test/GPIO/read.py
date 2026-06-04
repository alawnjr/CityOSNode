from gpiozero import MotionSensor
pir = MotionSensor(PIN)
pir.when_motion = lambda: print("motion!")
