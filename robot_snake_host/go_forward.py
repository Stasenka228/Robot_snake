import serial
import time
import math

ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=0.5)

NUM_SERVOS = 6

CENTER = 118
ALPHA = 80              # wave amplitude (how big is the side to side bending)
OMEGA = 2.0             # speed
DELTA = math.pi / 4     # phase shift and spacing between the joints

DT = 0.05
# ALPHA * math.sin(OMEGA * t + i * DELTA)
try:
    t = 0

    while True:

        values = []

        for i in range(NUM_SERVOS):

            v = int(
                CENTER - ALPHA * math.sin(OMEGA * t - i * DELTA)
                
            )

            v = max(0, min(255, v))

            values.append(v)

        # send commands to servos
        # values.reverse()
        # 123

        for v in values:
            ser.write(bytearray([v]))

        print("Sent:", values)

        t += DT
        time.sleep(DT)

except KeyboardInterrupt:
    print("Stopped")

    # center snake when done
    for _ in range(20):
        for i in range(NUM_SERVOS):
            ser.write(bytearray([CENTER]))
        time.sleep(0.05)

finally:
    ser.close()