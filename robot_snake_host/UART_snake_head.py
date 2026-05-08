import serial
import time
import math

ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=.5)

center = 127
alpha = 40
beta = 0
omega = 1
delta = math.pi/4
t = 0

while True:

    values = []

    for i in range (4):
        v = int(alpha*math.sin(omega*t+i*delta)+beta+center)
        v = max(0,min(255,v))
        values.append(v)

    try:

        for v in values:
            b = bytearray([v])
            ser.write(b)

        print("Sent:", values)

    except Exception as e:
        print("UART error:", e)
        
    t+=0.1
    time.sleep(0.02)