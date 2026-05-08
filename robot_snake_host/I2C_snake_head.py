from smbus2 import SMBus
import time
import math

addresses = [0x10] #0x11,0x12,0x13

alpha = 40
beta = 128
omega = 1
delta = math.pi/4
t = 0

bus = SMBus(1) #opens I2C bus connection

time.sleep(1)

while True:

    values = []

    for i in range(1):
        v = int(alpha*math.sin(omega*t + i*delta) + beta)
        v = max(0,min(255,v))
        values.append(v)

    try:

        for i in range(1):
            bus.write_i2c_block_data(addresses[i],0x00,[values[i]])

        print("Sent:",values)

    except OSError as e:
        print("I2C error:",e)

    t += 0.1
    time.sleep(0.1)