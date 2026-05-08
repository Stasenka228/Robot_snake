import serial
import time

ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=0.5)

CENTER = 120

try:
    while True:

        values = [CENTER, CENTER, CENTER, CENTER, CENTER, CENTER]

        for v in values:
            ser.write(bytearray([v]))

        print("Sent:", values)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Stopped.")

finally:
    ser.close()