import serial
import sys


if len(sys.argv) != 2 or sys.argv[1] not in ('on', 'off'):
    print(f"Usage: {sys.argv[0]} [on|off]")
    sys.exit(1)

s1 = serial.Serial('/dev/serial0', baudrate=9600, timeout=.5)

# Write "led_on\r" to the serial port
if sys.argv[1] == 'on':
    s1.write(b'led_on\r')
# Write "led_off\r" to the serial port
elif sys.argv[1] == 'off':
    s1.write(b'led_off\r')

print(f"Response: {s1.read(20).decode()}")

sys.exit(0)