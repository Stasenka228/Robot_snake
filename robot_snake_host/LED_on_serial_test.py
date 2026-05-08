import serial
s1=serial.Serial('/dev/serial0', baudrate=115200, timeout=.5)
s1.write(b'led_on\r')
print(list(s1.read(20)))