import smbus
import sys

if len(sys.argv) != 2 or sys.argv[1] not in ('on', 'off', '?'):
    print(f"Usage: {sys.argv[0]} [on|off|?]")
    sys.exit(1)

bus = smbus.SMBus(1)

if sys.argv[1] == 'on':
    bus.write_byte(0x43, 1)
    
elif sys.argv[1] == 'off':
    bus.write_byte(0x43, 0)
    
elif sys.argv[1] == '?':
    state = bus.read_byte(0x43)
    state_str = "on" if state != 0 else "off"
    print(f"State: {state_str}")

sys.exit(0)