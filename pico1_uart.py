from machine import PWM, UART, Pin
import utime

uart_in = UART(0, 9600, tx=Pin(0), rx=Pin(1))
uart_out = UART(0, 9600, tx=Pin(6), rx=Pin(7))

print("Pico UART ready")

class Servo:
    def __init__(self, MIN_DUTY=300000, MAX_DUTY=2300000, pin=16, freq=50):
        self.pwm = PWM(Pin(pin))
        self.pwm.freq(freq)
        self.MIN_DUTY = MIN_DUTY
        self.MAX_DUTY = MAX_DUTY
        
    def rotateDeg(self, pwm:int):
        if pwm < 0:
            pwm = 0
        elif pwm > 255:
            pwm = 255
            
        duty_ns = int(self.MAX_DUTY - pwm * (self.MAX_DUTY-self.MIN_DUTY)/255)
        self.pwm.duty_ns(duty_ns)

servo = Servo()

while True: 

    if uart_in.any():
            data = uart_in.read(1)
            
            if data:
                value = data[0]
                led.toggle()  # blinks when data received

                # Forward remaining bytes to next MC
                remaining = uart_in.read()
                if remaining:
                    print("Forwarding:", remaining)
                    uart_out.write(remaining)

                # move motor
                if current_angle < value:
                    current_angle = min(current_angle + 2, value)
                elif current_angle > value:
                    current_angle = max(current_angle - 2, value)

                servo.rotateDeg(current_angle)

    utime.sleep_ms(10)
