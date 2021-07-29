import RPi.GPIO as GPIO
import time


def rotate(repetition, sleep_time, is_clockwise, motor_pins):
    if is_clockwise:
        for _ in range(repetition):
            GPIO.output(motor_pins[0], GPIO.HIGH)
            GPIO.output(motor_pins[1], GPIO.HIGH)
            GPIO.output(motor_pins[2], GPIO.LOW)
            GPIO.output(motor_pins[3], GPIO.LOW)
            time.sleep(sleep_time)
            GPIO.output(motor_pins[0], GPIO.LOW)
            GPIO.output(motor_pins[1], GPIO.HIGH)
            GPIO.output(motor_pins[2], GPIO.HIGH)
            GPIO.output(motor_pins[3], GPIO.LOW)
            time.sleep(sleep_time)
            GPIO.output(motor_pins[0], GPIO.LOW)
            GPIO.output(motor_pins[1], GPIO.LOW)
            GPIO.output(motor_pins[2], GPIO.HIGH)
            GPIO.output(motor_pins[3], GPIO.HIGH)
            time.sleep(sleep_time)
            GPIO.output(motor_pins[0], GPIO.HIGH)
            GPIO.output(motor_pins[1], GPIO.LOW)
            GPIO.output(motor_pins[2], GPIO.LOW)
            GPIO.output(motor_pins[3], GPIO.HIGH)
            time.sleep(sleep_time)
    else:
        for _ in range(repetition):
            GPIO.output(motor_pins[0], GPIO.LOW)
            GPIO.output(motor_pins[1], GPIO.LOW)
            GPIO.output(motor_pins[2], GPIO.HIGH)
            GPIO.output(motor_pins[3], GPIO.HIGH)
            time.sleep(sleep_time)
            GPIO.output(motor_pins[0], GPIO.LOW)
            GPIO.output(motor_pins[1], GPIO.HIGH)
            GPIO.output(motor_pins[2], GPIO.HIGH)
            GPIO.output(motor_pins[3], GPIO.LOW)
            time.sleep(sleep_time)
            GPIO.output(motor_pins[0], GPIO.HIGH)
            GPIO.output(motor_pins[1], GPIO.HIGH)
            GPIO.output(motor_pins[2], GPIO.LOW)
            GPIO.output(motor_pins[3], GPIO.LOW)
            time.sleep(sleep_time)
            GPIO.output(motor_pins[0], GPIO.HIGH)
            GPIO.output(motor_pins[1], GPIO.LOW)
            GPIO.output(motor_pins[2], GPIO.LOW)
            GPIO.output(motor_pins[3], GPIO.HIGH)
            time.sleep(sleep_time)


def move_motor(depart, arrive, motor_pins1, motor_pins2):
    move = arrive - depart
    
    step = 8  # 8 repetition = 1 step = 5.625 degrees
    degrees_45 = step * 8
    degrees_61_875 = step * 11
    
    speed = 0.003

    if move > 0:
        for _ in range(abs(move)):
            rotate(degrees_45, speed, True, motor_pins1)
    elif move < 0:
        for _ in range(abs(move)):
            rotate(degrees_45, speed, False, motor_pins1)
    rotate(degrees_61_875, speed, True, motor_pins2)
    rotate(degrees_61_875, speed, False, motor_pins2)

    return arrive
