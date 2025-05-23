#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple example test script, to see how format works
"""
from gpiozero import Servo, DistanceSensor
from time import sleep
import serial
import time

# -------------------------------
# Hardware Setup
# -------------------------------
# Configure the servo on GPIO pin 18.
# Value range: -1 (max right turn) to 1 (max left turn), with 0 as straight (mid).
servo = Servo(18)

# Set up the two ultrasonic sensors:
# right sensor: echo on GPIO 17, trigger on GPIO 4.
right_sensor = DistanceSensor(echo=17, trigger=4)
# left sensor: echo on GPIO 22, trigger on GPIO 27.
left_sensor = DistanceSensor(echo=22, trigger=27)

#front_sensor = DistanceSensor(echo=,trigger=)

# Define a threshold distance (in meters) for obstacle detection.
THRESHOLD_DISTANCE_LR = 0.1
STEER_SLEEP_LEN = 0.1 # in seconds
STEER_INCREMENT = 0.1

# -------------------------------
# Serial Communication Setup
# -------------------------------
# Change the device name/port as needed.
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600

def send_command(arduino, cmd):
    """Send a command string to the Arduino over serial."""
    arduino.write((cmd).encode())
    print(f"Sent command:_{cmd}_")


def fwd_action(arduino):
    """
    With no obstacles, command the RC car to move forward.
    """
    send_command(arduino, "FORWARD")
    while arduino.inWaiting()==0: pass
    if  arduino.inWaiting()>0: 
        answer=arduino.readline()
        print(answer)
        arduino.flushInput() #remove data after reading

def stop_action(arduino):
    """
    If obstacles are detected on both sides,
    choose a default action, such as stopping.
    """
    print("Obstacles detected on both sides. Stopping.")
    send_command(arduino, "STOP")
    sleep(1)

# -------------------------------
# Main Loop: Integrated Control
# -------------------------------
def main():
    driving = False
    speed = 70
    steer = 0
    print("Starting obstacle detection and motor drive loop...")
    # Open serial communication with Arduino.
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as arduino:
        time.sleep(0.1)  # Wait briefly for the serial port to initialize.
        if arduino.isOpen():
            print(f"{arduino.port} connected!")
            # Set initial speed
            send_command(arduino, f"SPEED {speed}")
            # Set initial servo position (straight ahead).
            servo.value = steer
            try:
                while True:
                    # Read distances from both ultrasonic sensors.
                    left_distance = left_sensor.distance   # in meters
                    right_distance = right_sensor.distance # in meters
                    #front_distance = front_sensor.distance # in meters
                    print("Left distance: {:.2f} m, Right distance: {:.2f} m, Steer: {:.2f}".format(left_distance, right_distance, steer))

                    # Decide on action based on sensor readings.
                    #if front_distance < THRESHOLD_DISTANCE_FWD:
                     #   stop_action(arduino)
                    if driving and (left_distance < THRESHOLD_DISTANCE_LR and right_distance < THRESHOLD_DISTANCE_LR):
                        driving = False
                        stop_action(arduino)
                    elif right_distance < THRESHOLD_DISTANCE_LR:
                        if(steer <= 1-STEER_INCREMENT):
                            steer += STEER_INCREMENT
                        servo.value = steer
                        sleep(STEER_SLEEP_LEN)
                    elif left_distance < THRESHOLD_DISTANCE_LR:
                        if steer >= (-1+STEER_INCREMENT):
                            steer -= STEER_INCREMENT
                        servo.value = steer
                        sleep(STEER_SLEEP_LEN)
                    elif not driving:
                        driving = True
                        fwd_action(arduino)
            except KeyboardInterrupt:
                send_command(arduino, "STOP")
                sleep(1)
                print("KeyboardInterrupt caught, exiting.")
        else:
            print("Arduino not connected.")

if __name__ == '__main__':
    main()
