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
# left sensor: echo on GPIO 27, trigger on GPIO 22.
left_sensor = DistanceSensor(echo=22, trigger=27)

#front_sensor = DistanceSensor(echo=,trigger=)

# Define a threshold distance (in meters) for obstacle detection.
THRESHOLD_DISTANCE_LR = 0.2
THRESHOLD_DISTANCE_FWD = 0.5

# -------------------------------
# Serial Communication Setup
# -------------------------------
# Change the device name/port as needed.
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
speed = 200

def send_command(arduino, cmd):
    """Send a command string to the Arduino over serial."""

    arduino.write((cmd+'\n').encode())
    print(f"Sent command:_{cmd}_")

# -------------------------------
# Obstacle Avoidance Functions
# -------------------------------
def steer_right_action(arduino):
    """
    If an obstacle is detected on the left side,
    steer right (e.g., by setting servo to -1) and send a serial command.
    """
    print("Left obstacle detected! Steering right...")
    servo.value = -0.5   # Local servo action (steering right)
    sleep(1)
    # Return steering to straight
    servo.value = 0
    sleep(1)

def steer_left_action(arduino):
    """
    If an obstacle is detected on the right side,
    steer left (e.g., by setting servo to 1) and send a serial command.
    """
    print("Right obstacle detected! Steering left...")
    servo.value = 0.5     # Local servo action (steering left)
    sleep(1)
    # Return steering to straight
    servo.value = 0
    sleep(1)

def fwd_action(arduino):
    """
    With no obstacles, command the RC car to move forward.
    """
    print("No obstacles detected. Moving forward.")
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
    print("Starting obstacle detection and motor drive loop...")
    # Open serial communication with Arduino.
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as arduino:
        time.sleep(0.1)  # Wait briefly for the serial port to initialize.
        if arduino.isOpen():
            print(f"{arduino.port} connected!")
            # Set initial speed
            send_command(arduino, f"SPEED {speed}")
            # Set initial servo position (straight ahead).
            servo.value = 0
            try:
                while True:
                    # Read distances from both ultrasonic sensors.
                    left_distance = left_sensor.distance   # in meters
                    right_distance = right_sensor.distance # in meters
                    #front_distance = front_sensor.distance # in meters
                    print("Left distance: {:.2f} m, Right distance: {:.2f} m".format(left_distance, right_distance))

                    # Decide on action based on sensor readings.
                    #if front_distance < THRESHOLD_DISTANCE_FWD:
                     #   stop_action(arduino)
                    if left_distance < THRESHOLD_DISTANCE_LR and right_distance < THRESHOLD_DISTANCE_LR:
                        stop_action(arduino)
                    elif left_distance < THRESHOLD_DISTANCE_LR:
                        steer_right_action(arduino)
                    elif right_distance < THRESHOLD_DISTANCE_LR:
                        steer_left_action(arduino)
                    else:
                        fwd_action(arduino)
            except KeyboardInterrupt:
                print("KeyboardInterrupt caught, exiting.")
        else:
            print("Arduino not connected.")

if __name__ == '__main__':
    main()
