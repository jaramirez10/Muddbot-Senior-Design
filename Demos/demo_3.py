#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RC Car with Dual Ultrasonic + USB Camera (Red‑object) Fusion
"""
import cv2
import numpy as np
from gpiozero import Servo, DistanceSensor
from time import sleep
import serial
import time

# -------------------------------
# Hardware Setup
# -------------------------------
servo = Servo(18)

# Ultrasonic sensors
right_sensor = DistanceSensor(echo=17, trigger=4)
left_sensor  = DistanceSensor(echo=22, trigger=27)

THRESHOLD_DISTANCE_LR  = 0.2  # meters
THRESHOLD_DISTANCE_FWD = 0.5  # meters

# -------------------------------
# Serial Communication Setup
# -------------------------------
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE   = 9600
speed       = 150

def send_command(arduino, cmd):
    arduino.write((cmd).encode())
    print(f"Sent command: {cmd}")

# -------------------------------
# Obstacle Avoidance Functions
# -------------------------------
def steer_right_action():
    print("Steer right")
    servo.value = -0.3
    sleep(1)

def steer_left_action():
    print("Steer left")
    servo.value = 0.3
    sleep(1)

def fwd_action(arduino):
    print("Forward")
    send_command(arduino, "FORWARD")
    # wait for ack
    while arduino.inWaiting() == 0: pass
    print(arduino.readline().decode().strip())
    arduino.flushInput()

def stop_action(arduino):
    print("Stop")
    send_command(arduino, "STOP")
    sleep(1)

# -------------------------------
# Computer Vision Setup
# -------------------------------
cap = cv2.VideoCapture(0) # find usb port
if not cap.isOpened():
    raise IOError("Cannot open USB camera")


def detect_(frame):
    return 0

# Example function to detect red objects
def detect_red_obstacle(frame):
    """
    Returns True if a large red object is detected in the frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # red hue can wrap around, so mask two ranges
    lower1 = np.array([  0, 120,  70])
    upper1 = np.array([ 10, 255, 255])
    lower2 = np.array([170, 120,  70])
    upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)

    # Morphology to reduce noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Count non-zero (red) pixels
    red_count = cv2.countNonZero(mask)
    # Debug: show mask if needed
    # cv2.imshow("Red Mask", mask)
    return red_count > 2000  # tweak threshold to your environment

# -------------------------------
# Main Loop
# -------------------------------
def main():
    print("Starting fused obstacle detection loop...")
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as arduino:
        time.sleep(0.1)
        if not arduino.isOpen():
            print("Arduino not connected.")
            return

        print(f"{arduino.port} connected!")
        send_command(arduino, f"SPEED {speed}")
        servo.value = 0

        try:
            while True:
                # 1) Vision check
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed")
                    break

                if detect_red_obstacle(frame):
                    print("Vision: Red obstacle detected → stopping")
                    stop_action(arduino)
                    continue

                # 2) Ultrasonic check
                ld = left_sensor.distance
                rd = right_sensor.distance
                print(f"Left: {ld:.2f} m, Right: {rd:.2f} m")

                if ld < THRESHOLD_DISTANCE_LR and rd < THRESHOLD_DISTANCE_LR:
                    stop_action(arduino)
                elif ld < THRESHOLD_DISTANCE_LR:
                    steer_right_action()
                elif rd < THRESHOLD_DISTANCE_LR:
                    steer_left_action()
                else:
                    fwd_action(arduino)

                # short pause
                sleep(0.1)

        except KeyboardInterrupt:
            print("Exiting on user interrupt.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
