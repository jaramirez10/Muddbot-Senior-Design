import cv2 as cv
import numpy as np
import yaml
import time
import numpy as np
from gpiozero import Servo, DistanceSensor
from time import sleep
from utils.utils import *
import serial

# -------------------------------
# Hardware Setup
# -------------------------------
servo = Servo(18)

# Ultrasonic sensors
right_sensor = DistanceSensor(echo=17, trigger=4)
left_sensor  = DistanceSensor(echo=22, trigger=27)

THRESHOLD_DISTANCE_LR = 0.05
STEER_SLEEP_LEN = 0.1 # in seconds
STEER_INCREMENT = 0.1

# -------------------------------
# Serial Communication Setup
# -------------------------------
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE   = 9600


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
        sleep(2)
        fwd_action(arduino)
        sleep(2)
        stop_action(arduino)