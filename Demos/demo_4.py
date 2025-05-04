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
right_sensor = DistanceSensor(echo=12, trigger=13)
left_sensor  = DistanceSensor(echo=7, trigger=6)
front_sensor = DistanceSensor(echo=1, trigger=2)
THRESHOLD_DISTANCE_LR = 40
STEER_SLEEP_LEN = 0.1 # in seconds
STEER_INCREMENT = 0.1

# -------------------------------
# Serial Communication Setup
# -------------------------------
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE   = 115200
speed = 70
def backward():
    send_command(arduino, "BACKWARD")

def getFdists():
    answer = send_command(arduino, "SENSOR")
    lb = answer.index('[')
    bars = [i for i,c in enumerate(answer) if c == '|']
    rb = answer.index(']')
    left  = int(answer[lb+1    : bars[0]        ])
    right = int(answer[bars[0]+1 : bars[1]      ])
    front = int(answer[bars[1]+1 : rb            ])
    return left, right, front

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
        try:
            while True:
                left_dist,right_dist, fwd_dist = getFdists()
                print("Left distance: {:.2f} m, Right distance: {:.2f} m, fwd_dist {:.2f}".format(left_dist, right_dist, fwd_dist))
                if fwd_dist < THRESHOLD_DISTANCE_LR:
                    print("stop")
                    backward()
                
        except KeyboardInterrupt:
                send_command(arduino, "STOP")
                sleep(1)
                print("KeyboardInterrupt caught, exiting.")
        finally:            
            print("Done!")
