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

THRESHOLD_DISTANCE_LR = 0.1
STEER_SLEEP_LEN = 0.1 # in seconds
STEER_INCREMENT = 0.1

# -------------------------------
# Serial Communication Setup
# -------------------------------
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE   = 9600
speed       = 150

# LOAD CAMERA CALIBRATION
with open('calib.yaml') as f:
    data = yaml.safe_load(f)
K    = np.array(data['K'])
dist = np.array(data['dist'])

# odo_dist = 0.05  # example: car moved 5 cm between frames
odo_dist = None


# FEATURE DETECTOR & MATCHER 
orb = cv.ORB_create(2000)
bf  = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

# START VIDEO STREAM 
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
result = cv.VideoWriter('recording.avi',  
                        cv.VideoWriter_fourcc(*'MJPG'), 
                        10, size) 


# Read first frame
ret, first = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame")
prev_gray = cv.cvtColor(first, cv.COLOR_BGR2GRAY)
prev_gray = cv.undistort(prev_gray, K, dist)
kp_prev, des_prev = orb.detectAndCompute(prev_gray, None)

while True:
    ret, frame = cap.read()

    frame_out, dists, prev_gray, kp_prev, des_prev = process_frame(
    frame, prev_gray, kp_prev, des_prev, K, dist, orb, bf, odo_dist=None)

    cv.imshow('Depth + Features', frame_out)
    result.write(frame_out)
    # break on ESC
    if cv.waitKey(1) & 0xFF == 27:
            break

print("Done!")
result.release()
cap.release()
cv.destroyAllWindows()