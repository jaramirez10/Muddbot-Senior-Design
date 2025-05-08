import cv2 as cv
import numpy as np
import yaml
import time
import numpy as np

from time import sleep
from utils.utils import *

import serial

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE   = 115200


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

# Read first frame
ret, first = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame")

prev_gray = cv.cvtColor(first, cv.COLOR_BGR2GRAY)
prev_gray = cv.undistort(prev_gray, K, dist)
kp_prev, des_prev = orb.detectAndCompute(prev_gray, None)



def forward():
    send_command(arduino, "FORWARD")
def stop():
    send_command(arduino, "STOP")
def setSpeed(speed):
    send_command(arduino, f"SPEED {speed}")
def setSteer(steer):
    send_command(arduino, f"STEER {steer}")
def getLFRdists():
    return parse_sensor_prompt(send_command(arduino, "SENSOR"))



with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as arduino:
    time.sleep(0.1)  # Wait briefly for the serial port to initialize.
    if arduino.isOpen():
        print(f"{arduino.port} connected!")
        stop()
        t_init = time.time()
        try:
            while True:
                left_dist, fwd_dist, right_dist = getLFRdists() # in cm
                ret, frame = cap.read()
                # frame_out is the video frame (overlayed with the corners)
                # dists is the distances of all the corners
                # the rest are internal variables
                frame_out, dists, prev_gray, kp_prev, des_prev = process_frame(
                frame, prev_gray, kp_prev, des_prev, K, dist, orb, bf, odo_dist=None)
                
                t_now = time.time()

                cv.putText(frame_out,
                    f"forward_sensor: {fwd_dist:.2f} cm",
                    (20,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv.putText(frame_out, 
                    f"left_sensor: {left_dist: .2f} cm|| right_sensor: {right_dist: .2f} cm", 
                    (20,60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv.putText(frame_out,
                    f"time elapsed: {t_now - t_init: .2f}",
                    (20,90),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
        
                cv.imshow('frame', frame_out)
                if cv.waitKey(1) == ord('q'):
                    break
        except KeyboardInterrupt:
            send_command(arduino, "STOP")
            sleep(1)
            print("KeyboardInterrupt caught, exiting.")
        finally:            
            print("Done!")
            cap.release()
            cv.destroyAllWindows()
    else:
        print("Arduino not connected.")
            
