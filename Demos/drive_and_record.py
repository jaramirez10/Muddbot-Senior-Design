import cv2 as cv
import numpy as np
import yaml
import time
import numpy as np

from time import sleep
from utils.utils import *
import serial


THRESHOLD_DISTANCE_LR = 30 #cm
STEER_SLEEP_LEN = 1 # in seconds
STEER_INCREMENT = 7 # degrees
SERVO_MAX_L = 150
SERVO_MAX_R = 30
# -------------------------------
# Serial Communication Setup
# -------------------------------
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


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
clean_recording = cv.VideoWriter('clean_recording.avi',  
                        cv.VideoWriter_fourcc(*'MJPG'), 
                        10, size) 
edited_recording = cv.VideoWriter('edited_recording.avi',  
                        cv.VideoWriter_fourcc(*'MJPG'), 
                        10, size) 


# Read first frame
ret, first = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame")
prev_gray = cv.cvtColor(first, cv.COLOR_BGR2GRAY)
prev_gray = cv.undistort(prev_gray, K, dist)
kp_prev, des_prev = orb.detectAndCompute(prev_gray, None)

driving = False
speed = 70
steer = 90

def forward():
    send_command(arduino, "FORWARD")
def stop():
    send_command(arduino, "STOP")
def setSpeed(speed):
    send_command(arduino, f"SPEED {speed}")
def setSteer(steer):
    send_command(arduino, f"STEER {steer}")
def getLRdists():
    return parse_sensor_prompt(send_command(arduino, "SENSOR"))

print("Starting obstacle detection and motor drive loop...")
# Open serial communication with Arduino.
with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as arduino:
        time.sleep(0.1)  # Wait briefly for the serial port to initialize.
        if arduino.isOpen():
            print(f"{arduino.port} connected!")
            # Set initial servo position (straight ahead).
            t_init = time.time()
            t_0 = time.time()
            setSpeed(speed)
            forward()
            try:
                while True:
                    left_dist, right_dist = getLRdists() # in cm

                    ret, frame = cap.read()
                    # frame_out is the video frame (overlayed with the corners)
                    # dists is the distances of all the corners
                    # the rest are internal variables
                    frame_out, dists, prev_gray, kp_prev, des_prev = process_frame(
                    frame, prev_gray, kp_prev, des_prev, K, dist, orb, bf, odo_dist=None)
                    if dists.size:
                        fwd_dist = dists.min()
                    else:
                        fwd_dist = 0
                        
                    t_now = time.time()
                    print("Left distance: {:.2f} m, Right distance: {:.2f} m, Steer: {:.2f}, fwd_dist {:.2f}, driving: {}, delta t = {}".format(left_dist, right_dist, steer, fwd_dist, driving, (t_now - t_0)))

                    # Decide on action based on sensor readings.
                    if driving and (left_dist < THRESHOLD_DISTANCE_LR and right_dist < THRESHOLD_DISTANCE_LR):
                        driving = False
                        stop()
                    elif (t_now - t_0) < STEER_SLEEP_LEN: 
                        pass
                    elif right_dist < THRESHOLD_DISTANCE_LR:
                        if(steer <= SERVO_MAX_L):
                            steer += STEER_INCREMENT
                        setSteer(steer)
                        t_0 = t_now
                    elif left_dist < THRESHOLD_DISTANCE_LR:
                        print("steer right")
                        if steer >= SERVO_MAX_R:
                            steer -= STEER_INCREMENT
                        setSteer(steer)
                        t_0 = t_now
                    elif not driving:
                        driving = True
                        forward()

                    # edit video with relevant information:
                    clean_recording.write(frame_out)

                    cv.putText(frame_out,
                        f"Closest: {fwd_dist:.2f} {'m' if odo_dist else 'units'}",
                        (20,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv.putText(frame_out, 
                        f"left_sensor: {left_dist: .2f} || right_sensor: {right_dist: .2f}", 
                        (20,60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv.putText(frame_out,
                        f"steer: {steer: .2f} || speed: {speed}",
                        (20,90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv.putText(frame_out,
                        f"driving: {driving}",
                        (20,120),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv.putText(frame_out,
                        f"time elapsed: {t_now - t_init}",
                        (20,150),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    
                    
                    edited_recording.write(frame_out)
            except KeyboardInterrupt:
                send_command(arduino, "STOP")
                sleep(1)
                print("KeyboardInterrupt caught, exiting.")
            finally:            
                print("Done!")
                clean_recording.release()
                edited_recording.release()
                cap.release()
                cv.destroyAllWindows()
        else:
            print("Arduino not connected.")
