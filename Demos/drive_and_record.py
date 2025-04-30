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
STEER_SLEEP_LEN = 1 # in seconds
STEER_INCREMENT = 0.1
SERVO_MAX = 0.5

# -------------------------------
# Serial Communication Setup
# -------------------------------
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE   = 9600


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
speed = 100
steer = 0
print("Starting obstacle detection and motor drive loop...")
# Open serial communication with Arduino.
with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as arduino:
        time.sleep(0.1)  # Wait briefly for the serial port to initialize.
        if arduino.isOpen():
            print(f"{arduino.port} connected!")
            startup(arduino, speed) # sets speed to {speed} after starting speed at {speed+30} for 0.2s
            # Set initial servo position (straight ahead).
            servo.value = steer
            t_0 = time.time()
            fwd_action(arduino)
            sleep(2)
            try:
                while True:
                    # Read distances from both ultrasonic sensors.
                    left_distance = left_sensor.distance   # in meters
                    right_distance = right_sensor.distance # in meters
                    #front_distance = front_sensor.distance # in meters

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
                        
                    print("Left distance: {:.2f} m, Right distance: {:.2f} m, Steer: {:.2f}, fwd_dist {:.2f}, driving: {}".format(left_distance, right_distance, steer, fwd_dist, driving))

                    # Decide on action based on sensor readings.
                    t_now = time.time()
                    print(f"\Delta t = {t_now - t_0}")
                    if driving and (left_distance < THRESHOLD_DISTANCE_LR and right_distance < THRESHOLD_DISTANCE_LR):
                        print("stop")
                        driving = False
                        stop_action(arduino)
                    elif (t_now - t_0) < STEER_SLEEP_LEN: 
                        print("pass_pre")
                        pass
                        print("passed")
                    elif right_distance < THRESHOLD_DISTANCE_LR:
                        print("steer left")
                        if(steer <= SERVO_MAX):
                            steer += STEER_INCREMENT
                        servo.value = steer
                        t_0 = t_now
                    elif left_distance < THRESHOLD_DISTANCE_LR:
                        print("steer right")
                        if steer >= (-1*SERVO_MAX):
                            steer -= STEER_INCREMENT
                        servo.value = steer
                        t_0 = t_now
                    elif not driving:
                        print("start up")
                        driving = True
                        fwd_action(arduino)
                    else:
                        print("continue")
                    # edit video with relevant information:
                    clean_recording.write(frame_out)

                    cv.putText(frame_out,
                        f"Closest: {fwd_dist:.2f} {'m' if odo_dist else 'units'}",
                        (20,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv.putText(frame_out, 
                        f"left_sensor: {left_distance: .2f} || right_sensor: {right_distance: .2f}", 
                        (20,60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv.putText(frame_out,
                        f"steer: {steer: .2f} || speed: {speed}",
                        (20,90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv.putText(frame_out,
                        f"driving: {driving}",
                        (20,120),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    
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
