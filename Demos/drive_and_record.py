import cv2 as cv
import numpy as np
import yaml
import time
import numpy as np

from time import sleep
from utils.utils import *
import serial


FWD_THRESH = 50 #cm
LR_THRESH_URGENT = 20 #cm
LR_THRESH_SUBTLE = 100 #cm

URGENT_STEER_LEN = 1.0 # in seconds
URGENT_STEER_VAL = 15 # degrees
SUBTLE_STEER_INCR = 5 # degrees
SUBTLE_STEER_ARR_LEN = 15 # array length
SUBTLE_STEER_DRIFT_THRESH = 5 #cm
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

def LR_dist_arrays(left_dist, right_dist, l_dists, r_dists, l_dists_num_elements, r_dists_num_elements):
    l_dists = np.roll(l_dists, 1)
    r_dists = np.roll(r_dists, 1)
    l_dists[0] = left_dist
    r_dists[0] = right_dist
    l_dists_num_elements = l_dists_num_elements + 1 if l_dists_num_elements < SUBTLE_STEER_ARR_LEN else SUBTLE_STEER_ARR_LEN
    r_dists_num_elements = r_dists_num_elements + 1 if r_dists_num_elements < SUBTLE_STEER_ARR_LEN else SUBTLE_STEER_ARR_LEN
    return l_dists, r_dists, l_dists_num_elements, r_dists_num_elements

def flush_LR_dist_arrays():
    l_dists = np.array([-1 for _ in range(SUBTLE_STEER_ARR_LEN)])
    r_dists = np.array([-1 for _ in range(SUBTLE_STEER_ARR_LEN)])
    l_dists_num_elements = 0
    r_dists_num_elements = 0
    return l_dists, r_dists, l_dists_num_elements, r_dists_num_elements

def drifting(lr_dists, lr_dists_num_elements):
    if lr_dists_num_elements < SUBTLE_STEER_ARR_LEN:
        return False
    elif lr_dists[0] + 5 >= lr_dists[lr_dists_num_elements - 1]:
        return False
    else:
        for i in range(lr_dists_num_elements - 1):
            if lr_dists[i+1] < lr_dists[i]:
                return False
        return True

l_dists, r_dists, l_dists_num_elements, r_dists_num_elements = flush_LR_dist_arrays()

driving = False
speed = 100
steer = 90
urgent_steering = False

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
                    left_dist, fwd_dist, right_dist = getLFRdists() # in cm
                    l_dists, r_dists, l_dists_num_elements, r_dists_num_elements = LR_dist_arrays(left_dist, right_dist, l_dists, r_dists, l_dists_num_elements, r_dists_num_elements)

                    ret, frame = cap.read()
                    # frame_out is the video frame (overlayed with the corners)
                    # dists is the distances of all the corners
                    # the rest are internal variables
                    frame_out, dists, prev_gray, kp_prev, des_prev = process_frame(
                    frame, prev_gray, kp_prev, des_prev, K, dist, orb, bf, odo_dist=None)
                        
                    t_now = time.time()
                    print("Left distance: {:.2f} cm, fwd_dist {:.2f} cm, Right distance: {:.2f} cm, Steer: {:.2f},  driving: {}, delta t = {:.2f}".format(left_dist, fwd_dist, right_dist, steer, driving, (t_now - t_0)))
                    print(f"l_dists_num_elements: {l_dists_num_elements} -- l_dists: {l_dists} -- drift? {drifting(l_dists, l_dists_num_elements)}")
                    print(f"r_dists_num_elements: {l_dists_num_elements} -- r_dists: {r_dists} -- drift? {drifting(r_dists, r_dists_num_elements)}")
                
                    # Decide on action based on sensor readings.
                    if driving and (fwd_dist < FWD_THRESH or (left_dist < LR_THRESH_URGENT and right_dist < LR_THRESH_URGENT)):
                        driving = False
                        stop()
                        print("obstacle encountered -- STOPPING")
                    elif driving and urgent_steering and (t_now - t_0) > URGENT_STEER_LEN:  # it is amenable to turn now
                        print("setting course straight")
                        steer = 90
                        setSteer(steer)
                        urgent_steering = False
                    elif driving and not urgent_steering and left_dist < LR_THRESH_URGENT:
                        print("urgently steering right")
                        steer = 90 - URGENT_STEER_VAL
                        setSteer(steer)
                        urgent_steering = True
                        t_0 = t_now
                    elif driving and not urgent_steering and right_dist < LR_THRESH_URGENT:
                        print("urgently steering left")
                        steer = 90 + URGENT_STEER_VAL
                        setSteer(steer)
                        urgent_steering = True
                        t_0 = t_now
                    elif driving and not urgent_steering:
                        if left_dist < LR_THRESH_SUBTLE and drifting(l_dists, l_dists_num_elements):
                            print("subtle steering right due to left drift")
                            steer = steer - SUBTLE_STEER_INCR
                            setSteer(steer)
                            l_dists, r_dists, l_dists_num_elements, r_dists_num_elements = flush_LR_dist_arrays()
                        elif right_dist < LR_THRESH_SUBTLE and drifting(r_dists, r_dists_num_elements):
                            print("subtle steering left due to right drift")
                            steer = steer + SUBTLE_STEER_INCR
                            setSteer(steer)
                            l_dists, r_dists, l_dists_num_elements, r_dists_num_elements = flush_LR_dist_arrays()
                        else:
                            print("just keep swimmin\'")
                    elif not driving and not (fwd_dist < FWD_THRESH or (left_dist < LR_THRESH_URGENT and right_dist < LR_THRESH_URGENT)):
                        driving = True
                        forward()
                        print("start driving again")
                    else:
                        print("staying stopped")

                    # edit video with relevant information:
                    clean_recording.write(frame_out)

                    cv.putText(frame_out,
                        f"forward_sensor: {fwd_dist:.2f} cm",
                        (20,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv.putText(frame_out, 
                        f"left_sensor: {left_dist: .2f} cm|| right_sensor: {right_dist: .2f} cm", 
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
