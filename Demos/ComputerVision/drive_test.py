#!/usr/bin/env python3
import cv2
import numpy as np
import serial
import time
from time import sleep

# -------------------------------
# Arduino / Motor Control Setup
# -------------------------------
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE   = 115200

arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(0.1)

def send_cmd(cmd: str):
    arduino.write(cmd.encode())

def forward():
    send_cmd("FORWARD")

def stop():
    send_cmd("STOP")

def setSpeed(speed: int):
    send_cmd(f"SPEED {speed}")

def setSteer(steer: int):
    send_cmd(f"STEER {steer}")

# -------------------------------
# Camera & Line-Follower Params
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# set a modest resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# steering constants
SPEED           = 70
STEER_CENTER    = 90
Kp              = 0.3     # proportional gain
STEER_LEFT_MAX  = 110
STEER_RIGHT_MAX =  70

# threshold for binary inverse (black tape → white mask)
THRESH_VAL      = 60

# lost-line logic
lost_counter    = 0
LOST_THRESHOLD  = 10      # frames before we declare end-of-path

# region of interest (fraction of frame height)
ROI_Y_START     = 0.5     # start ROI at 50% down the frame

# -------------------------------
# Initialize Motors
# -------------------------------
setSpeed(SPEED)
forward()
sleep(0.2)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        roi = frame[int(h*ROI_Y_START):h, :]

        # 1) preprocess
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # 2) threshold (black → white)
        _, mask = cv2.threshold(blur, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)

        # 3) clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 4) find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            lost_counter = 0

            # 5) largest contour centroid
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # draw for debug
                cv2.circle(roi, (cx, cy), 5, (0,255,0), -1)
                cv2.line(roi, (w//2, 0), (w//2, int(h*(1-ROI_Y_START))), (255,0,0), 2)

                # 6) compute steering
                error = cx - (w / 2)
                steer = int(STEER_CENTER - Kp * error)
                steer = max(min(steer, STEER_LEFT_MAX), STEER_RIGHT_MAX)
                setSteer(steer)
                forward()
        else:
            lost_counter += 1
            # no line: stop briefly or continue trying
            stop()
            sleep(0.05)

            if lost_counter > LOST_THRESHOLD:
                # assume end of path
                print("End of path detected. Stopping.")
                stop()
                break

        # 7) debug display
        cv2.imshow("ROI", roi)
        cv2.imshow("Mask", mask)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    stop()
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
