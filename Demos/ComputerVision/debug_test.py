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
    """Send cmd and print it for debug."""
    arduino.write(cmd.encode())
    print(f"[ARDUINO] → {cmd}")

def forward():
    send_cmd("FORWARD")

def stop():
    send_cmd("STOP")

def setSpeed(speed: int):
    send_cmd(f"SPEED {speed}")

def setSteer(steer: int):
    send_cmd(f"STEER {steer}")

# -------------------------------
# Camera & Logging Setup
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# lower res for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# video writers: clean ROI on left, debug overlay on right
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
clean_out = cv2.VideoWriter('clean.avi', fourcc, 15, (w, int(h//2)))
debug_out = cv2.VideoWriter('debug.avi', fourcc, 15, (w, h))

# -------------------------------
# Line-Follower Params
# -------------------------------
SPEED           = 70
STEER_CENTER    = 90
Kp              = 0.3
STEER_LEFT_MAX  = 110
STEER_RIGHT_MAX =  70

THRESH_VAL      = 60
lost_counter    = 0
LOST_THRESHOLD  = 10
ROI_Y_START     = 0.5

# initialize
setSpeed(SPEED)
forward()
sleep(0.2)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # draw ROI box on full frame
        y0 = int(h * ROI_Y_START)
        cv2.rectangle(frame, (0,y0), (w,h), (255,0,0), 2)

        # crop ROI
        roi = frame[y0:h, :]

        # 1) preprocess
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # 2) threshold
        _, mask = cv2.threshold(blur, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)

        # 3) morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 4) contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug = roi.copy()
        if contours:
            lost_counter = 0
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"]>0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                # draw contour + centroid
                cv2.drawContours(debug, [c], -1, (0,255,0), 2)
                cv2.circle(debug, (cx, cy), 5, (0,0,255), -1)
                # steering
                error = cx - (w/2)
                steer = int(STEER_CENTER - Kp * error)
                steer = max(min(steer, STEER_LEFT_MAX), STEER_RIGHT_MAX)
                setSteer(steer)
                forward()
                # draw arrow
                arrow_end = (int(w/2 - Kp*error), cy)
                cv2.arrowedLine(debug, (w//2, cy), arrow_end, (255,0,0), 2)
                cv2.putText(debug, f"err={error:.1f}", (10,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
                cv2.putText(debug, f"steer={steer}", (10,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        else:
            lost_counter += 1
            stop()
            sleep(0.05)
            if lost_counter > LOST_THRESHOLD:
                print("[INFO] End of path detected.")
                stop()
                break

        # show & write outputs
        cv2.imshow("Full Debug", frame)
        cv2.imshow("ROI Debug", debug)
        cv2.imshow("Mask", mask)

        # record
        clean_out.write(roi)
        debug_full = cv2.vconcat([frame, cv2.resize(debug, (w,h))])[:h, :w]  
        debug_out.write(debug_full)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    stop()
    clean_out.release()
    debug_out.release()
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
