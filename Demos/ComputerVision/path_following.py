#!/usr/bin/env python3
import cv2
import numpy as np
import serial
import time
from time import sleep

from utils.utils import *

THRESH_VAL       = 60       # binary inverse threshold for black tape
MIN_CONTOUR_AREA = 500      # ignore contours smaller than this (in pixels)
BLUR_KERNEL      = (5, 5)

# small proportional gain for steering
KP_STEER = 0.1

# -------------------------------
# Arduino / Motor Control Setup
# -------------------------------
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE   = 115200

arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(0.1)

def forward():
    send_command(arduino, "FORWARD")
def stop():
    send_command(arduino, "STOP")
def setSpeed(speed):
    send_command(arduino, f"SPEED {speed}")
def setSteer(steer):
    send_command(arduino, f"STEER {steer}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

y_crop_percentage = 0.2
x_crop_percentage = 0.5

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

y_0 = int(frame_height * y_crop_percentage)
x_mid = frame_width / 2
x_left = int(x_mid - (frame_width * x_crop_percentage / 2))
x_right = int(x_mid + (frame_width * x_crop_percentage / 2))

ret, first = cap.read()
h, w = first.shape[:2]
roi_first = first[y_0:h, x_left:x_right].copy()

cropped_size = (roi_first.shape[1], roi_first.shape[0])  # (width, height)
                
clean_recording = cv2.VideoWriter('clean.avi',  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        10, size) 
final_masked_recording = cv2.VideoWriter('masked_recording.avi',  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        10, cropped_size) 

print(f"y_0: {y_0}, x_left: {x_left}, x_right: {x_right}")

speed = 70
setSpeed(speed)
forward()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        # 1) crop
        roi_color = frame[y_0:h, x_left:x_right].copy()

        # 2) grayscale→blur→threshold
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(roi_gray, BLUR_KERNEL, 0)
        _, mask = cv2.threshold(blur, THRESH_VAL, 255,
                                cv2.THRESH_BINARY_INV)

        # 3) opening + closing
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)

        # 4) contour detection
        contours, _ = cv2.findContours(mask_clean,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        display = np.zeros_like(roi_color)

        # --- NEW: Steering logic based on largest contour centroid ---
        if contours:
            # pick the largest
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > MIN_CONTOUR_AREA:
                # draw it
                cv2.drawContours(display, [c], -1, (0,255,0), 2)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    cv2.circle(display, (cx, cy), 5, (0,0,255), -1)

                    # compute error relative to center of ROI
                    roi_w = roi_color.shape[1]
                    error = cx - (roi_w // 2)

                    # map error → steer angle around 90°
                    steer = int(90 - KP_STEER * error)
                    steer = max(70, min(110, steer))  # clip to [70,110]
                    setSteer(steer)
                    cv2.putText(display, f"Steer:{steer}", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255,255,255), 2)
        else:
            # no line found: you could stop or keep previous steering
            stop()
            time.sleep(0.5)
            forward()
        # ------------------------------------------------------------**

        # 5) record each stage
        clean_recording.write(frame)
        final_masked_recording.write(cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR))

        # 6) display if desired
        cv2.imshow("Cropped ROI", display)
        cv2.imshow("Mask", mask_clean)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    stop()
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
