#!/usr/bin/env python3
import cv2
import numpy as np
import serial
import time
from time import sleep


THRESH_VAL       = 60       # binary inverse threshold for black tape
MIN_CONTOUR_AREA = 500      # ignore contours smaller than this (in pixels)
BLUR_KERNEL      = (5, 5)


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

    
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
clean_recording = cv2.VideoWriter('clean.avi',  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        10, size) 

cropped_recording = cv2.VideoWriter('cropped.avi',  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        10, size) 

pre_processed_recording = cv2.VideoWriter('pre_processed.avi',  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        10, size) 

final_masked_recording = cv2.VideoWriter('masked_recording.avi',  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        10, size) 


y_crop_percentage = 0.4
x_crop_percentage = 0.4

y_0 = int(frame_height * y_crop_percentage)
x_mid = frame_width / 2
x_left = int(x_mid - (frame_width * x_crop_percentage / 2))
x_right = int(x_mid + (frame_width * x_crop_percentage / 2))

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
        roi_color = frame[y_0:h, x_left:x_right].copy()

        
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(roi_gray, BLUR_KERNEL, 0)
        _, mask = cv2.threshold(blur, THRESH_VAL, 255,
                                cv2.THRESH_BINARY_INV)

        cv2.imshow("blur", blur)
        cv2.imshow("mask", mask)

        # --- Morphological Opening (remove small specks) ---
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        cv2.imshow("mask_open", mask_open)
        
        # --- Morphological Closing (fill small holes) ---
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)


        # --- Contour Detection ---
        contours, _ = cv2.findContours(mask_clean,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        display = np.zeros_like(roi_color)  # black background

        for c in contours:
            area = cv2.contourArea(c)
            if area > MIN_CONTOUR_AREA:
                cv2.drawContours(display, [c], -1,
                                 (0, 255, 0), 2)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(display, (cx, cy),
                               5, (0, 0, 255), -1)
                    
       
        # --- Display ---
        clean_recording.write(frame)
        cropped_recording.write(roi_color)
        pre_processed_recording.write(blur)
        final_masked_recording.write(mask_clean)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    stop()
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
