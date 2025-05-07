#!/usr/bin/env python3
import cv2
import numpy as np
import serial
import time
from time import sleep
from utils.utils import *
# -------------------------------
# Configuration Parameters
# -------------------------------
# --- Configurable Parameters ---
THRESH_VAL        = 60       # binary inverse threshold for black tape
MIN_CONTOUR_AREA  = 500      # ignore contours smaller than this (in px)
BLUR_KERNEL       = (5, 5)
KP_STEER          = 0.1      # steering gain

Y_CROP_PCT        = 0.4      # crop bottom 80% of frame
X_CROP_PCT        = 0.8      # crop center 50% of width
CENTER_TOLERANCE  = 10       # px tolerance for “straight”

# -------------------------------
# Arduino / Motor Control Setup
# -------------------------------
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE   = 115200

# Open serial port
arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(0.1)  # let Arduino reset

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


# -------------------------------
# Camera Setup
# -------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

# Reduce resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Precompute ROI bounds in pixels
y0      = int(fh * Y_CROP_PCT)
x_mid   = fw // 2
x_left  = int(x_mid - (fw * X_CROP_PCT / 2))
x_right = int(x_mid + (fw * X_CROP_PCT / 2))

print(f"[INFO] ROI y0={y0}, x=[{x_left},{x_right}]")
print("Press ESC to quit.")

# Kick off the car
setSpeed(60)
forward()
steer = 90
steer_increment = 3

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

ret, first = cap.read()
h, w = first.shape[:2]
roi_first = first[0:y0, x_left:x_right].copy()

cropped_size = (roi_first.shape[1], roi_first.shape[0])  # (width, height)
clean_recording = cv2.VideoWriter('clean.avi',  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        10, size) 

cropped_recording = cv2.VideoWriter('cropped.avi',  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        10, cropped_size) 

pre_processed_recording = cv2.VideoWriter('pre_processed.avi',  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        10, cropped_size) 

final_masked_recording = cv2.VideoWriter('masked_recording.avi',  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        10, cropped_size) 
try:
    while True:
        # 1) Grab a frame
        ret, frame = cap.read()
        if not ret:
            continue

        # 2) Crop to the region where the tape should appear
        roi = frame[0:y0, x_left:x_right].copy()

        # 3) Preprocess: grayscale & blur for threshold stability
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

        # 4) Binary segmentation: black tape → white mask
        _, mask = cv2.threshold(blur, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)

        # 5) Morphological cleaning
        ker_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        ker_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker_close)

        # 6) Contour detection
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 7) Prepare a color debug view of the ROI
        disp = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        direction = "STRAIGHT"

        if cnts:
            # 8) Choose the largest contour (assumed to be the tape)
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > MIN_CONTOUR_AREA:
                # Draw the contour outline
                cv2.drawContours(disp, [c], -1, (0,255,0), 2)

                # 9) Compute its centroid
                M = cv2.moments(c)
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                cv2.circle(disp, (cx, cy), 6, (255,0,0), -1)

                # 10) Steering law: error = centroid offset from ROI center
                rw = disp.shape[1]
                error = cx - (rw//2)
                steer = int(90 - KP_STEER * error)
                steer = max(70, min(110, steer))  # clip to safe range

                # 11) Send steering command
                setSteer(steer)

                # 12) Decide textual direction
                if   error >  CENTER_TOLERANCE: direction = "LEFT"
                elif error < -CENTER_TOLERANCE: direction = "RIGHT"

                # 13) Draw an arrow showing the correction vector
                start = (rw//2, disp.shape[0]//2)
                end   = (cx, cy)
                cv2.arrowedLine(disp, start, end, (0,0,255), 2, tipLength=0.2)
        else:
            # 14) If no tape found, stop the car
            stop()

        if direction == "LEFT":
            steer += steer_increment
        elif direction == "RIGHT":
            steer -= steer_increment
        elif direction == "STRAIGHT":
            steer = 90

        setSteer(steer)

        # 15) Annotate the direction text
        cv2.putText(disp, direction, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(disp, f"STEER: {steer}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        


        blur_bgr = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

        # --- Display ---
        clean_recording.write(frame)
        cropped_recording.write(roi)
        pre_processed_recording.write(blur_bgr)
        final_masked_recording.write(disp)
        # 17) Exit cleanly on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # 18) Cleanup: stop car, release camera, close windows
    stop()
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
