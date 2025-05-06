#!/usr/bin/env python3
import cv2
import numpy as np

# -------------------------------
# Camera Setup
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# Optionally set a lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# -------------------------------
# Vision Parameters
# -------------------------------
THRESH_VAL   = 60        # Black tape threshold
ROI_Y_START  = 0.5       # Process bottom 50% of the frame
MORPH_KERNEL = (5, 5)    # Clean-up kernel size

# -------------------------------
# Main Loop
# -------------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        y0 = int(h * ROI_Y_START)

        # 1) Crop ROI
        roi_color = frame[y0:h, 0:w]
        roi_gray  = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

        # 2) Blur + Threshold (black→white mask)
        blur = cv2.GaussianBlur(roi_gray, (5,5), 0)
        _, mask = cv2.threshold(blur, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)

        # 3) Morphological Close to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 4) Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5) Draw the largest contour
        debug = roi_color.copy()
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(debug, [largest], -1, (0,255,0), 2)

        # 6) Overlay ROI back onto full frame
        output = frame.copy()
        output[y0:h, 0:w] = debug
        cv2.rectangle(output, (0,y0), (w,h), (255,0,0), 2)  # ROI boundary

        # 7) Show windows
        cv2.imshow("Live Contour Outline", output)
        cv2.imshow("Mask", mask)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
