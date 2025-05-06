#!/usr/bin/env python3
import cv2
import numpy as np

# -------------------------------
# Camera Setup
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -------------------------------
# Vision Params
# -------------------------------
THRESH_VAL   = 60
ROI_Y_START  = 0.5
# Small kernel to knock out tiny specks
K_OPEN       = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# Larger kernel to fill gaps in the tape
K_CLOSE      = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop ROI
    y0 = int(h * ROI_Y_START)
    roi_color = frame[y0:h, :].copy()
    roi_gray  = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    # 1) Blur + Threshold
    blur = cv2.GaussianBlur(roi_gray, (5,5), 0)
    _, mask = cv2.threshold(blur, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)

    # 2) Morphological Opening -> removes small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_OPEN, iterations=1)
    # 3) Morphological Closing -> fills gaps in line
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K_CLOSE, iterations=1)

    # 4) Keep only the largest connected component
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        # fill the largest contour
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)

    # 5) Create a black‐background RGB image and paint only the tape pixels
    path_only = np.zeros_like(roi_color)
    path_only[clean_mask==255] = roi_color[clean_mask==255]
    # draw the contour outline in green
    if contours:
        cv2.drawContours(path_only, [largest], -1, (0,255,0), 2)

    # 6) Composite back into full‐frame for display
    output = np.zeros_like(frame)
    output[y0:h, :] = path_only
    cv2.rectangle(output, (0,y0), (w,h), (255,0,0), 1)

    # 7) Show result
    cv2.imshow("Isolated Path", output)
    cv2.imshow("Filtered Mask", clean_mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
