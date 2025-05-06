#!/usr/bin/env python3
import cv2
import numpy as np

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Parameters
THRESH_VAL   = 60
ROI_Y_START  = 0.5
MORPH_KERNEL = (5,5)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[WARN] Empty frame, skipping")
        continue

    # Recompute in case resolution changed
    h, w = frame.shape[:2]
    y0 = int(h * ROI_Y_START)

    # Guard: ensure ROI is valid
    if y0 >= h or y0 < 0:
        print(f"[ERROR] ROI start {y0} out of bounds for height {h}")
        break

    roi_color = frame[y0:h, :]
    if roi_color.size == 0:
        print("[WARN] ROI is empty, skipping")
        continue

    # Convert to gray safely
    roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    # The rest of your pipeline:
    blur = cv2.GaussianBlur(roi_gray, (5,5), 0)
    _, mask = cv2.threshold(blur, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create isolated-path image
    path_only = np.zeros_like(roi_color)
    if contours:
        # choose the best contour (e.g. by area or centroid-proximity):
        c = max(contours, key=cv2.contourArea)
        # copy only the path pixels
        path_only[mask == 255] = roi_color[mask == 255]
        cv2.drawContours(path_only, [c], -1, (0,255,0), 2)

    # Build a black background frame and insert path_only
    output = np.zeros_like(frame)
    output[y0:h, :] = path_only
    cv2.rectangle(output, (0,y0), (w,h), (255,0,0), 1)

    cv2.imshow("Path Isolated (black bg)", output)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
