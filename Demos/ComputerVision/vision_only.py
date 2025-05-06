#!/usr/bin/env python3
import cv2
import numpy as np

# -------------------------------
# Camera Setup
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# lower res for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -------------------------------
# Vision Params
# -------------------------------
THRESH_VAL   = 60
ROI_Y_START  = 0.5
MORPH_KERNEL = (5, 5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # define ROI
    y0 = int(h * ROI_Y_START)
    roi_color = frame[y0:h, :].copy()
    roi_gray  = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    # 1) blur + threshold
    blur = cv2.GaussianBlur(roi_gray, (5,5), 0)
    _, mask = cv2.threshold(blur, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)

    # 2) clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3) find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4) create a black background and copy only the path region
    path_only = np.zeros_like(roi_color)          # black image same size
    # Use mask to copy color pixels where mask==255
    path_only[mask==255] = roi_color[mask==255]

    # 5) draw contour on the path_only image
    if contours:
        # filter by area & centroid proximity if needed...
        # here we just draw the largest one
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(path_only, [c], -1, (0,255,0), 2)

    # 6) composite back into full frame for display
    output = np.zeros_like(frame)
    output[y0:h, :] = path_only

    # draw ROI boundary
    cv2.rectangle(output, (0,y0), (w,h), (255,0,0), 1)

    # 7) show
    cv2.imshow("Path Isolated (black bg)", output)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
