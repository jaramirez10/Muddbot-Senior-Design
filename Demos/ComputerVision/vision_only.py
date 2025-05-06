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

        # Crop ROI
        roi_color = frame[y0:h, :].copy()
        roi_gray  = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

        # Preprocess & mask
        blur = cv2.GaussianBlur(roi_gray, (5,5), 0)
        _, mask = cv2.threshold(blur, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find all contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug = roi_color.copy()
        best = None
        best_err = float('inf')
        min_area = (h - y0) * w * 0.02  # e.g. at least 2% of ROI area

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            cx = int(M['m10']/M['m00'])
            err = abs(cx - (w//2))
            # pick the contour whose centroid is closest to center
            if err < best_err:
                best_err = err
                best = c

        if best is not None:
            # draw only that one
            cv2.drawContours(debug, [best], -1, (0,255,0), 2)
            M = cv2.moments(best)
            cx = int(M['m10']/M['m00'])
            cy = int((h-y0)/2)
            cv2.circle(debug, (cx, cy), 5, (0,0,255), -1)

        # Overlay and display
        output = frame.copy()
        output[y0:h, :] = debug
        cv2.rectangle(output, (0,y0), (w,h), (255,0,0), 2)

        cv2.imshow("Live Contour Outline", output)
        cv2.imshow("Mask", mask)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
