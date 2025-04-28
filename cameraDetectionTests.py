#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RC Car with Dual Ultrasonic + USB Camera (Red‑object) Fusion
"""
import cv2 as cv
import numpy as np
from time import sleep
import time


cap = cv.VideoCapture(0) # find usb port
if not cap.isOpened():
    raise IOError("Cannot open USB camera")

# Example function to detect red objects
def detect_red_obstacle(frame):
    """
    Returns True if a large red object is detected in the frame.
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # red hue can wrap around, so mask two ranges
    lower1 = np.array([  0, 120,  70])
    upper1 = np.array([ 10, 255, 255])
    lower2 = np.array([170, 120,  70])
    upper2 = np.array([180, 255, 255])
    mask = cv.inRange(hsv, lower1, upper1) + cv.inRange(hsv, lower2, upper2)

    # Morphology to reduce noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)

    # Count non-zero (red) pixels
    red_count = cv.countNonZero(mask)
    # Debug: show mask if needed
    # cv.imshow("Red Mask", mask)
    return red_count > 2000  # tweak threshold to your environment

if __name__ == "__main__":
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break
            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break
            
            if detect_red_obstacle(frame):
                print("Vision: Red obstacle detected → stopping")
                continue
            else:
                print("no red object!")

    except KeyboardInterrupt:
        print("Exiting on user interrupt.")
    finally:
        cap.release()
        cv.destroyAllWindows()