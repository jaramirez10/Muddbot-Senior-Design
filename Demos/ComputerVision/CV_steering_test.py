#!/usr/bin/env python3
import cv2
import numpy as np

# --- Configurable Parameters ---
THRESH_VAL        = 60       # binary inverse threshold for black tape
MIN_CONTOUR_AREA  = 500      # ignore contours smaller than this (in px)
BLUR_KERNEL       = (5, 5)
KP_STEER          = 0.1      # steering gain

Y_CROP_PCT        = 0.4      # crop bottom 80% of frame
X_CROP_PCT        = 0.8      # crop center 50% of width
CENTER_TOLERANCE  = 10       # px tolerance for “straight”

# --- Stubbed Motor Functions ---
def forward():   print("[MOTOR] FORWARD")
def stop():      print("[MOTOR] STOP")
def setSpeed(s): print(f"[MOTOR] SPEED {s}")
def setSteer(s): print(f"[MOTOR] STEER {s}")

# --- Camera Setup ---
cap = cv2.VideoCapture(0, cv2.CAP_V4L)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

# Lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Compute ROI coords
y0      = int(fh * Y_CROP_PCT)
x_mid   = fw // 2
x_left  = int(x_mid - (fw * X_CROP_PCT / 2))
x_right = int(x_mid + (fw * X_CROP_PCT / 2))

print(f"[INFO] ROI y0={y0}, x=[{x_left},{x_right}]")
print("Press ESC to quit.")

# Start stubs
setSpeed(70)
forward()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 1) Crop ROI
        roi = frame[0:y0, x_left:x_right]

        # 2) Preprocess
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
        _, mask = cv2.threshold(blur, THRESH_VAL, 255,
                                cv2.THRESH_BINARY_INV)

        # 3) Morphology
        ko = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)
        kc = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)

        # 4) Find contours & prepare display
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        disp = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # show the actual ROI in color

        direction = "STRAIGHT"
        if cnts:
            # pick largest
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > MIN_CONTOUR_AREA:
                # draw contour
                cv2.drawContours(disp, [c], -1, (0,255,0), 2)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    # draw centroid
                    cv2.circle(disp, (cx, cy), 6, (0,0,255), -1)

                    # compute steering error & pseudo-steer
                    rw = disp.shape[1]
                    error = cx - (rw//2)
                    steer = int(90 - KP_STEER * error)
                    steer = max(70, min(110, steer))

                    # decide text direction
                    if   error >  CENTER_TOLERANCE: direction = "LEFT"
                    elif error < -CENTER_TOLERANCE: direction = "RIGHT"

                    # draw arrow from center→centroid
                    start = (rw//2, disp.shape[0]//2)
                    end   = (cx, cy)
                    cv2.arrowedLine(disp, start, end, (255,0,0), 2,
                                    tipLength=0.2)

                    # print stub steer
                    setSteer(steer)

        else:
            stop()

        # overlay direction text
        cv2.putText(disp, f"{direction}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255,255,255), 2)

        # 5) Show live feeds
        cv2.imshow("ROI + Contour", disp)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    stop()
    cap.release()
    cv2.destroyAllWindows()
