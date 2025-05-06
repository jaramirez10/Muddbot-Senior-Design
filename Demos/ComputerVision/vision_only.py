#!/usr/bin/env python3
import cv2
import numpy as np

# -------------------------------
# Synthetic Line-Follower Test
# -------------------------------

# frame size
w, h = 320, 240

# controller constants
Kp              = 0.3
STEER_CENTER    = 90
STEER_LEFT_MAX  = 110
STEER_RIGHT_MAX =  70

# vision constants
THRESH_VAL   = 60
ROI_Y_START  = 0.5  # bottom 50% of image

def process_synthetic_line(offset_px, angle_deg=0):
    """
    Generate a synthetic frame with a black line offset by offset_px
    and at angle_deg, then run the line-follower pipeline.
    Returns (steer, debug_roi, mask) for display.
    """
    # 1) blank white frame
    frame = np.full((h, w), 255, dtype=np.uint8)

    # 2) draw thick black line in ROI center
    y0 = int(h * ROI_Y_START)
    cx = w//2 + offset_px
    length = h
    rad = np.deg2rad(angle_deg)
    dx = int((length/2) * np.cos(rad))
    dy = int((length/2) * np.sin(rad))
    x1, y1 = cx - dx, y0 - dy
    x2, y2 = cx + dx, y0 + dy
    cv2.line(frame, (x1, y1), (x2, y2), 0, 10)

    # 3) crop ROI
    roi = frame[y0:h, :]

    # 4) blur + threshold
    blur = cv2.GaussianBlur(roi, (5,5), 0)
    _, mask = cv2.threshold(blur, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)

    # 5) morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 6) find contours and compute steer
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    steer = None

    if contours:
        # pick largest contour
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx2 = int(M["m10"]/M["m00"])
            error = cx2 - (w/2)
            steer = int(STEER_CENTER - Kp * error)
            steer = max(min(steer, STEER_LEFT_MAX), STEER_RIGHT_MAX)

            # draw contour, centroid, centerline, and arrow
            cv2.drawContours(debug_roi, [c], -1, (0,255,0), 2)
            cy = (h - y0)//2
            cv2.circle(debug_roi, (cx2, cy), 5, (0,0,255), -1)
            cv2.line(debug_roi, (w//2, 0), (w//2, h-y0), (255,0,0), 2)
            arrow_end = (int(w/2 - Kp*error), cy)
            cv2.arrowedLine(debug_roi, (w//2, cy), arrow_end, (255,0,0), 2)
            cv2.putText(debug_roi, f"err={error:.1f}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(debug_roi, f"steer={steer}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    return steer, debug_roi, mask

def main():
    offsets = [-80, -40, 0, 40, 80]
    for off in offsets:
        steer, roi_vis, mask = process_synthetic_line(off, angle_deg=0)
        print(f"Line offset={off:>3} px → steer={steer}")

        # show the debug ROI and mask windows
        cv2.imshow("Synthetic ROI Debug", roi_vis)
        cv2.imshow("Synthetic Mask", mask)

        # wait until a key is pressed (0 = infinite). ESC will break.
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC key
            print("Test aborted by user.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
