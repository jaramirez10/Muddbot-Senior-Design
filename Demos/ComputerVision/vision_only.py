import cv2
import numpy as np

# --- Configurable Parameters ---
ROI_Y_START     = 0.5       # fraction from top of frame
THRESH_VAL      = 60        # binary inverse threshold for black tape
MIN_CONTOUR_AREA = 500      # noise filter
BLUR_KERNEL     = (5, 5)

# --- Camera Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open camera.")

# Set small resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print(" Camera initialized. Press ESC to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Frame capture failed.")
            continue

        h, w = frame.shape[:2]
        y0 = int(h * ROI_Y_START)
        if y0 >= h:
            print("⚠️ ROI start exceeds frame height.")
            continue

        # Crop to ROI
        roi_color = frame[y0:h, :].copy()
        if roi_color.size == 0:
            print("⚠️ Empty ROI.")
            continue

        # --- Preprocess ---
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(roi_gray, BLUR_KERNEL, 0)
        _, mask = cv2.threshold(blur, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)

        # Optional: clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # --- Contour Detection ---
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        display = np.zeros_like(roi_color)  # black background

        for c in contours:
            area = cv2.contourArea(c)
            if area > MIN_CONTOUR_AREA:
                cv2.drawContours(display, [c], -1, (0, 255, 0), 2)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

        # Show outputs
        cv2.imshow("Contours", display)
        cv2.imshow("Binary Mask", mask_clean)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    print("🧹 Cleaning up.")
    cap.release()
    cv2.destroyAllWindows()
