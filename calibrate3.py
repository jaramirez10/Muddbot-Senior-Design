import cv2 as cv
import numpy as np
import glob
import yaml
import os

# Parameters
pattern_size = (9, 6)
square_size = 30  # mm or any consistent unit

# Create object points: (0,0,0), (1,0,0), (2,0,0) ... * square_size
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2) * square_size

objpoints = []
imgpoints = []

# Create output directory for visuals
os.makedirs('camera_calibration_visuals', exist_ok=True)

# Load calibration images
for idx, fname in enumerate(glob.glob('camera_calibration_photos/*.png')):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, pattern_size,
                                            cv.CALIB_CB_ADAPTIVE_THRESH +
                                            cv.CALIB_CB_NORMALIZE_IMAGE)
    if not ret:
        print(f"Chessboard not found in {fname}")
        continue

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    objpoints.append(objp)
    imgpoints.append(corners2)

    # Draw and save the visual result
    vis_img = img.copy()
    cv.drawChessboardCorners(vis_img, pattern_size, corners2, ret)
    out_path = f'camera_calibration_visuals/corners_{idx:02}.png'
    cv.imwrite(out_path, vis_img)
    print(f"Saved: {out_path}")

# Calibrate
ret, K, distCoeffs, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# Save calibration parameters
data = {
    'K':        K.tolist(),
    'dist':     distCoeffs.flatten().tolist()
}

with open('calib.yaml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False)

print("Calibration complete. Parameters saved to calib.yaml.")
# Create folder for reprojection error visuals
os.makedirs('camera_reprojection_error_visuals', exist_ok=True)

# Visualize reprojection errors
for idx, (img_path, objp, imgp, rvec, tvec) in enumerate(zip(
        glob.glob('camera_calibration_photos/*.png'),
        objpoints, imgpoints, rvecs, tvecs)):

    img = cv.imread(img_path)
    reprojected_points, _ = cv.projectPoints(objp, rvec, tvec, K, distCoeffs)

    for i in range(len(imgp)):
        pt_actual = tuple(np.round(imgp[i, 0]).astype(int))
        pt_reproj = tuple(np.round(reprojected_points[i, 0]).astype(int))

        # Actual point (green)
        cv.circle(img, pt_actual, 4, (0, 255, 0), -1)

        # Reprojected point (red)
        cv.circle(img, pt_reproj, 4, (0, 0, 255), -1)

        # Line connecting them (blue)
        cv.line(img, pt_actual, pt_reproj, (255, 0, 0), 1)

    out_path = f'camera_reprojection_error_visuals/reproj_{idx:02}.png'
    cv.imwrite(out_path, img)
    print(f"Saved reprojection visualization: {out_path}")