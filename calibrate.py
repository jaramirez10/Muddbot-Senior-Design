import cv2 as cv
import numpy as np
import glob
import yaml

# 1) prepare the known 3D points of the chessboard corners
#    e.g. a 9×6 board → 9 inner corners wide, 6 high
pattern_size = (9,6)
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
# assume each square is “1 unit”; if you want real-world scale, multiply by square_size_mm
objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2) * 30

objpoints = []   # 3D points in world space
imgpoints = []   # 2D points in image plane

# 2) load all your calibration images
for fname in glob.glob('camera_calibration_photos/*.jpg'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 3) find chessboard corners
    ret, corners = cv.findChessboardCorners(gray, pattern_size,
                                            cv.CALIB_CB_ADAPTIVE_THRESH
                                            + cv.CALIB_CB_NORMALIZE_IMAGE)
    if not ret:
        continue

    # 4) refine corner locations to subpixel accuracy
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    # 5) collect corresponding points
    objpoints.append(objp)       # same objp for every image
    imgpoints.append(corners2)   # corners2.shape → (N_corners, 1, 2)

# 6) calibrate
ret, K, distCoeffs, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

data = {
    'K':        K.tolist(),
    'dist':     distCoeffs.flatten().tolist()
    'rvecs':    rvecs.toList()
    'tvecs':    tvecs.toList()
}

with open('calib.yaml', 'w') as f:
    # default_flow_style=False makes it more human-readable
    yaml.dump(data, f, default_flow_style=False)