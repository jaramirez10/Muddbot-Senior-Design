import cv2 as cv
import numpy as np
import yaml
import time
try:
    # LOAD CAMERA CALIBRATION
    with open('calib.yaml') as f:
        data = yaml.safe_load(f)
    K    = np.array(data['K'])
    dist = np.array(data['dist'])

    # odo_dist = 0.05  # example: car moved 5 cm between frames
    odo_dist = None


    # FEATURE DETECTOR & MATCHER 
    orb = cv.ORB_create(2000)
    bf  = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    # START VIDEO STREAM 
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")


    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv.VideoWriter('filename.avi',  
                            cv.VideoWriter_fourcc(*'MJPG'), 
                            10, size) 

    # Read first frame
    ret, prev_raw = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")
    prev_gray = cv.cvtColor(prev_raw, cv.COLOR_BGR2GRAY)
    prev_gray = cv.undistort(prev_gray, K, dist)
    kp_prev, des_prev = orb.detectAndCompute(prev_gray, None)

    while True:
        t0 = time.time()
        ret, raw = cap.read()
        if not ret:
            break

        # Undistort & grayscale
        gray = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
        gray = cv.undistort(gray, K, dist)

        # DETECT & MATCH FEATURES
        kp, des = orb.detectAndCompute(gray, None)
        if des is None or des_prev is None or len(des_prev) < 8:
            prev_gray, kp_prev, des_prev = gray, kp, des
            continue

        matches = bf.knnMatch(des_prev, des, k=2)
        good = [m for m,n in matches if m.distance < 0.75*n.distance]
        if len(good) < 8:
            prev_gray, kp_prev, des_prev = gray, kp, des
            continue

        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good])
        pts      = np.float32([   kp[m.trainIdx].pt    for m in good])

        # Draw matched keypoints on the current frame
        for m in good:
            x,y = kp[m.trainIdx].pt
            cv.circle(raw, (int(x),int(y)), 3, (0,0,255), -1)

        # ESSENTIAL MATRIX & POSE 
        E, mask = cv.findEssentialMat(pts_prev, pts, K,
                                    method=cv.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            prev_gray, kp_prev, des_prev = gray, kp, des
            cv.imshow('Live Distance + Features', raw)
            if cv.waitKey(1)&0xFF==27: break
            continue

        _, R, t, mask_pose = cv.recoverPose(E, pts_prev, pts, K)

        # Compute scale
        scale = 1.0
        if odo_dist is not None:
            norm_t = np.linalg.norm(t)
            if norm_t>1e-6:
                scale = odo_dist / norm_t

        # TRIANGULATION 
        P1 = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
        P2 = K.dot(np.hstack((R, scale*t)))
        pts4d = cv.triangulatePoints(P1, P2, pts_prev.T, pts.T)
        pts3d = (pts4d[:3] / pts4d[3]).T
        dists = np.linalg.norm(pts3d, axis=1)
        min_dist = np.min(dists)

        # DISPLAY DISTANCE & FEATURES
        text = f"Closest Obj: {min_dist:.2f} {'m' if odo_dist else 'units'}"
        cv.putText(raw, text, (20,30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv.imshow('Live Distance + Features', raw)
        result.write(raw)

        # Shift frames
        prev_gray, kp_prev, des_prev = gray, kp, des

        # Break on ESC
        if cv.waitKey(1) & 0xFF == 27:
            break

        # Maintain ~20 FPS
        dt = time.time() - t0
        if dt<0.05: time.sleep(0.05-dt)
except KeyboardInterrupt:
    print("Exiting on user interrupt.")
finally:
    result.release()
    cap.release()
    cv.destroyAllWindows()