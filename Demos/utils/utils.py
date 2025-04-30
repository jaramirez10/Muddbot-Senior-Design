import cv2 as cv
import numpy as np
import yaml
import time
import numpy as np
from gpiozero import Servo, DistanceSensor
from time import sleep
import serial

def send_command(arduino, cmd):
    """Send a command string to the Arduino over serial."""
    arduino.write((cmd).encode())
    print(f"Sent command:_{cmd}_")


def fwd_action(arduino):
    """
    With no obstacles, command the RC car to move forward.
    """
    send_command(arduino, "FORWARD")

def stop_action(arduino):
    """
    If obstacles are detected on both sides,
    choose a default action, such as stopping.
    """
    print("Obstacles detected on both sides. Stopping.")
    send_command(arduino, "STOP")
    sleep(1)

import cv2 as cv
import numpy as np

def process_frame(raw,
                  prev_gray,
                  kp_prev,
                  des_prev,
                  K,
                  dist,
                  orb,
                  bf,
                  odo_dist=None):
    """
    Process one new frame for live ORB‐based depth estimation.

    Args:
      raw       : current BGR frame from cv.VideoCapture.read()
      prev_gray : previous undistorted grayscale frame
      kp_prev   : ORB keypoints from previous frame
      des_prev  : ORB descriptors from previous frame
      K, dist   : camera intrinsics & distortion coeffs
      orb       : pre‐created cv.ORB detector
      bf        : pre‐created cv.BFMatcher
      odo_dist  : optional real‐world movement per frame (for metric scale)

    Returns:
      raw_out   : BGR frame with overlaid matched keypoints & distance text
      dists     : numpy array of depths (up to scale) for each triangulated pt
      prev_gray : this frame’s undistorted grayscale (for next call)
      kp_prev   : this frame’s keypoints
      des_prev  : this frame’s descriptors
    """
    # undistort & gray
    gray = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
    gray = cv.undistort(gray, K, dist)

    # detect & compute
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or des_prev is None or len(des_prev) < 8:
        return raw, np.array([]), gray, kp, des

    # match & ratio‐test
    matches = bf.knnMatch(des_prev, des, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]
    if len(good) < 8:
        return raw, np.array([]), gray, kp, des

    pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good])
    pts      = np.float32([   kp[m.trainIdx].pt    for m in good])

    # draw matches
    for m in good:
        x,y = kp[m.trainIdx].pt
        cv.circle(raw, (int(x),int(y)), 3, (0,0,255), -1)

    # Essential matrix & pose
    E, mask = cv.findEssentialMat(pts_prev, pts, K,
                                  method=cv.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return raw, np.array([]), gray, kp, des

    _, R, t, _ = cv.recoverPose(E, pts_prev, pts, K)

    # scale
    scale = 1.0
    if odo_dist is not None:
        norm_t = np.linalg.norm(t)
        if norm_t>1e-6:
            scale = odo_dist / norm_t

    # triangulate
    P1 = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    P2 = K.dot(np.hstack((R, scale*t)))
    pts4d = cv.triangulatePoints(P1, P2, pts_prev.T, pts.T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    dists = np.linalg.norm(pts3d, axis=1)


    return raw, dists, gray, kp, des


def startup(arduino, final_speed):
    send_command(arduino, f"SPEED {final_speed+30}")
    fwd_action(arduino)
    sleep(0.2)
    fwd_action(arduino, f"SPEED {final_speed}")
    