#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import yaml
import time
from time import sleep
from collections import deque
from gpiozero import DistanceSensor
import serial

from utils.utils import process_frame, parse_sensor_prompt

# -----------------------------------------------------------------------------
# PARAMETERS & HARDWARE SETUP
# -----------------------------------------------------------------------------
# Ultrasonic thresholds (cm)
THRESHOLD_DISTANCE_LR = 40  

# Steering parameters
STEER_INCREMENT = 15   # degrees
SERVO_CENTER   = 90   # neutral
SERVO_MAX_L    = 110
SERVO_MAX_R    = 70
STEER_COOLDOWN = 1.5  # seconds between steer commands

# Serial port
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE   = 115200

# Floorplan & landmarks
FLOORPLAN_IMG = "mudd13_bw.png"     # B/W floorplan: 1=open, 0=wall
landmark_world = {                  # world coords in meters
    'A': (1.0, 0.5),
    'B': (3.0, 0.5),
    'C': (3.0, 2.0),
}
landmark_grid = {                   # corresponding grid cells
    'A': (10,  2),
    'B': (10, 10),
    'C': ( 2, 10),
}

# Chessboard pattern (for localization)
PATTERN_SIZE   = (9,6)
SQUARE_SIZE_M  = 0.03  # 3cm squares

# Camera calibration
with open('calib.yaml') as f:
    data = yaml.safe_load(f)
K    = np.array(data['K'])
dist = np.array(data['dist'])

# ORB for front‐distance (unused here, but required by process_frame)
orb = cv.ORB_create(2000)
bf  = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

# Video
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# Ultrasonic sensors
left_sense  = DistanceSensor(echo=22, trigger=27)
right_sense = DistanceSensor(echo=17, trigger=4)

# Arduino serial
arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(0.1)

# BFS Maze load
bw   = cv.imread(FLOORPLAN_IMG, cv.IMREAD_GRAYSCALE)
maze = (bw > 128).astype(np.uint8)  # 1=open

# Precompute chessboard object points
objp = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:PATTERN_SIZE[0],0:PATTERN_SIZE[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE_M

# -----------------------------------------------------------------------------
# UTILS: sending commands
# -----------------------------------------------------------------------------
def send_cmd(cmd):
    arduino.write((cmd + "\n").encode())
def forward():   send_cmd("FORWARD")
def stop():      send_cmd("STOP")
def setSpeed(s): send_cmd(f"SPEED {s}")
def setSteer(s): send_cmd(f"STEER {s}")
def get_lr():    return parse_sensor_prompt(send_cmd("SENSOR"))

# -----------------------------------------------------------------------------
# BFS PATHFINDING
# -----------------------------------------------------------------------------
def bfs(start, goal, grid):
    R,C = grid.shape
    vis = np.zeros_like(grid, bool)
    parent = {}
    q = deque([start])
    vis[start] = True

    while q:
        r,c = q.popleft()
        if (r,c)==goal:
            path,cur = [],(r,c)
            while cur!=start:
                path.append(cur)
                cur = parent[cur]
            path.append(start)
            return path[::-1]
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = r+dr, c+dc
            if 0<=nr<R and 0<=nc<C and not vis[nr,nc] and grid[nr,nc]:
                vis[nr,nc]=True
                parent[(nr,nc)]=(r,c)
                q.append((nr,nc))
    return []

# -----------------------------------------------------------------------------
# CHESSBOARD LOCALIZATION
# -----------------------------------------------------------------------------
def detect_board_cell(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, PATTERN_SIZE)
    if not ret:
        return None
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1),
                              (cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_MAX_ITER,30,1e-3))
    _, rvec, tvec = cv.solvePnP(objp, corners2, K, dist)
    R,_ = cv.Rodrigues(rvec)
    # camera pos in board coords
    cam_in_board = -R.T.dot(tvec).flatten()
    # find nearest landmark
    best,name,dmin = None,None,1e6
    for nm,(wx,wy) in landmark_world.items():
        d = np.hypot(wx - cam_in_board[0], wy - cam_in_board[1])
        if d<dmin:
            dmin, best = d, nm
    if dmin>1.0:
        return None
    return landmark_grid[best]

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    # init
    setSpeed(70)
    setSteer(SERVO_CENTER)
    forward()
    path = bfs(landmark_grid['A'], landmark_grid['C'], maze)
    cur = path[0]
    last_steer_t = time.time()

    # initial ORB state for process_frame
    ret, first = cap.read()
    prev_gray = cv.cvtColor(first, cv.COLOR_BGR2GRAY)
    prev_gray = cv.undistort(prev_gray, K, dist)
    kp_prev, des_prev = orb.detectAndCompute(prev_gray, None)

    while True:
        # 1) sensors
        ld, rd = get_lr()
        ret, frame = cap.read()
        frame_out, dists, prev_gray, kp_prev, des_prev = process_frame(
            frame, prev_gray, kp_prev, des_prev, K, dist, orb, bf, odo_dist=None)
        fwd_dist = dists.min() if dists.size else 999

        # 2) localization (if board seen)
        cell = detect_board_cell(frame)
        if cell and cell!=cur:
            cur = cell
            path = bfs(cur, landmark_grid['C'], maze)
            print("Replanned from", cur, "→", path)

        # 3) immediate avoidance
        now = time.time()
        if ld<THRESHOLD_DISTANCE_LR and rd<THRESHOLD_DISTANCE_LR:
            stop()
        elif now - last_steer_t > STEER_SLEEP_LEN and rd<THRESHOLD_DISTANCE_LR:
            # steer left
            STEER = min(SERVO_MAX_L, STEER_CENTER+STEER_INCREMENT)
            setSteer(STEER); last_steer_t=now
        elif now - last_steer_t > STEER_SLEEP_LEN and ld<THRESHOLD_DISTANCE_LR:
            # steer right
            STEER = max(SERVO_MAX_R, SERVO_CENTER-STEER_INCREMENT)
            setSteer(STEER); last_steer_t=now
        else:
            # 4) follow path if clear
            if len(path)>1 and fwd_dist>THRESHOLD_DISTANCE_LR:
                # simple timed forward to next cell
                forward(); sleep(0.5)
                cur = path[1]
                path = path[1:]
                setSteer(SERVO_CENTER)
                print("Moved to", cur)
            else:
                stop()

        # overlay info
        cv.putText(frame_out, f"LD:{ld:.1f} RD:{rd:.1f} FD:{fwd_dist:.2f}",
                   (20,30), cv.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv.imshow("Nav", frame_out)
        if cv.waitKey(1)&0xFF==27:
            break

    cap.release()
    cv.destroyAllWindows()
    arduino.close()

if __name__=="__main__":
    main()
