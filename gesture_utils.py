import math
import numpy as np


#Pang calculate ng distance between the 2 landmarks naten
def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

def calculate_distance_xy(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def normalize_landmarks(hand_landmarks_list):
    wrist = hand_landmarks_list[0]
    data = []
    for lm in hand_landmarks_list:
        data.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return np.array(data, dtype=np.float32)

#0-1) to screen pixel coordinates
def map_to_screen(x, y, frame_w, frame_h, screen_w, screen_h,  margin=80):
    x = np.clip(x, margin / frame_w, 1 - margin / frame_w)
    y = np.clip(y, margin / frame_h, 1 - margin / frame_h)
    screen_x = int(np.interp(x, (margin / frame_w, 1 - margin / frame_w), (0, screen_w)))
    screen_y = int(np.interp(y, (margin / frame_h, 1 - margin / frame_h), (0, screen_h)))
    return screen_x, screen_y




#From Kane
def draw_hand_landmarks(frame, hand_landmarks_list):
    import cv2
    h, w, _ = frame.shape

    # MediaPipe hand connections
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),        # thumb
        (0,5),(5,6),(6,7),(7,8),        # index
        (0,9),(9,10),(10,11),(11,12),   # middle  (extra from 5→9)
        (0,13),(13,14),(14,15),(15,16), # ring    (extra from 9→13)
        (0,17),(17,18),(18,19),(19,20), # pinky   (extra from 13→17)
        (5,9),(9,13),(13,17),           # palm
    ]

    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        p1 = hand_landmarks_list[start_idx]
        p2 = hand_landmarks_list[end_idx]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # Draw landmarks
    for lm in hand_landmarks_list:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)