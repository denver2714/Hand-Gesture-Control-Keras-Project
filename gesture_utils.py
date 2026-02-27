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

