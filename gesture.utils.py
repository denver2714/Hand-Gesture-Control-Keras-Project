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


