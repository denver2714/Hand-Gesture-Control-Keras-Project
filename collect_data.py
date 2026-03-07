print("Collecting Data")

import os
import csv
import cv2
import mediapipe as mp
import numpy as np
from gesture_utils import normalize_landmarks, draw_hand_landmarks, GESTURE_LABELS


#===================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_ASSET = os.path.join(BASE_DIR, 'models', 'hand_landmarker.task')
SAMPLES_PER_GESTURE = 200



#================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOpt = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    options = HandLandmarkerOpt(
        base_options=BaseOptions(model_asset_path=MODEL_ASSET),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        for gesture in GESTURE_LABELS:
            print(f"\nGesture: {gesture}")
            print("Press SPACE to start recording, Q to skip\n")

            samples = []
            recording = False

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    hand_lm = result.hand_landmarks[0] 
                    draw_hand_landmarks(frame, hand_lm)

                    if recording:
                        data = normalize_landmarks(hand_lm)
                        samples.append(data)

                # HUD
                status = "RECORDING" if recording else "PAUSED"
                colour = (0, 0, 255) if recording else (200, 200, 200)
                cv2.putText(frame,
                            f"Gesture: {gesture}  |  {status}  |  {len(samples)}/{SAMPLES_PER_GESTURE}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

                cv2.imshow("Collect Data", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    recording = not recording
                elif key == ord('q'):
                    break

                if len(samples) >= SAMPLES_PER_GESTURE:
                    print(f"Collected {len(samples)} samples for '{gesture}'")
                    break
            

            if samples:
                path = os.path.join(DATA_DIR, f"{gesture}.csv")
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    for s in samples:
                        writer.writerow(s.tolist())
                print(f"Saved -> {path}")




if __name__ == '__main__':
    main()