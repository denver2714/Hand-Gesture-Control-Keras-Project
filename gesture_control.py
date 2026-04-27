

import os
import sys
import time
import subprocess
import numpy as np
import cv2
import mediapipe as mp
import pyautogui

from gesture_utils import (
    detect_gesture_simple,
    normalize_landmarks,
    calculate_distance,
    map_to_screen,
    draw_hand_landmarks,
    GESTURE_LABELS,
)

# ─── Configuration ────────────────────────────────────────────────────────────
FRAME_W, FRAME_H = 640, 480
SMOOTHING = 7                # cursor smoothing window
CLICK_DIST_THRESH = 0.05     # pinch threshold (normalised landmark distance)
CLICK_COOLDOWN = 1.0          # seconds between clicks
GESTURE_STABLE_FRAMES = 10   # consecutive frames needed before mode switch
VOLUME_MIN_DIST = 30          # pixels — fingers together = 0 %
VOLUME_MAX_DIST = 250         # pixels — fingers apart   = 100 %
DRAW_THICKNESS = 3

# Disable PyAutoGUI fail-safe (corner abort) for smoother experience
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

# ─── Colours ──────────────────────────────────────────────────────────────────
CLR_DRAW   = (0, 255, 128)    # neon green
CLR_VOL    = (255, 180, 0)    # amber
CLR_CLICK  = (80, 180, 255)   # sky blue
CLR_HUD_BG = (30, 30, 30)
CLR_WHITE  = (255, 255, 255)
CLR_GRAY   = (150, 150, 150)

MODES      = ['DRAW', 'VOLUME', 'CLICK']
MODE_CLRS  = {'DRAW': CLR_DRAW, 'VOLUME': CLR_VOL, 'CLICK': CLR_CLICK}

# ─── MediaPipe Tasks Setup ────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'hand_landmarker.task')

BaseOptions       = mp.tasks.BaseOptions
HandLandmarker    = mp.tasks.vision.HandLandmarker
HandLandmarkerOpt = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ─── Try loading Keras gesture model ──────────────────────────────────────────
USE_MODEL = False
keras_model = None
label_list = None

KERAS_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'gesture_model.keras')
LABEL_PATH       = os.path.join(BASE_DIR, 'models', 'labels.npy')

if os.path.exists(KERAS_MODEL_PATH) and os.path.exists(LABEL_PATH):
    try:
        from tensorflow import keras
        keras_model = keras.models.load_model(KERAS_MODEL_PATH)
        label_list = np.load(LABEL_PATH, allow_pickle=True).tolist()
        USE_MODEL = True
        print(f" Loaded Keras model from {KERAS_MODEL_PATH}")
    except Exception as e:
        print(f"Could not load model: {e}")
        print("   Falling back to rule-based detection")
else:
    print("No trained Keras model found — using rule-based gesture detection")
    print("   (Run collect_data.py + train_model.py to train one)\n")


# ─── macOS Volume Control ─────────────────────────────────────────────────────

def set_volume(percent):
    """Set macOS system volume (0–100)."""
    vol = max(0, min(100, int(percent)))
    subprocess.run(
        ['osascript', '-e', f'set volume output volume {vol}'],
        capture_output=True,
    )
    return vol


def get_volume():
    """Get current macOS system volume."""
    try:
        result = subprocess.run(
            ['osascript', '-e', 'output volume of (get volume settings)'],
            capture_output=True, text=True,
        )
        return int(result.stdout.strip())
    except Exception:
        return 50


# ─── Gesture Recognition Wrapper ──────────────────────────────────────────────

def recognise_gesture(hand_landmarks_list):
    """
    Classify gesture using Keras model or rule-based fallback.
    hand_landmarks_list: list of 21 NormalizedLandmark objects.
    """
    if USE_MODEL:
        features = normalize_landmarks(hand_landmarks_list).reshape(1, -1)
        pred = keras_model.predict(features, verbose=0)
        idx = np.argmax(pred)
        confidence = pred[0][idx]
        if confidence > 0.7:
            return label_list[idx]
    return detect_gesture_simple(hand_landmarks_list)


# ─── Mode Switching Logic ─────────────────────────────────────────────────────

def gesture_to_mode(gesture, current_mode):
    """Map a recognised gesture to an operating mode."""
    mapping = {
        'open_hand': 'DRAW',
        'peace':     'VOLUME',
        'fist':      'CLICK',
    }
    new_mode = mapping.get(gesture)
    if new_mode and new_mode != current_mode:
        return new_mode
    return current_mode


# ─── HUD Drawing ──────────────────────────────────────────────────────────────

def draw_hud(frame, mode, gesture, fps, volume=None):
    """Overlay a translucent HUD bar at the top of the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_W, 52), CLR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    colour = MODE_CLRS.get(mode, CLR_WHITE)

    cv2.putText(frame, f"MODE: {mode}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    cv2.putText(frame, f"Gesture: {gesture}", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_GRAY, 1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (FRAME_W - 100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_GRAY, 1)

    if mode == 'VOLUME' and volume is not None:
        bar_x, bar_y = FRAME_W - 50, 60
        bar_h = 200
        filled = int(bar_h * volume / 100)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 25, bar_y + bar_h), CLR_GRAY, 2)
        cv2.rectangle(frame, (bar_x, bar_y + bar_h - filled),
                      (bar_x + 25, bar_y + bar_h), CLR_VOL, -1)
        cv2.putText(frame, f"{volume}%", (bar_x - 5, bar_y + bar_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_VOL, 2)


def draw_mode_instructions(frame, mode):
    """Show context-sensitive instructions at the bottom."""
    instructions = {
        'DRAW':   "Point finger to draw | Open hand to lift pen | C to clear",
        'VOLUME': "Pinch thumb+index to set volume | Spread fingers = louder",
        'CLICK':  "Point to move cursor | Pinch to click",
    }
    text = instructions.get(mode, "")
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, FRAME_H - 32), (FRAME_W, FRAME_H), CLR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (10, FRAME_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_WHITE, 1)


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("❌ Cannot open webcam")
        sys.exit(1)

    screen_w, screen_h = pyautogui.size()

    # State
    mode = 'DRAW'
    gesture = 'none'
    volume = get_volume()
    prev_time = time.time()
    fps = 0

    # Drawing state — canvas created after first frame to match actual size
    canvas = None
    actual_h, actual_w = FRAME_H, FRAME_W
    prev_draw_pt = None
    draw_color_idx = 0
    draw_colors = [
        (0, 255, 128),    # green
        (255, 100, 100),  # coral
        (100, 200, 255),  # sky
        (200, 100, 255),  # purple
        (0, 255, 255),    # yellow
    ]

    # Click state
    last_click_time = 0
    cursor_history = []
    pinch_was_released = True  # require release between clicks

    # Mode switch cooldown + stability buffer
    last_mode_switch = 0
    MODE_SWITCH_COOLDOWN = 2.0
    gesture_counter = {}       # tracks consecutive frames per gesture
    last_raw_gesture = 'none'
    gesture_start_time = None  # when current gesture candidate first appeared

    print("Gesture Control started!")
    print("   Press Q to quit | C to clear canvas | M to cycle mode | D to change colour")
    print(f"   Screen: {screen_w}×{screen_h}\n")

    # Create HandLandmarker
    options = HandLandmarkerOpt(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Match canvas to actual frame size (may differ from requested)
            actual_h, actual_w = frame.shape[:2]
            if canvas is None or canvas.shape[:2] != (actual_h, actual_w):
                canvas = np.zeros((actual_h, actual_w, 3), dtype=np.uint8)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand landmarks
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            # FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-9)
            prev_time = curr_time

            if result.hand_landmarks:
                hand_lm = result.hand_landmarks[0]  # list of 21 NormalizedLandmark

                # Draw hand skeleton
                draw_hand_landmarks(frame, hand_lm)

                # Recognise gesture
                gesture = recognise_gesture(hand_lm)

                # Stability buffer: only switch mode after N consecutive
                # frames of the same gesture
                if gesture == last_raw_gesture:
                    gesture_counter[gesture] = gesture_counter.get(gesture, 0) + 1
                else:
                    gesture_counter = {gesture: 1}
                    gesture_start_time = curr_time
                last_raw_gesture = gesture

                stable_count = gesture_counter.get(gesture, 0)

                # Mode switching with cooldown + stability
                if (curr_time - last_mode_switch > MODE_SWITCH_COOLDOWN
                        and stable_count >= GESTURE_STABLE_FRAMES):
                    new_mode = gesture_to_mode(gesture, mode)
                    if new_mode != mode:
                        elapsed = curr_time - gesture_start_time if gesture_start_time else 0.0
                        mode = new_mode
                        last_mode_switch = curr_time
                        prev_draw_pt = None
                        gesture_counter = {}
                        print(f"Mode → {mode}  |  gesture '{gesture}' held {elapsed:.2f}s to register")

                lm = hand_lm  # list of NormalizedLandmark

                # ── DRAW MODE ─────────────────────────────────────────────
                if mode == 'DRAW':
                    if gesture == 'point':
                        ix = int(lm[8].x * actual_w)
                        iy = int(lm[8].y * actual_h)
                        if prev_draw_pt is not None:
                            cv2.line(canvas, prev_draw_pt, (ix, iy),
                                     draw_colors[draw_color_idx], DRAW_THICKNESS)
                        prev_draw_pt = (ix, iy)

                        cv2.circle(frame, (ix, iy), 8,
                                   draw_colors[draw_color_idx], -1)
                    else:
                        prev_draw_pt = None

                # ── VOLUME MODE ───────────────────────────────────────────
                elif mode == 'VOLUME':
                    tx = int(lm[4].x * actual_w)
                    ty = int(lm[4].y * actual_h)
                    ix = int(lm[8].x * actual_w)
                    iy = int(lm[8].y * actual_h)

                    dist = np.sqrt((tx - ix) ** 2 + (ty - iy) ** 2)

                    cv2.line(frame, (tx, ty), (ix, iy), CLR_VOL, 3)
                    cv2.circle(frame, (tx, ty), 8, CLR_VOL, -1)
                    cv2.circle(frame, (ix, iy), 8, CLR_VOL, -1)

                    vol_pct = np.interp(dist,
                                        [VOLUME_MIN_DIST, VOLUME_MAX_DIST],
                                        [0, 100])
                    volume = set_volume(vol_pct)

                # ── CLICK MODE ────────────────────────────────────────────
                elif mode == 'CLICK':
                    ix = lm[8].x
                    iy = lm[8].y

                    sx, sy = map_to_screen(ix, iy, FRAME_W, FRAME_H,
                                           screen_w, screen_h)

                    cursor_history.append((sx, sy))
                    if len(cursor_history) > SMOOTHING:
                        cursor_history.pop(0)

                    avg_x = int(np.mean([p[0] for p in cursor_history]))
                    avg_y = int(np.mean([p[1] for p in cursor_history]))

                    pyautogui.moveTo(avg_x, avg_y)

                    pix_x = int(lm[8].x * actual_w)
                    pix_y = int(lm[8].y * actual_h)
                    cv2.circle(frame, (pix_x, pix_y), 12, CLR_CLICK, 2)

                    pinch_dist = calculate_distance(lm[4], lm[8])
                    if pinch_dist < CLICK_DIST_THRESH:
                        if (pinch_was_released
                                and curr_time - last_click_time > CLICK_COOLDOWN):
                            pyautogui.click()
                            last_click_time = curr_time
                            pinch_was_released = False
                            cv2.circle(frame, (pix_x, pix_y), 20, (0, 0, 255), -1)
                            print("   🖱  Click!")
                    else:
                        pinch_was_released = True

            else:
                gesture = 'none'
                prev_draw_pt = None

            # Merge canvas onto frame (draw mode)
            if mode == 'DRAW':
                gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
                frame = cv2.add(frame_bg, canvas_fg)

            # HUD
            draw_hud(frame, mode, gesture, fps, volume if mode == 'VOLUME' else None)
            draw_mode_instructions(frame, mode)

            cv2.imshow("Hand Gesture Control", frame)

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas = np.zeros((actual_h, actual_w, 3), dtype=np.uint8)
                print("Canvas cleared")
            elif key == ord('m'):
                idx = MODES.index(mode)
                mode = MODES[(idx + 1) % len(MODES)]
                last_mode_switch = curr_time
                prev_draw_pt = None
                print(f"   🔄 Mode → {mode}")
            elif key == ord('d'):
                draw_color_idx = (draw_color_idx + 1) % len(draw_colors)
                print(f"Draw colour changed")

    cap.release()
    cv2.destroyAllWindows()
    print("\nGoodbye!")


if __name__ == '__main__':
    main()
