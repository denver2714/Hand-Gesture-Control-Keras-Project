
# Hand Gesture Control System

---

Real-time hand gesture recognition system that lets you **draw**, **control volume**, and **click** using just your hand — powered by MediaPipe + Keras

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange)
![macOS](https://img.shields.io/badge/macOS-supported-brightgreen)

---

## Features

| Gesture       | Mode       | Action                                |
| ------------- | ---------- | ------------------------------------- |
| ✋ Open hand  | **Draw**   | Trace lines with your index finger    |
| ✌️ Peace sign | **Volume** | Pinch thumb+index to adjust system volume |
| ✊ Fist       | **Click**  | Point to move cursor, pinch to click  |

---

## Requirements

- macOS (volume control uses `osascript`)
- Python 3.9+
- Webcam

---

## Setup

```bash
# Clone the repo
git clone https://github.com/denver2714/hand-gesture-project.git
cd hand-gesture-project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Run with pre-trained model (recommended)

If `models/gesture_model.keras` and `models/labels.npy` exist, the system uses the Keras classifier automatically.

```bash
python3 gesture_control.py
```

### 2. Run without a trained model

Falls back to rule-based gesture detection. No training required.

```bash
python3 gesture_control.py
# Console will print: "No trained Keras model found — using rule-based gesture detection"
```

---

## Keyboard Controls

| Key | Action                        |
|-----|-------------------------------|
| `Q` | Quit                          |
| `C` | Clear drawing canvas          |
| `M` | Manually cycle mode           |
| `D` | Cycle drawing color           |

---

## Gesture Reference

| Gesture    | How to form it                              | Triggers               |
|------------|---------------------------------------------|------------------------|
| `open_hand`| All fingers extended                        | Switch to DRAW mode    |
| `peace`    | Index + middle up, others folded            | Switch to VOLUME mode  |
| `fist`     | All fingers curled                          | Switch to CLICK mode   |
| `point`    | Index up only                               | Draw stroke / move cursor |
| `pinch`    | Thumb tip + index tip close together        | Click / set volume     |

> Mode switches require the gesture to be held for **10 consecutive frames** with a **2-second cooldown** between switches.

---

## Mode Details

### DRAW Mode
- Point (`index up`) → draws a line on the canvas overlay
- Any other gesture → lifts the pen
- Press `C` to clear canvas
- Press `D` to change draw color (5 colors cycle)

### VOLUME Mode
- Distance between thumb tip and index tip controls system volume
- Fingers together → 0%, fingers apart → 100%
- Amber line and volume bar shown on screen

### CLICK Mode
- Index fingertip position maps to screen cursor
- Pinch (thumb + index tip closer than threshold) → left click
- Cursor smoothed over 7-frame window to reduce jitter

---

## Training Your Own Model

### Step 1 — Collect data

```bash
python3 collect_data.py
```

- Cycles through all 5 gestures
- Press `SPACE` to start/stop recording per gesture
- Collects 200 samples each → saves to `data/<gesture>.csv`

### Step 2 — Train the model

```bash
python3 train_model.py
```

- Loads all CSVs from `data/`
- Trains a Dense neural network (128 → 64 → 5) for 50 epochs
- Saves model to `models/gesture_model.keras`
- Saves label order to `models/labels.npy`

---

## Project Structure

```
hand-gesture-project/
├── gesture_control.py   # Main application — runtime loop
├── gesture_utils.py     # Shared utilities (landmark math, rule-based detection)
├── collect_data.py      # Training data collection tool
├── train_model.py       # Keras model trainer
├── requirements.txt     # Python dependencies
├── data/                # Collected landmark CSVs (one per gesture)
│   ├── open_hand.csv
│   ├── fist.csv
│   ├── point.csv
│   ├── pinch.csv
│   └── peace.csv
└── models/
    ├── hand_landmarker.task     # MediaPipe pre-trained hand model
    ├── gesture_model.keras      # Trained gesture classifier (generated)
    └── labels.npy               # Label order array (generated)
```

---

## How It Works

1. OpenCV captures webcam frames at 640×480
2. MediaPipe `HandLandmarker` detects 21 hand keypoints per frame
3. Landmarks are normalized relative to the wrist (position-invariant)
4. Keras model classifies the 63-feature vector into a gesture label (confidence threshold: 0.7); falls back to rule-based logic if model is absent
5. A stability buffer (10 consecutive frames) and 2-second cooldown gate mode switches to prevent flicker
6. The active mode drives its action: OpenCV canvas drawing, PyAutoGUI cursor/click, or osascript volume command
7. HUD overlay shows current mode, gesture, FPS, and (in VOLUME mode) a live volume bar

---

## Dependencies

| Package          | Version   | Purpose                          |
|------------------|-----------|----------------------------------|
| opencv-python    | ≥ 4.8.0   | Webcam capture, frame rendering  |
| mediapipe        | ≥ 0.10.0  | Hand landmark detection          |
| tensorflow       | ≥ 2.15.0  | Keras gesture classifier         |
| numpy            | ≥ 1.24.0  | Landmark math, feature arrays    |
| pyautogui        | ≥ 0.9.54  | Mouse cursor and click control   |
