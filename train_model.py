"""
train_model.py — Train a Keras gesture classifier on collected landmark data.

Usage:
    python3 train_model.py

Reads CSV files from data/, trains a Dense neural network, and saves:
    models/gesture_model.keras   — the trained model
    models/labels.npy            — ordered label list
"""

import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from gesture_utils import GESTURE_LABELS

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'gesture_model.keras')
LABEL_PATH = os.path.join(MODEL_DIR, 'labels.npy')


def load_data():
    """Load CSV data and return X (features) and y (label indices)."""
    X, y = [], []
    for idx, gesture in enumerate(GESTURE_LABELS):
        path = os.path.join(DATA_DIR, f"{gesture}.csv")
        if not os.path.exists(path):
            print(f"Missing data for '{gesture}', skipping")
            continue
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                X.append([float(v) for v in row])
                y.append(idx)
        print(f"Loaded {gesture}: {sum(1 for label in y if label == idx)} samples")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def build_model(input_dim, num_classes):
    """Build a small Dense classifier."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main():
    print("Loading data …")
    X, y = load_data()

    if len(X) == 0:
        print("No data found. Run collect_data.py first.")
        return

    num_classes = len(GESTURE_LABELS)
    print(f"\nTotal samples: {len(X)}  |  Classes: {num_classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n🏗  Building model …")
    model = build_model(X.shape[1], num_classes)
    model.summary()

    print("\nTraining …")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=1,
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n Test accuracy: {acc * 100:.1f}%")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    np.save(LABEL_PATH, np.array(GESTURE_LABELS))
    print(f"Model saved → {MODEL_PATH}")
    print(f"Labels saved → {LABEL_PATH}")


if __name__ == '__main__':
    main()
