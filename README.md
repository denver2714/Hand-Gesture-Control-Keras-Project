## 1.0 Executive Summary
Real-time hand gesture recognition system that lets you **draw**, **control volume**, and **click** using just your hand — powered by MediaPipe + Keras

The system is built on
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange)
![macOS](https://img.shields.io/badge/macOS-supported-brightgreen)

---

## 2.0 Core System Functionalities
The application maps specific, predefined hand gestures to distinct operational modes. The current iteration supports the following primary functions:

---

## 2.1 Virtual Annotation (Draw Mode)
•	Input Gesture: Open Hand (🖐️)
•	System Action: The system continuously tracks the spatial coordinates of the index finger, rendering digital strokes on the screen. This allows for freehand drawing or highlighting directly over the active display.

---

## 2.2 Dynamic Audio Control (Volume Mode)
•	Input Gesture: Peace Sign (✌️)
•	System Action: The application calculates the Euclidean distance between the tips of the thumb and the index finger. Adjusting this distance dynamically scales the master system volume up or down in real-time.

---

## 2.3 Cursor Emulation (Click & Navigate Mode)
•	Input Gesture: Fist / Pointing (✊)
•	System Action: Translates the overall hand trajectory into precise mouse cursor movements across the screen. A designated "pinch" gesture is recognized as a physical mouse click event, enabling full OS navigation.
|

---




