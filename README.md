# Volume Controller — OpenCV & Hand Tracking

Control your system volume using hand gestures via webcam.

## How it works
- Detects hand landmarks using MediaPipe
- Measures distance between thumb (4) and index finger (8)
- Maps that distance to system volume using numpy interp
- Controls Windows audio via pycaw

## Tech used
- Python, OpenCV, MediaPipe, NumPy, pycaw

## How to run
pip install opencv-python mediapipe numpy pycaw
Run VolumeControl.py — pinch fingers to lower volume, spread to raise
