#Core Idea
# Liveness Detection App

# 1. The app randomly asks the user to either blink twice, smile, or wave.

# 2. A short 5–10 sec video is captured.

# 3. The AI model checks if the requested action was actually performed.

# 4. If it matches → Liveness Confirmed.

# Key Tech
# 1. Mediapipe → Face & hand landmark detection.

# 2. OpenCV → Video capture & blink detection.

# 3. Pre-trained expression classifier → Detect smile.

# 4. Wave detection logic → Track repeated left-right hand movement.

# Import necessary libraries
import cv2
import mediapipe as mp
import random
import time
import numpy as np

# Initialize MediaPipe Face Mesh and Drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize video capture