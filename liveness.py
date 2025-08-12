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

# Challenge list
CHALLENGES = ["blink", "smile", "wave"]
selected_challenge = random.choice(CHALLENGES)
print(f"Your challenge: Please {selected_challenge}!")

# Eye blink detection function
def eye_aspect_ratio(landmarks, eye_indices):
    # EAR calculation based on face mesh points
    p1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p5 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p6 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    
    vertical_1 = np.linalg.norm(p2 - p4)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p6)
    
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# Initialize video capture
cap = cv2.VideoCapture(0)

blink_count = 0
blink_start = time.time()
smile_detected = False
wave_detected = False
wave_positions = []

with mp_face_mesh.FaceMesh(max_num_faces=1) as face_mesh, mp_hands.Hands(max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_face = face_mesh.process(rgb)
        result_hands = hands.process(rgb)

        # Face landmarks processing
        if result_face.multi_face_landmarks:
            landmarks = result_face.multi_face_landmarks[0].landmark

        # Blink detection
            EAR_LEFT = eye_aspect_ratio(landmarks, [33, 160, 158, 133, 153, 144])
            EAR_RIGHT = eye_aspect_ratio(landmarks, [263, 387, 385, 362, 380, 373])
            EAR_THRESH = 0.22

            if EAR_LEFT < EAR_THRESH and EAR_RIGHT < EAR_THRESH:
                if time.time() - blink_start > 0.2:
                    blink_count += 1
                    blink_start = time.time()

        # Smile detection (simple version: mouth width > height ratio)
            mouth_width = np.linalg.norm(
                np.array([landmarks[61].x, landmarks[61].y]) - 
                np.array([landmarks[291].x, landmarks[291].y])
            )
            mouth_height = np.linalg.norm(
                np.array([landmarks[13].x, landmarks[13].y]) - 
                np.array([landmarks[14].x, landmarks[14].y])
            )
            if mouth_width / mouth_height > 1.8:
                smile_detected = True