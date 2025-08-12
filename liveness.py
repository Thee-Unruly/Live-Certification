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

        # Draw face landmarks
        if result_face.multi_face_landmarks:
            for face_landmarks in result_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks)