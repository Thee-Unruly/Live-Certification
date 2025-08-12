import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import json
import random
import datetime
from io import BytesIO
from PIL import Image

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
mp_hands = mp.solutions.hands.Hands()

CHALLENGES = ["Blink twice", "Smile", "Wave"]

def detect_blink(landmarks):
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    def aspect_ratio(points):
        vertical = np.linalg.norm(points[1] - points[2]) + np.linalg.norm(points[4] - points[5])
        horizontal = np.linalg.norm(points[0] - points[3])
        return vertical / (2.0 * horizontal)
    left_points = np.array([(landmarks[i].x, landmarks[i].y) for i in LEFT_EYE])
    right_points = np.array([(landmarks[i].x, landmarks[i].y) for i in RIGHT_EYE])
    return aspect_ratio(left_points) < 0.2 and aspect_ratio(right_points) < 0.2

def detect_smile(landmarks):
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    return (landmarks[RIGHT_MOUTH].x - landmarks[LEFT_MOUTH].x) > 0.5

def detect_wave(hand_landmarks):
    return hand_landmarks is not None  # simple presence check

def anti_spoof(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > 100  # high variance = sharp image

def log_result(success, challenge):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "challenge": challenge,
        "success": success
    }
    with open("certification_log.json", "a") as f:
        f.write(json.dumps(entry) + "\n")

st.title("Liveness Certification Test")
if "challenge" not in st.session_state:
    st.session_state.challenge = None

if st.button("Start Test"):
    st.session_state.challenge = random.choice(CHALLENGES)
    st.write(f"**Challenge:** {st.session_state.challenge}")

if st.session_state.challenge:
    img_file = st.camera_input("Show your face and complete the challenge")
    if img_file:
        image = Image.open(img_file)
        img_array = np.array(image)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # Face & hand detection
        face_results = mp_face_mesh.process(img_rgb)
        hand_results = mp_hands.process(img_rgb)

        passed = False
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            if st.session_state.challenge == "Blink twice":
                passed = detect_blink(landmarks)
            elif st.session_state.challenge == "Smile":
                passed = detect_smile(landmarks)
        if st.session_state.challenge == "Wave":
            passed = detect_wave(hand_results.multi_hand_landmarks)

        # Anti-spoof check
        if passed and not anti_spoof(img_array):
            passed = False
            st.error("Spoof detected! Please use a real face.")

        if passed:
            st.success("✅ Challenge passed!")
            log_result(True, st.session_state.challenge)
            st.session_state.challenge = None
        else:
            st.warning("❌ Challenge not completed. Try again.")
