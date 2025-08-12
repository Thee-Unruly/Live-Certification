"""
live_certification.py
Proof-of-concept liveness certification prototype:
- Random challenge: blink twice, smile, or wave
- Uses Mediapipe for face & hand landmarks
- Blink detection via Eye Aspect Ratio (EAR)
- Smile detection via mouth width/height ratio
- Wave detection via wrist x-position variance
- Anti-spoofing via Local Binary Pattern (LBP) texture heuristic
- Writes a JSON log entry on pass/fail with timestamp
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time
import json
import os
from datetime import datetime

# Optional dependency: scikit-image for LBP
try:
    from skimage.feature import local_binary_pattern
except Exception as e:
    local_binary_pattern = None
    # We'll handle missing dependency at runtime with a clear error.

# ---------------------------
# Config
# ---------------------------
CHALLENGES = ["blink", "smile", "wave"]
EAR_THRESH = 0.22          # eye aspect ratio threshold for closed eye
BLINK_INTERVAL = 0.2      # minimum seconds between registered blinks
WAVE_MOVEMENT_THRESH = 0.15
SMILE_RATIO_THRESH = 1.8  # mouth width / height ratio threshold
ANTI_SPOOF_FRAMES_REQUIRED = 4  # number of frames texture-check must pass
CAPTURE_TIMEOUT = 20    # seconds to wait for challenge success

LOG_FILE = "verification_log.json"

# ---------------------------
# Mediapipe setup
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------------------
# Utility functions
# ---------------------------
def eye_aspect_ratio(landmarks, eye_indices):
    """Calculate simplified EAR using face mesh landmark indices (normalized coords)."""
    # Convert landmarks list of mediapipe Landmarks into numpy arrays for selected indices
    def p(i): return np.array([landmarks[i].x, landmarks[i].y])
    # chosen indices follow Mediapipe FaceMesh mapping for eyes (used as example)
    p1 = p(eye_indices[0]); p2 = p(eye_indices[1]); p3 = p(eye_indices[2])
    p4 = p(eye_indices[3]); p5 = p(eye_indices[4]); p6 = p(eye_indices[5])
    # vertical distances
    vertical_1 = np.linalg.norm(p2 - p4)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p6) + 1e-6
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def mouth_ratio(landmarks):
    """Compute mouth width / height ratio using Mediapipe face mesh canonical indices."""
    # left corner 61, right corner 291, top 13, bottom 14 (approx in face mesh)
    left = np.array([landmarks[61].x, landmarks[61].y])
    right = np.array([landmarks[291].x, landmarks[291].y])
    top = np.array([landmarks[13].x, landmarks[13].y])
    bottom = np.array([landmarks[14].x, landmarks[14].y])
    width = np.linalg.norm(left - right)
    height = np.linalg.norm(top - bottom) + 1e-6
    return width / height

def landmarks_to_bbox(landmarks, image_w, image_h, padding=0.1):
    """Compute a bounding box (x,y,w,h) in pixel coords from face landmarks."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # add padding
    w = (max_x - min_x)
    h = (max_y - min_y)
    min_x -= w * padding
    min_y -= h * padding
    max_x += w * padding
    max_y += h * padding
    x1 = int(max(0, min_x * image_w))
    y1 = int(max(0, min_y * image_h))
    x2 = int(min(image_w - 1, max_x * image_w))
    y2 = int(min(image_h - 1, max_y * image_h))
    return x1, y1, x2 - x1, y2 - y1

def anti_spoofing_lbp(frame, face_box):
    """Simple LBP texture-based anti-spoofing heuristic.
       Returns True if likely live, False if likely spoof.
       Requires scikit-image's local_binary_pattern (skimage)."""
    if local_binary_pattern is None:
        raise RuntimeError("scikit-image not installed. Run: pip install scikit-image")
    x, y, w, h = face_box
    if w <= 10 or h <= 10:
        return False
    face_roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    # Parameters
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    # Histogram
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))
    hist = hist.astype("float32")
    if hist.sum() == 0:
        return False
    hist /= hist.sum()
    texture_variety = np.count_nonzero(hist > 0.01)
    # Heuristic threshold — tune on real data; 10 is a reasonable starting point
    return texture_variety > 10

def write_log(result_obj):
    existing = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(result_obj)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

# ---------------------------
# Main verification function
# ---------------------------
def run_liveness_challenge():
    challenge = random.choice(CHALLENGES)
    print(f"[INFO] Selected challenge: {challenge.upper()}")
    print("[INFO] You will have up to", CAPTURE_TIMEOUT, "seconds to perform the action.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    blink_count = 0
    last_blink_time = 0
    smile_detected = False
    wave_detected = False
    wave_positions = []
    anti_spoof_pass_count = 0
    start_time = time.time()
    passed = False
    reason = None

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh, \
         mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_h, image_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb)
            hand_results = hands.process(rgb)

            # Draw prompt text
            cv2.putText(frame, f"Challenge: {challenge}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            # If face detected -> do face-based checks
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0].landmark

                # EAR for both eyes (left & right approximate indices)
                try:
                    ear_left = eye_aspect_ratio(face_landmarks, [33, 160, 158, 133, 153, 144])
                    ear_right = eye_aspect_ratio(face_landmarks, [362, 385, 387, 263, 380, 373])
                except Exception:
                    ear_left = ear_right = 1.0

                # Blink detection
                if ear_left < EAR_THRESH and ear_right < EAR_THRESH:
                    now = time.time()
                    if now - last_blink_time > BLINK_INTERVAL:
                        blink_count += 1
                        last_blink_time = now
                        print(f"[DEBUG] Blink detected. Total blinks: {blink_count}")

                # Smile detection (mouth width / height ratio)
                try:
                    mr = mouth_ratio(face_landmarks)
                    if mr > SMILE_RATIO_THRESH:
                        smile_detected = True
                except Exception:
                    smile_detected = smile_detected or False

                # Compute face bbox for anti-spoofing
                x, y, w, h = landmarks_to_bbox(face_landmarks, image_w, image_h)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

                # Run anti-spoofing LBP: accumulate passes over multiple frames
                try:
                    if anti_spoofing_lbp(frame, (x,y,w,h)):
                        anti_spoof_pass_count += 1
                    else:
                        anti_spoof_pass_count = max(0, anti_spoof_pass_count - 1)
                except RuntimeError as re:
                    cap.release()
                    cv2.destroyAllWindows()
                    print("[ERROR]", str(re))
                    return

            # Hands -> wave detection
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # Using wrist landmark index 0 for horizontal movement
                    wrist_x = hand_landmarks.landmark[0].x
                    wave_positions.append(wrist_x)
                    if len(wave_positions) > 6:
                        movement = max(wave_positions) - min(wave_positions)
                        if movement > WAVE_MOVEMENT_THRESH:
                            wave_detected = True
                        wave_positions.pop(0)

            # Visual debug text
            cv2.putText(frame, f"BlinkCount: {blink_count}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Smile: {smile_detected}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Wave: {wave_detected}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"AntiSpoofCount: {anti_spoof_pass_count}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            cv2.imshow("Live Certification", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                reason = "user_cancel"
                break

            # Check pass conditions for selected challenge
            # First require anti-spoof pass (sustained for a number of frames)
            anti_spoof_ok = anti_spoof_pass_count >= ANTI_SPOOF_FRAMES_REQUIRED

            if challenge == "blink" and blink_count >= 2 and anti_spoof_ok:
                passed = True
                reason = "blink_verified_and_anti_spoof_ok"
                break
            elif challenge == "smile" and smile_detected and anti_spoof_ok:
                passed = True
                reason = "smile_verified_and_anti_spoof_ok"
                break
            elif challenge == "wave" and wave_detected and anti_spoof_ok:
                passed = True
                reason = "wave_verified_and_anti_spoof_ok"
                break

            # Check timeout
            if time.time() - start_time > CAPTURE_TIMEOUT:
                reason = "timeout"
                break

    cap.release()
    cv2.destroyAllWindows()

    # Log and print result
    timestamp = datetime.utcnow().isoformat() + "Z"
    result_obj = {
        "timestamp": timestamp,
        "challenge": challenge,
        "passed": bool(passed),
        "reason": reason,
    }
    write_log(result_obj)
    if passed:
        print("[RESULT] Liveness Confirmed ✅")
    else:
        print("[RESULT] Liveness NOT confirmed ❌ -", reason)
    print(f"[INFO] Logged result to {LOG_FILE}")

if __name__ == "__main__":
    run_liveness_challenge()
