"""
Offline Live Certification:
- Passes if hand wave detected AND ANY of these are met:
    1. Real face detected (anti-spoof)
    2. Smile detected (FER .tflite)
    3. Blink detected (EAR)
- Logs result to verification_log.json
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import json
import os
from datetime import datetime

# ================
# CONFIGURATION
# ================
MODEL_PATH = r"C:\Users\ibrahim.fadhili\OneDrive - Agile Business Solutions\Live Certification\model.tflite"

SMILE_CONFIDENCE_THRESHOLD = 0.60
EAR_THRESHOLD = 0.20
EYE_CLOSED_FRAMES_REQUIRED = 2
BLINKS_REQUIRED = 1
ANTI_SPOOF_LAPLACIAN_THRESH = 50.0
ANTI_SPOOF_COLOR_STD_THRESH = 15.0
MICRO_MOVEMENT_THRESH = 0.0001  # New: For micro-movement check
BACKGROUND_MOTION_THRESH = 500  # New: For background motion check
WAVE_FRAMES_REQUIRED = 5  # New: For continuous hand wave detection

CAPTURE_TIMEOUT = 30
LOG_FILE = "verification_log.json"

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

# ================
# LOAD TFLITE MODEL
# ================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"TFLite model not found at: {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

_, IN_H, IN_W, IN_C = (int(x) for x in input_details[0]['shape'])

# ================
# MediaPipe setup
# ================
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                               max_num_faces=1,
                                               refine_landmarks=True,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands.Hands(static_image_mode=False,
                                    max_num_hands=1,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 380, 373]

# ================
# Helper functions
# ================
def eye_aspect_ratio(landmarks, eye_indices):
    def lm(i):
        return np.array([landmarks[i].x, landmarks[i].y])
    p1 = lm(eye_indices[0]); p2 = lm(eye_indices[1]); p3 = lm(eye_indices[2])
    p4 = lm(eye_indices[3]); p5 = lm(eye_indices[4]); p6 = lm(eye_indices[5])
    vertical_1 = np.linalg.norm(p2 - p4)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p6) + 1e-8
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def landmarks_to_bbox(landmarks, image_w, image_h, padding=0.15):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x
    h = max_y - min_y
    min_x -= w * padding; min_y -= h * padding
    max_x += w * padding; max_y += h * padding
    x1 = int(max(0, min_x * image_w))
    y1 = int(max(0, min_y * image_h))
    x2 = int(min(image_w - 1, max_x * image_w))
    y2 = int(min(image_h - 1, max_y * image_h))
    return x1, y1, x2 - x1, y2 - y1

def anti_spoof_checks(face_roi):
    if face_roi.size == 0:
        return False
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    color_std = np.std(face_roi)
    if lap_var < ANTI_SPOOF_LAPLACIAN_THRESH or color_std < ANTI_SPOOF_COLOR_STD_THRESH:
        return False
    if SKIMAGE_AVAILABLE:
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points+3), range=(0, n_points+2))
        hist = hist.astype("float32")
        if hist.sum() == 0:
            return False
        hist /= hist.sum()
        if np.count_nonzero(hist > 0.01) < 8:
            return False
    return True

def detect_micro_movements(landmarks, prev_landmarks, threshold=MICRO_MOVEMENT_THRESH):
    if prev_landmarks is None:
        return True
    nose_current = np.array([landmarks[1].x, landmarks[1].y])  # Nose tip
    nose_prev = np.array([prev_landmarks[1].x, prev_landmarks[1].y])
    movement = np.linalg.norm(nose_current - nose_prev)
    return movement > threshold

def check_background_motion(frames, threshold=BACKGROUND_MOTION_THRESH):
    if len(frames) < 2:
        return True
    diff = cv2.absdiff(frames[-1], frames[-2])
    non_zero_count = np.count_nonzero(cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY))
    return non_zero_count > threshold

def detect_hand_wave(hand_landmarks_list, prev_hand_landmarks, threshold=0.01):
    if not hand_landmarks_list or len(hand_landmarks_list) == 0:
        return False, None
    if prev_hand_landmarks is None:
        return True, hand_landmarks_list[0].landmark
    # Check hand motion (e.g., wrist landmark)
    wrist_current = np.array([hand_landmarks_list[0].landmark[0].x, hand_landmarks_list[0].landmark[0].y])
    wrist_prev = np.array([prev_hand_landmarks[0].x, prev_hand_landmarks[0].y])
    motion = np.linalg.norm(wrist_current - wrist_prev)
    return motion > threshold, hand_landmarks_list[0].landmark

def predict_emotion_tflite(face_roi):
    if face_roi.size == 0:
        return None, 0.0
    img = cv2.resize(face_roi, (IN_W, IN_H))
    if IN_C == 1:
        img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        img_proc = img_proc.reshape(1, IN_H, IN_W, 1)
    else:
        img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_proc = img_proc.reshape(1, IN_H, IN_W, IN_C)
    img_proc = img_proc / 255.0
    img_proc = img_proc.astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], img_proc)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(out))
    label = EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else str(idx)
    return label, float(out[idx])

def write_log(entry):
    arr = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                arr = json.load(f)
        except Exception:
            arr = []
    arr.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(arr, f, indent=2, ensure_ascii=False)

# ================
# MAIN: run webcam verification
# ================
def run_verification():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Verification passes if: hand wave AND (real face OR smile OR blink)")
    start_ts = time.time()
    smile_ok = False
    blink_count = 0
    eye_closed_frames = 0
    live_ok = False
    passed = False
    reason = None
    prev_landmarks = None
    background_frames = []
    hand_wave_counter = 0
    prev_hand_landmarks = None


    with mp_face_mesh as face_mesh, mp_hands as hands:
        while True:
            micro_movement_ok = False  # Ensure variable is always defined
            background_ok = True       # Ensure variable is always defined
            ret, frame = cap.read()
            if not ret:
                reason = "capture_failed"
                break
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            background_frames.append(rgb)
            face_results = face_mesh.process(rgb)
            hand_results = hands.process(rgb)

            # Anti-spoofing: Background motion
            if len(background_frames) > 10:
                background_ok = check_background_motion(background_frames)
                background_frames.pop(0)
            else:
                background_ok = True

            elapsed = time.time() - start_ts
            cv2.putText(frame, f"Time left: {int(max(0, CAPTURE_TIMEOUT - elapsed))}s", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                # Anti-spoofing: Micro-movements
                micro_movement_ok = detect_micro_movements(landmarks, prev_landmarks)
                prev_landmarks = landmarks

                x, y, bw, bh = landmarks_to_bbox(landmarks, w, h, padding=0.18)
                x2, y2 = x + bw, y + bh
                if bw > 0 and bh > 0 and x < w and y < h:
                    face_roi = frame[y:y2, x:x2].copy()
                    live_ok = anti_spoof_checks(face_roi) and micro_movement_ok and background_ok
                    if live_ok:
                        cv2.rectangle(frame, (x,y), (x2,y2), (0,255,0), 2)
                        label, conf = predict_emotion_tflite(face_roi)
                        if label:
                            cv2.putText(frame, f"{label} {conf:.2f}", (x, y-15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                            if label.lower() == "happy" and conf >= SMILE_CONFIDENCE_THRESHOLD:
                                smile_ok = True
                        try:
                            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
                            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
                        except Exception:
                            left_ear = right_ear = 1.0
                        avg_ear = (left_ear + right_ear) / 2.0
                        if avg_ear < EAR_THRESHOLD:
                            eye_closed_frames += 1
                        else:
                            if eye_closed_frames >= EYE_CLOSED_FRAMES_REQUIRED:
                                blink_count += 1
                            eye_closed_frames = 0
                    else:
                        cv2.putText(frame, "Spoof Suspected", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # Hand wave detection
            hand_wave_detected, curr_hand_landmarks = detect_hand_wave(hand_results.multi_hand_landmarks, prev_hand_landmarks)
            prev_hand_landmarks = curr_hand_landmarks
            if hand_wave_detected:
                hand_wave_counter += 1
                cv2.putText(frame, f"Wave frames: {hand_wave_counter}/{WAVE_FRAMES_REQUIRED}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,255,180), 2)
            else:
                hand_wave_counter = 0

            status_lines = [
                f"Real face: {'OK' if live_ok else 'NOT'}",
                f"Smile: {'OK' if smile_ok else 'NOT'}",
                f"Blink count: {blink_count}/{BLINKS_REQUIRED}",
                f"Wave: {'OK' if hand_wave_counter >= WAVE_FRAMES_REQUIRED else 'NOT'}",
                f"Micro Movement: {'OK' if micro_movement_ok else 'NOT'}",
                f"Background: {'OK' if background_ok else 'NOT'}"
            ]
            for i, s in enumerate(status_lines):
                cv2.putText(frame, s, (10, 80 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)

            cv2.imshow("Live Certification", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                reason = "user_cancel"
                break

            # PASS condition: Hand wave AND (real face OR smile OR blink)
            if hand_wave_counter >= WAVE_FRAMES_REQUIRED and (live_ok or smile_ok or blink_count >= BLINKS_REQUIRED):
                passed = True
                reason = "hand_wave_and_any_condition_verified"
                break

            if elapsed > CAPTURE_TIMEOUT:
                passed = False
                reason = "timeout"
                break

    cap.release()
    cv2.destroyAllWindows()

    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "passed": bool(passed),
        "reason": reason,
        "live_ok": bool(live_ok),
        "smile_ok": bool(smile_ok),
        "blink_count": int(blink_count),
        "hand_wave_ok": bool(hand_wave_counter >= WAVE_FRAMES_REQUIRED),
        "micro_movement_ok": bool(micro_movement_ok),
        "background_ok": bool(background_ok)
    }
    write_log(log_entry)
    if passed:
        print("[RESULT] Liveness Confirmed ✅")
    else:
        print("[RESULT] Liveness NOT confirmed ❌ -", reason)
    print(f"[INFO] Logged to {LOG_FILE}")

if __name__ == "__main__":
    run_verification()