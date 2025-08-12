import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import json
import os
from datetime import datetime

# Load TFLite model for emotion detection (64x64x3 input, 7 outputs)
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','neutral']

def detect_emotion(face_img):
    img = cv2.resize(face_img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = np.argmax(preds)
    return EMOTIONS[idx], preds[idx]

# Mediapipe for face & hands
mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Anti-spoofing: simple blur + color variance check
def is_live_face(frame, box):
    x, y, w, h = box
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    color_std = np.std(roi)
    return lap_var > 50 and color_std > 15

# Logging
LOG_FILE = "verification_log.json"
def write_log(entry):
    array = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                array = json.load(f)
        except:
            array = []
    array.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(array, f, indent=2)

# Run signature check
def run_liveness():
    cap = cv2.VideoCapture(0)
    wave_detected = False
    smile_detected = False
    start_time = time.time()
    challenge = np.random.choice(["blink twice", "smile", "wave"])
    print(f"Challenge: {challenge.upper()}, 20s to complete")

    blink_count = 0
    last_blink = time.time()

    with mp_face as face_detector, mp_hands as hands_detector:
        while time.time() - start_time < 20:
            ret, frame = cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hand_res = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.detections:
                det = results.detections[0]
                bbox = det.location_data.relative_bounding_box
                x, y = int(bbox.xmin*w), int(bbox.ymin*h)
                w_box, h_box = int(bbox.width*w), int(bbox.height*h)
                cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (255, 0, 0), 2)

                if is_live_face(frame, (x, y, w_box, h_box)):
                    face_roi = frame[y:y+h_box, x:x+w_box]
                    emotion, score = detect_emotion(face_roi)
                    cv2.putText(frame, f"{emotion} {score:.2f}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    if challenge == "smile" and emotion == "happy" and score > 0.5:
                        smile_detected = True

                    # Blink detection
                    if 'keypoints' in det.location_data:
                        # Not detailed here, as Mediapipe face_detection doesn't expose landmarks for eyes easily.
                        pass

                else:
                    cv2.putText(frame, "Spoof?", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if hand_res.multi_hand_landmarks:
                wave_detected = True
                cv2.putText(frame, "Wave detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Status
            status = []
            if challenge.startswith("smile") and smile_detected:
                status.append("Smile Passed")
            if challenge.startswith("wave") and wave_detected:
                status.append("Wave Passed")
            if status:
                cv2.putText(frame, " & ".join(status) + " ✅", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                write_log({"timestamp": datetime.utcnow().isoformat()+"Z", "challenge":challenge, "passed":True})
                break

            cv2.imshow("Live Certification", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_liveness()
