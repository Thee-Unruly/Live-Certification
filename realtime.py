import face_recognition
import os
import cv2
import numpy as np

# ---------------------------
# Load known faces
# ---------------------------
known_face_encodings = []
known_face_names = []

known_faces_dir = "known_faces"  # folder containing known face images

for filename in os.listdir(known_faces_dir):
    path = os.path.join(known_faces_dir, filename)
    if not os.path.isfile(path):
        continue  # skip if not a file

    try:
        # Load image using face_recognition's load_image_file which handles formats correctly
        image = face_recognition.load_image_file(path)
        
        # Convert to RGB if it's grayscale (single channel)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # If image has alpha channel, remove it
            image = image[:, :, :3]
            
        # Ensure image is uint8
        if image.dtype != 'uint8':
            image = image.astype('uint8')

        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
            print(f"[INFO] Loaded {name}")
        else:
            print(f"[WARN] No face found in {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to process {filename}: {str(e)}")

print(f"[INFO] Loaded {len(known_face_encodings)} known faces.")

# ---------------------------
# Start webcam
# ---------------------------
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert webcam frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces and encode them
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw bounding box and label
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    # Show the video
    cv2.imshow("Real-Time Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

# How this works

# 1. Encoding known faces: face_recognition.face_encodings() creates a 128-d vector for each face in your dataset.

# 2. Real-time matching: For each frame, the script detects faces, encodes them, and compares them to your stored encodings using compare_faces.

# 3. Tolerance: Lowering the tolerance (< 0.6) makes matching stricter, raising it makes it more lenient.