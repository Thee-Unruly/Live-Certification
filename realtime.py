import face_recognition
import os
import cv2
import numpy as np

# ---------------------------
# Load known faces (JPG-specific solution)
# ---------------------------
known_face_encodings = []
known_face_names = []

known_faces_dir = "known_faces"  # folder containing known face images

for filename in os.listdir(known_faces_dir):
    if not filename.lower().endswith('.jpg'):
        continue  # skip non-JPG files
        
    path = os.path.join(known_faces_dir, filename)
    
    try:
        # Method 1: Load with OpenCV (ensure it's reading properly)
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"OpenCV couldn't read {filename}")
            
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Method 2: Verify with face_recognition's loader
        try:
            alt_image = face_recognition.load_image_file(path)
            if np.array_equal(rgb_image, alt_image):
                print(f"[DEBUG] Both methods agree on {filename}")
            else:
                print(f"[DEBUG] Methods differ for {filename}, using OpenCV version")
        except Exception as e:
            print(f"[DEBUG] face_recognition loader failed: {str(e)}")
        
        # Ensure proper format
        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
            
        # Verify image structure
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            raise ValueError("Image must be 3-channel RGB")
            
        # Get face encodings
        encodings = face_recognition.face_encodings(rgb_image)
        if not encodings:
            raise ValueError("No faces found in image")
            
        known_face_encodings.append(encodings[0])
        name = os.path.splitext(filename)[0]
        known_face_names.append(name)
        print(f"[SUCCESS] Loaded {name}")
        
    except Exception as e:
        print(f"[FAILED] {filename}: {str(e)}")
        continue

if not known_face_encodings:
    print("[CRITICAL] No valid faces loaded - check your images!")
    exit()

print(f"\n[SUMMARY] Successfully loaded {len(known_face_encodings)} faces")

# ---------------------------
# Webcam Processing
# ---------------------------
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Webcam frame capture failed")
        break

    try:
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.6)
            name = "Unknown"
            
            if True in matches:
                name = known_face_names[matches.index(True)]
            
            # Draw UI elements
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        
    except Exception as e:
        print(f"[FRAME ERROR] {str(e)}")
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# How this works

# 1. Encoding known faces: face_recognition.face_encodings() creates a 128-d vector for each face in your dataset.

# 2. Real-time matching: For each frame, the script detects faces, encodes them, and compares them to your stored encodings using compare_faces.

# 3. Tolerance: Lowering the tolerance (< 0.6) makes matching stricter, raising it makes it more lenient.