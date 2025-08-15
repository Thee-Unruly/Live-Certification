import face_recognition
import os
import cv2
import numpy as np

def debug_image_properties(image, name):
    print(f"\n[DEBUG] {name} properties:")
    print(f"Type: {type(image)}")
    if hasattr(image, 'shape'): print(f"Shape: {image.shape}")
    if hasattr(image, 'dtype'): print(f"Data type: {image.dtype}")
    print(f"Min value: {np.min(image) if hasattr(image, 'shape') else 'N/A'}")
    print(f"Max value: {np.max(image) if hasattr(image, 'shape') else 'N/A'}")

# ---------------------------
# Load known faces
# ---------------------------
known_face_encodings = []
known_face_names = []
known_faces_dir = "known_faces"

if not os.path.exists(known_faces_dir):
    print(f"[CRITICAL ERROR] Directory {known_faces_dir} does not exist")
    exit()

for filename in [f for f in os.listdir(known_faces_dir) if f.lower().endswith(('.jpg', '.png'))]:
    path = os.path.join(known_faces_dir, filename)
    
    try:
        print(f"\n=== Processing {filename} ===")
        
        # Load with OpenCV
        cv_image = cv2.imread(path)
        if cv_image is None:
            raise ValueError("OpenCV couldn't read image")
        
        debug_image_properties(cv_image, "OpenCV Image")
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        rgb_image = np.ascontiguousarray(rgb_image)  # Ensure contiguous memory
        debug_image_properties(rgb_image, "RGB Image")
        
        # Ensure uint8
        if rgb_image.dtype != np.uint8:
            print("Converting to uint8")
            rgb_image = rgb_image.astype(np.uint8)
        
        # Verify image shape
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            raise ValueError("Image must be 3-channel RGB")
        
        debug_image_properties(rgb_image, "Final Image")
        
        # Face detection
        print("Attempting face detection...")
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        print(f"Found {len(face_locations)} face(s): {face_locations}")
        
        if not face_locations:
            raise ValueError("No faces detected")
        
        # Face encodings
        print("Attempting face encodings...")
        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if not encodings:
            raise ValueError("Face detected but encoding failed")
        
        known_face_encodings.append(encodings[0])
        known_face_names.append(os.path.splitext(filename)[0])
        print(f"[SUCCESS] Loaded {filename}")
        
    except Exception as e:
        print(f"[FAILURE] {filename}: {str(e)}")
        continue

if not known_face_encodings:
    print("\n[CRITICAL ERROR] No faces loaded. Possible solutions:")
    print("1. Verify images contain clear frontal faces")
    print("2. Try different image formats (PNG instead of JPG)")
    print("3. Check image resolution (should be at least 128x128)")
    print("4. Try with simpler images (plain background, good lighting)")
    exit()

# ---------------------------
# Webcam processing
# ---------------------------
print("\nStarting webcam...")
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("[CRITICAL ERROR] Could not open webcam")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to capture frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        
        if True in matches:
            name = known_face_names[matches.index(True)]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()