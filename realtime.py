import face_recognition
import os
import cv2
import numpy as np
from PIL import Image

def debug_image_properties(image, name):
    """Helper function to debug image properties"""
    print(f"\n[DEBUG] {name} properties:")
    print(f"Type: {type(image)}")
    if hasattr(image, 'shape'): print(f"Shape: {image.shape}")
    if hasattr(image, 'dtype'): print(f"Data type: {image.dtype}")
    if hasattr(image, 'mode'): print(f"Mode: {image.mode}")
    print(f"Min value: {np.min(image)}")
    print(f"Max value: {np.max(image)}")

# ---------------------------
# Load known faces with enhanced debugging
# ---------------------------
known_face_encodings = []
known_face_names = []
known_faces_dir = "known_faces"

for filename in [f for f in os.listdir(known_faces_dir) if f.lower().endswith('.jpg')]:
    path = os.path.join(known_faces_dir, filename)
    
    try:
        print(f"\n=== Processing {filename} ===")
        
        # Method 1: Load with PIL first
        pil_image = Image.open(path)
        debug_image_properties(pil_image, "PIL Image")
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            print(f"Converting from {pil_image.mode} to RGB")
            pil_image = pil_image.convert('RGB')
        
        pil_array = np.array(pil_image)
        debug_image_properties(pil_array, "PIL Array")
        
        # Method 2: Load with OpenCV
        cv_image = cv2.imread(path)
        if cv_image is None:
            raise ValueError("OpenCV couldn't read image")
        
        cv_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        debug_image_properties(cv_rgb, "OpenCV RGB")
        
        # Choose which image to use (prioritize PIL)
        working_image = pil_array
        
        # Double-check array properties
        if working_image.dtype != np.uint8:
            print("Converting to uint8")
            working_image = working_image.astype(np.uint8)
        
        if len(working_image.shape) != 3 or working_image.shape[2] != 3:
            raise ValueError("Final image must be 3-channel RGB")
        
        debug_image_properties(working_image, "Final Image")
        
        # Try face encodings
        print("Attempting face detection...")
        face_locations = face_recognition.face_locations(working_image)
        print(f"Found {len(face_locations)} face(s)")
        
        if not face_locations:
            raise ValueError("No faces detected")
        
        print("Attempting face encodings...")
        encodings = face_recognition.face_encodings(working_image, face_locations)
        
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

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
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

# How this works

# 1. Encoding known faces: face_recognition.face_encodings() creates a 128-d vector for each face in your dataset.

# 2. Real-time matching: For each frame, the script detects faces, encodes them, and compares them to your stored encodings using compare_faces.

# 3. Tolerance: Lowering the tolerance (< 0.6) makes matching stricter, raising it makes it more lenient.