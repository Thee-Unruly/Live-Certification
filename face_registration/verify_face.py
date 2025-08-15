import cv2
import face_recognition
import numpy as np
import os

REGISTERED_FACES_DIR = "registered_faces"

def load_registered_faces():
    registered_data = {}
    for file in os.listdir(REGISTERED_FACES_DIR):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]
            encoding = np.load(os.path.join(REGISTERED_FACES_DIR, file))
            registered_data[name] = encoding
    return registered_data

def verify_face():
    registered_faces = load_registered_faces()
    if not registered_faces:
        print("Error: No registered faces found")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Face verification starting... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with registered faces
            matches = face_recognition.compare_faces(
                list(registered_faces.values()), 
                face_encoding,
                tolerance=0.6
            )
            
            name = "Unknown"
            if True in matches:
                matched_idx = matches.index(True)
                name = list(registered_faces.keys())[matched_idx]
            
            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        cv2.imshow('Face Verification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== Face Verification System ===")
    verify_face()