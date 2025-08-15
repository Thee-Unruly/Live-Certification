import cv2
import face_recognition
import os
import time
import numpy as np

def verify_face(user_id, max_attempts=3):
    """Verify face against registered images"""
    
    user_folder = f"faces/{user_id}"
    if not os.path.exists(user_folder):
        print(f"❌ No registered faces found for {user_id}.")
        return False
    
    # Load registered face encodings
    known_encodings = []
    for img_file in os.listdir(user_folder):
        img_path = os.path.join(user_folder, img_file)
        try:
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_encodings.append(encodings[0])
        except Exception as e:
            print(f"Skipping {img_file}: {str(e)}")
    
    if not known_encodings:
        print("❌ No valid face data found.")
        return False
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera error.")
        return False
    
    print(f"\n🔍 Verifying {user_id}...")
    print("Look at the camera and wait for detection.")
    
    attempt = 0
    while attempt < max_attempts:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            matches = face_recognition.compare_faces(known_encodings, face_encodings[0], tolerance=0.5)
            
            if True in matches:
                cv2.putText(frame, "✅ VERIFIED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Verification", frame)
                cv2.waitKey(2000)
                cap.release()
                cv2.destroyAllWindows()
                return True
            else:
                cv2.putText(frame, "❌ NOT RECOGNIZED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                attempt += 1
        
        cv2.imshow("Face Verification", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return False

if __name__ == "__main__":
    user_id = input("Enter your Pension ID: ").strip()
    if verify_face(user_id):
        print("\n🎉 Verification successful! Proceeding to live certification...")
        # Add live certification steps here
    else:
        print("\n🔴 Verification failed. Please try again or re-register.")