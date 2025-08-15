import cv2
import face_recognition
import os
import numpy as np

def verify_face(user_id, max_attempts=3):
    """Verify face with proper image loading"""
    
    user_folder = f"faces/{user_id}"
    if not os.path.exists(user_folder):
        print(f"❌ No registration found for {user_id}")
        return False
    
    # Load registered faces with proper format checking
    known_encodings = []
    valid_images = 0
    
    for img_file in os.listdir(user_folder):
        img_path = os.path.join(user_folder, img_file)
        try:
            # Read with OpenCV
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"⚠️ Skipping {img_file}: Failed to load image")
                continue
                
            # Verify image properties
            if len(img.shape) != 3 or img.shape[2] != 3:
                print(f"⚠️ Skipping {img_file}: Invalid image format (Shape: {img.shape})")
                continue
                
            # Convert to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get face encodings
            encodings = face_recognition.face_encodings(rgb_img)
            if encodings:
                known_encodings.append(encodings[0])
                valid_images += 1
                print(f"✅ Loaded {img_file} (Shape: {img.shape})")
            else:
                print(f"⚠️ Skipping {img_file}: No faces detected")
        except Exception as e:
            print(f"⚠️ Skipping {img_file}: {str(e)}")
            continue
    
    if not known_encodings:
        print("❌ No usable face images found")
        return False
    
    print(f"🔍 Loaded {valid_images} valid face images for verification")
    
    # Start verification
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera error")
        return False
    
    attempt = 0
    while attempt < max_attempts:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
            
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Compare with all registered faces
            matches = face_recognition.compare_faces(known_encodings, face_encodings[0], tolerance=0.5)
            
            if True in matches:
                cv2.putText(frame, "✅ VERIFIED", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Result", frame)
                cv2.waitKey(2000)
                cap.release()
                cv2.destroyAllWindows()
                return True
            else:
                attempt += 1
                cv2.putText(frame, f"Attempt {attempt}/{max_attempts}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Face Verification", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("🔴 Verification failed")
    return False

if __name__ == "__main__":
    user_id = input("Enter Pension ID: ").strip()
    if verify_face(user_id):
        print("\n🎉 Verification successful!")
    else:
        print("\nPlease try again or contact support")