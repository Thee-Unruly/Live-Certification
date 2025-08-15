import cv2
import face_recognition
import os

def verify_face(user_id):
    # Load registered face
    registered_image_path = f"faces/{user_id}.jpg"
    if not os.path.exists(registered_image_path):
        print("No registered face found. Please register first.")
        return False
    
    known_image = face_recognition.load_image_file(registered_image_path)
    known_encoding = face_recognition.face_encodings(known_image)[0]
    
    # Capture current face
    cap = cv2.VideoCapture(0)
    print("Please look at the camera for verification...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB (face_recognition uses RGB)
        rgb_frame = frame[:, :, ::-1]
        
        # Find faces in current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding in face_encodings:
            # Compare with registered face
            matches = face_recognition.compare_faces([known_encoding], face_encoding)
            
            if True in matches:
                print("Verification successful! Face matched.")
                cap.release()
                cv2.destroyAllWindows()
                return True
            else:
                print("Face not recognized. Please try again.")
                
        cv2.imshow("Face Verification", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return False

# Example usage
user_id = input("Enter your pension ID: ")
if verify_face(user_id):
    print("Proceeding to live certification...")
else:
    print("Verification failed. Please try again.")