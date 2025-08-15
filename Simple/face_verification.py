import cv2
import face_recognition
import os

from Simple.live_cert import live_certification

def verify_face(user_id):
    user_folder = f"faces/{user_id}"
    if not os.path.exists(user_folder):
        print("No registered face found. Please register first.")
        return False
    
    # Load all registered images
    known_encodings = []
    for image_file in os.listdir(user_folder):
        image_path = os.path.join(user_folder, image_file)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
    
    if not known_encodings:
        print("No valid face images found in registration.")
        return False
    
    # Capture current face
    cap = cv2.VideoCapture(0)
    print("Please look at the camera for verification...")
    verification_success = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB
        rgb_frame = frame[:, :, ::-1]
        
        # Find faces in current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding in face_encodings:
            # Compare with all registered faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            
            if True in matches:
                print("Verification successful! Face matched.")
                verification_success = True
                break
            else:
                cv2.putText(frame, "NOT RECOGNIZED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        cv2.imshow("Face Verification", frame)
        
        if verification_success or cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return verification_success

# Example usage
user_id = input("Enter your pension ID: ")
if verify_face(user_id):
    print("Proceeding to live certification...")
    live_certification()  # From previous example
else:
    print("Verification failed. Please try again or re-register.")