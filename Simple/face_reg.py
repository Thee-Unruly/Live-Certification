import cv2
import os
import time

def register_face(user_id):
    # Create user directory
    user_folder = f"faces/{user_id}"
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    else:
        print("User already registered. Adding more images...")
    
    cap = cv2.VideoCapture(0)
    count = 0
    captures_needed = 5
    
    print(f"Please look at the camera. We'll take {captures_needed} photos.")
    print("Move your head slightly between captures for better recognition.")
    
    while count < captures_needed:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Display instructions
        text = f"Capture {count+1}/{captures_needed} - Press 's'"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Registration", frame)
        
        key = cv2.waitKey(1)
        
        if key == ord('s'):
            # Save the face image
            timestamp = int(time.time())
            face_path = f"{user_folder}/{timestamp}.jpg"
            cv2.imwrite(face_path, frame)
            print(f"Saved image {count+1}/{captures_needed}")
            count += 1
            time.sleep(1)  # Pause between captures
            
        if key == ord('q'):
            print("Registration cancelled")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if count == captures_needed:
        print(f"Registration complete! {captures_needed} images saved.")
    else:
        print(f"Registration incomplete. Only {count} images saved.")

# Example usage
user_id = input("Enter your pension ID: ")
register_face(user_id)