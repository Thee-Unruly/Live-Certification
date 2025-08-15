import cv2
import os

def register_face(user_id):
    # Create directory if it doesn't exist
    if not os.path.exists('faces'):
        os.makedirs('faces')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    count = 0
    
    print("Please look at the camera. Press 's' to save your face, 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Display the frame
        cv2.imshow("Face Registration - Smile and Press 's'", frame)
        
        key = cv2.waitKey(1)
        
        if key == ord('s'):
            # Save the face image
            face_path = f"faces/{user_id}.jpg"
            cv2.imwrite(face_path, frame)
            print(f"Face saved successfully at {face_path}")
            break
            
        if key == ord('q'):
            print("Registration cancelled")
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Example usage
user_id = input("Enter your pension ID: ")
register_face(user_id)