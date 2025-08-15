import cv2
import os
import time

def register_face(user_id, num_captures=5):
    """Capture and save multiple face images in correct format"""
    
    user_folder = f"faces/{user_id}"
    os.makedirs(user_folder, exist_ok=True)
    
    # Clear any existing files
    for file in os.listdir(user_folder):
        os.remove(os.path.join(user_folder, file))
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False
    
    print(f"\nFace Registration for ID: {user_id}")
    print(f"We'll take {num_captures} photos. Please face the camera directly.")
    
    count = 0
    while count < num_captures:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
            
        # Verify frame is in correct format
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"Error: Invalid frame format. Shape: {frame.shape}")
            break
            
        # Display countdown
        cv2.putText(frame, f"Photo {count+1}/{num_captures}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to capture", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Face Registration", frame)
        
        key = cv2.waitKey(1)
        
        if key == 32:  # SPACE key
            # Convert to RGB and back to BGR for saving
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            img_path = f"{user_folder}/{int(time.time())}.jpg"
            
            # Save as high-quality JPEG
            success = cv2.imwrite(img_path, bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if success:
                print(f"Saved: {img_path}")
                # Verify saved image
                test_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if test_img is None or len(test_img.shape) != 3 or test_img.shape[2] != 3:
                    print(f"Error: Saved image {img_path} is invalid. Shape: {test_img.shape if test_img is not None else 'None'}")
                else:
                    print(f"Verified: {img_path} is valid (Shape: {test_img.shape})")
                count += 1
            else:
                print(f"Error: Failed to save {img_path}")
            time.sleep(0.5)  # Brief pause
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if count == num_captures:
        print(f"\n✅ Success! {count} photos saved for {user_id}")
        return True
    else:
        print(f"\n⚠️ Only {count} photos saved. Registration incomplete.")
        return False

if __name__ == "__main__":
    user_id = input("Enter Pension ID: ").strip()
    register_face(user_id)