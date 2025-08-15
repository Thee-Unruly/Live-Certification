import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

# Configuration
REGISTERED_FACES_DIR = "registered_faces"
os.makedirs(REGISTERED_FACES_DIR, exist_ok=True)

def initialize_camera():
    """Initialize the camera with preferred backend"""
    # Try different backends
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
    return None

def validate_image(image):
    """Ensure image is in correct format for face_recognition"""
    if image is None:
        raise ValueError("Image is None")
    
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        elif image.shape[2] == 1:  # Single channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] != 3:  # Unexpected channels
            raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
    
    # Ensure values are in 0-255 range
    if np.min(image) < 0 or np.max(image) > 255:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image

def process_frame(frame):
    """Process frame for face detection and encoding"""
    try:
        # Convert and validate the frame
        rgb_frame = validate_image(frame[:, :, ::-1])  # BGR to RGB
        
        # Ensure contiguous memory
        rgb_frame = np.ascontiguousarray(rgb_frame)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        return face_locations, face_encodings
    except Exception as e:
        st.error(f"Frame processing error: {str(e)}")
        return [], []

def save_face_encoding(encoding, user_id):
    """Save face encoding to file"""
    np.save(os.path.join(REGISTERED_FACES_DIR, f"{user_id}.npy"), encoding)

def load_face_encodings():
    """Load all registered face encodings"""
    encodings = []
    ids = []
    for file in os.listdir(REGISTERED_FACES_DIR):
        if file.endswith(".npy"):
            encodings.append(np.load(os.path.join(REGISTERED_FACES_DIR, file)))
            ids.append(os.path.splitext(file)[0])
    return encodings, ids

def main():
    st.title("Face Authentication System")
    
    # Initialize session state
    if 'camera' not in st.session_state:
        st.session_state.camera = None
    if 'register_mode' not in st.session_state:
        st.session_state.register_mode = False
    if 'verify_mode' not in st.session_state:
        st.session_state.verify_mode = False
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose Mode", ["Home", "Register Face", "Verify Face"])
    
    if app_mode == "Home":
        st.write("""
        ## Welcome to Face Authentication System
        
        Please select an option from the sidebar:
        - **Register Face**: Capture and store your facial data
        - **Verify Face**: Authenticate using your face
        """)
        
    elif app_mode == "Register Face":
        st.header("Face Registration")
        user_id = st.text_input("Enter your ID (e.g., name or employee number):")
        
        if st.button("Start Registration"):
            if not user_id:
                st.warning("Please enter your ID first")
            else:
                st.session_state.register_mode = True
                st.session_state.camera = initialize_camera()
                if st.session_state.camera is None:
                    st.error("Could not initialize camera")
                    st.session_state.register_mode = False
                
        if st.session_state.register_mode:
            st.write("Position your face in the frame and click 'Capture'")
            st.write("The system will automatically detect your face")
            
            # Placeholder for camera feed
            camera_placeholder = st.empty()
            capture_button = st.button("Capture")
            
            try:
                while st.session_state.register_mode:
                    ret, frame = st.session_state.camera.read()
                    if not ret:
                        st.error("Failed to capture frame")
                        break
                    
                    # Display the frame
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(display_frame, channels="RGB")
                    
                    if capture_button:
                        face_locations, face_encodings = process_frame(frame)
                        if face_encodings:
                            save_face_encoding(face_encodings[0], user_id)
                            st.success(f"Face registered successfully for {user_id}!")
                            st.session_state.register_mode = False
                        else:
                            st.warning("No face detected. Please try again.")
            finally:
                if st.session_state.camera:
                    st.session_state.camera.release()
                st.session_state.register_mode = False
    
    elif app_mode == "Verify Face":
        st.header("Face Verification")
        
        if st.button("Start Verification"):
            st.session_state.verify_mode = True
            st.session_state.camera = initialize_camera()
            if st.session_state.camera is None:
                st.error("Could not initialize camera")
                st.session_state.verify_mode = False
            
        if st.session_state.verify_mode:
            st.write("Look at the camera to verify your identity")
            
            # Load registered faces
            registered_encodings, registered_ids = load_face_encodings()
            
            if not registered_encodings:
                st.warning("No faces registered yet. Please register first.")
                st.session_state.verify_mode = False
                return
            
            # Placeholder for camera feed and results
            camera_placeholder = st.empty()
            result_placeholder = st.empty()
            
            try:
                while st.session_state.verify_mode:
                    ret, frame = st.session_state.camera.read()
                    if not ret:
                        st.error("Failed to capture frame")
                        break
                    
                    # Process frame
                    face_locations, face_encodings = process_frame(frame)
                    
                    # Draw rectangles around faces
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Display the frame
                    camera_placeholder.image(display_frame, channels="RGB")
                    
                    # Verify faces
                    if face_encodings:
                        for face_encoding in face_encodings:
                            matches = face_recognition.compare_faces(registered_encodings, face_encoding)
                            if True in matches:
                                first_match_index = matches.index(True)
                                id = registered_ids[first_match_index]
                                result_placeholder.success(f"Verified as {id}")
                                time.sleep(2)
                                st.session_state.verify_mode = False
                                break
            finally:
                if st.session_state.camera:
                    st.session_state.camera.release()
                st.session_state.verify_mode = False

if __name__ == "__main__":
    main()