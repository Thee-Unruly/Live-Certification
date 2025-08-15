import face_recognition
import os
import cv2
import numpy as np
from PIL import Image
import sys

def comprehensive_image_test(image_path):
    """Comprehensive test of image loading and processing"""
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Test 1: Basic file info
    print("\n[TEST 1] File Information:")
    try:
        file_size = os.path.getsize(image_path)
        print(f"✓ File exists, size: {file_size} bytes")
    except Exception as e:
        print(f"✗ File error: {e}")
        return None
    
    # Test 2: face_recognition.load_image_file()
    print("\n[TEST 2] face_recognition.load_image_file():")
    try:
        img = face_recognition.load_image_file(image_path)
        print(f"✓ Loaded successfully")
        print(f"  Shape: {img.shape}")
        print(f"  Dtype: {img.dtype}")
        print(f"  Memory layout: C-contiguous={img.flags.c_contiguous}, F-contiguous={img.flags.f_contiguous}")
        print(f"  Min/Max: {img.min()}/{img.max()}")
        
        # Test the array directly with face_recognition
        print("\n[TEST 2a] Testing face_locations with loaded image:")
        try:
            # Create a copy to ensure memory layout
            img_copy = np.array(img, copy=True)
            print(f"  Copy properties - Shape: {img_copy.shape}, Dtype: {img_copy.dtype}")
            print(f"  Copy memory: C-contiguous={img_copy.flags.c_contiguous}")
            
            locations = face_recognition.face_locations(img_copy)
            print(f"✓ face_locations worked! Found {len(locations)} faces")
            return img_copy
            
        except Exception as e:
            print(f"✗ face_locations failed: {e}")
            
    except Exception as e:
        print(f"✗ Loading failed: {e}")
    
    # Test 3: PIL approach with different conversions
    print("\n[TEST 3] PIL with various conversions:")
    try:
        pil_img = Image.open(image_path)
        print(f"✓ PIL loaded - Mode: {pil_img.mode}, Size: {pil_img.size}")
        
        # Convert to RGB
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Different numpy conversion methods
        methods = [
            ("Direct np.array", lambda: np.array(pil_img)),
            ("np.asarray", lambda: np.asarray(pil_img)),
            ("np.array with copy", lambda: np.array(pil_img, copy=True)),
            ("np.array with dtype", lambda: np.array(pil_img, dtype=np.uint8))
        ]
        
        for method_name, method_func in methods:
            try:
                img_array = method_func()
                print(f"\n  {method_name}:")
                print(f"    Shape: {img_array.shape}, Dtype: {img_array.dtype}")
                print(f"    C-contiguous: {img_array.flags.c_contiguous}")
                
                # Test with face_recognition
                try:
                    locations = face_recognition.face_locations(img_array)
                    print(f"    ✓ face_locations worked! Found {len(locations)} faces")
                    return img_array
                except Exception as e:
                    print(f"    ✗ face_locations failed: {e}")
                    
            except Exception as e:
                print(f"    ✗ Conversion failed: {e}")
                
    except Exception as e:
        print(f"✗ PIL failed: {e}")
    
    # Test 4: OpenCV approach
    print("\n[TEST 4] OpenCV approach:")
    try:
        cv_img = cv2.imread(image_path)
        if cv_img is not None:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            print(f"✓ OpenCV loaded and converted to RGB")
            print(f"  Shape: {rgb_img.shape}, Dtype: {rgb_img.dtype}")
            print(f"  C-contiguous: {rgb_img.flags.c_contiguous}")
            
            try:
                locations = face_recognition.face_locations(rgb_img)
                print(f"✓ face_locations worked! Found {len(locations)} faces")
                return rgb_img
            except Exception as e:
                print(f"✗ face_locations failed: {e}")
        else:
            print("✗ OpenCV couldn't load image")
    except Exception as e:
        print(f"✗ OpenCV failed: {e}")
    
    # Test 5: Memory layout fixes
    print("\n[TEST 5] Memory layout fixes:")
    try:
        # Try the face_recognition loader again
        img = face_recognition.load_image_file(image_path)
        
        # Various memory layout fixes
        fixes = [
            ("np.ascontiguousarray", lambda x: np.ascontiguousarray(x)),
            ("Copy with order='C'", lambda x: np.array(x, order='C')),
            ("Reshape and back", lambda x: x.reshape(-1).reshape(x.shape)),
            ("Manual copy", lambda x: x.copy()),
        ]
        
        for fix_name, fix_func in fixes:
            try:
                fixed_img = fix_func(img)
                print(f"\n  {fix_name}:")
                print(f"    C-contiguous: {fixed_img.flags.c_contiguous}")
                
                locations = face_recognition.face_locations(fixed_img)
                print(f"    ✓ face_locations worked! Found {len(locations)} faces")
                return fixed_img
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                
    except Exception as e:
        print(f"✗ Memory fix test failed: {e}")
    
    print("\n✗ All tests failed!")
    return None

def create_test_image():
    """Create a simple test image to verify the setup"""
    print("\n[CREATING TEST IMAGE]")
    try:
        # Create a simple RGB image
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        test_img[50:150, 50:150] = [128, 128, 128]  # Gray square
        
        # Save it
        test_path = "test_image.png"
        Image.fromarray(test_img).save(test_path)
        print(f"✓ Created test image: {test_path}")
        
        # Test it
        try:
            locations = face_recognition.face_locations(test_img)
            print(f"✓ face_recognition.face_locations works with synthetic image")
            print(f"  Found {len(locations)} faces (expected: 0)")
            return True
        except Exception as e:
            print(f"✗ face_recognition.face_locations failed even with synthetic image: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Test image creation failed: {e}")
        return False

def main():
    print("FACE RECOGNITION DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # System info
    print(f"Python version: {sys.version}")
    try:
        import dlib
        print(f"dlib version: {dlib.DLIB_VERSION}")
    except:
        print("dlib version: Could not determine")
    
    # Test with synthetic image first
    if not create_test_image():
        print("\n[CRITICAL] Basic face_recognition functionality is broken!")
        print("This suggests an installation or environment issue.")
        print("\nTroubleshooting steps:")
        print("1. Reinstall face_recognition: pip uninstall face_recognition && pip install face_recognition")
        print("2. Check dlib installation: pip install dlib")
        print("3. Try in a fresh virtual environment")
        return
    
    # Check for known faces directory
    known_faces_dir = "known_faces"
    if not os.path.exists(known_faces_dir):
        print(f"\n[ERROR] Directory '{known_faces_dir}' not found!")
        return
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(known_faces_dir) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not image_files:
        print(f"\n[ERROR] No image files found in '{known_faces_dir}'")
        return
    
    # Test each image
    successful_images = []
    for filename in image_files:
        path = os.path.join(known_faces_dir, filename)
        result = comprehensive_image_test(path)
        if result is not None:
            successful_images.append((filename, result))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total images tested: {len(image_files)}")
    print(f"Successful images: {len(successful_images)}")
    
    if successful_images:
        print("\n✓ WORKING IMAGES:")
        for filename, img_array in successful_images:
            print(f"  - {filename}")
            # Try to get face encodings
            try:
                locations = face_recognition.face_locations(img_array)
                encodings = face_recognition.face_encodings(img_array, locations)
                print(f"    Faces: {len(locations)}, Encodings: {len(encodings)}")
            except Exception as e:
                print(f"    Error getting encodings: {e}")
        
        # If we have working images, create a simple face recognition script
        print(f"\n[CREATING WORKING SCRIPT]")
        create_working_script(successful_images)
    else:
        print("\n✗ NO WORKING IMAGES FOUND")
        print("\nRecommendations:")
        print("1. Try different source images (webcam selfies work well)")
        print("2. Ensure images are not corrupted")
        print("3. Try smaller image sizes (resize to 400x400)")
        print("4. Check your face_recognition installation")

def create_working_script(successful_images):
    """Create a working face recognition script based on successful test results"""
    script_content = '''import face_recognition
import cv2
import numpy as np
import os

# Load known faces (using the method that worked in testing)
known_face_encodings = []
known_face_names = []

'''
    
    for filename, _ in successful_images:
        name = os.path.splitext(filename)[0]
        script_content += f'''
# Load {filename}
try:
    image = face_recognition.load_image_file("known_faces/{filename}")
    image = np.ascontiguousarray(image)  # Fix memory layout
    locations = face_recognition.face_locations(image)
    if locations:
        encodings = face_recognition.face_encodings(image, locations)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append("{name}")
            print(f"✓ Loaded {filename}")
        else:
            print(f"✗ No encodings for {filename}")
    else:
        print(f"✗ No faces found in {filename}")
except Exception as e:
    print(f"✗ Failed to load {filename}: {{e}}")
'''
    
    script_content += '''
print(f"Loaded {len(known_face_encodings)} known faces")

if not known_face_encodings:
    print("No faces loaded. Exiting.")
    exit()

# Start webcam
video_capture = cv2.VideoCapture(0)
print("Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame)  # Ensure memory layout
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
'''
    
    with open('working_face_recognition.py', 'w') as f:
        f.write(script_content)
    
    print("✓ Created 'working_face_recognition.py'")
    print("  This script uses the methods that worked in testing.")

if __name__ == "__main__":
    main()