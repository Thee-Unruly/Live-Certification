# Test individual images
import cv2

def test_image(path):
    try:
        img = cv2.imread(path)
        if img is None:
            print("OpenCV can't read this file")
            return
        print(f"Shape: {img.shape}, Type: {img.dtype}")
        cv2.imshow("Test", img)
        cv2.waitKey(0)
    except Exception as e:
        print(f"Error: {str(e)}")

test_image("known_faces/me.jpg")