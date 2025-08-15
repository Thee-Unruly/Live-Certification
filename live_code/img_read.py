import cv2
import numpy as np
img = cv2.imread("known_faces/me_3.jpg")
if img is None:
    print("Failed to load image with OpenCV")
else:
    print("Image loaded successfully:", img.shape, img.dtype)