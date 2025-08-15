from PIL import Image
import numpy as np

img = Image.open("known_faces/converted_me.png")
print(f"Image mode: {img.mode}")
img_array = np.array(img)
print(f"Array shape: {img_array.shape}, dtype: {img_array.dtype}")
print(f"Min value: {np.min(img_array)}, Max value: {np.max(img_array)}")