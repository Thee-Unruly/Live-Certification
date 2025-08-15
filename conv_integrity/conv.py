from PIL import Image
img = Image.open("known_faces/converted_me.jpg")
img.save("known_faces/converted_me.png", "PNG")
print("Saved as PNG: known_faces/converted_me.png")