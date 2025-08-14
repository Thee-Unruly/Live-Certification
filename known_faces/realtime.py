import face_recognition #handles detection + embedding generation
import os

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir("known_faces"):
    path = os.path.join("known_faces", filename)
    img = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(img)
    if encodings:  # Make sure a face was found
        known_face_encodings.append(encodings[0])
        name = os.path.splitext(filename)[0]
        known_face_names.append(name)


