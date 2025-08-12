# Liveness Certification Prototype

## Overview
This is a **proof-of-concept** application that verifies if a person is alive in real time using a webcam.  
The system randomly challenges the user to perform one of three actions:
- **Blink twice**
- **Smile**
- **Wave their hand**

If the correct action is detected within a short time window, the system confirms "Liveness Passed."

---

## Features
- **Random Challenge Selection** to prevent pre-recorded video spoofing.
- **Blink Detection** using Eye Aspect Ratio (EAR) from facial landmarks.
- **Smile Detection** based on mouth width-to-height ratio.
- **Wave Detection** using wrist movement tracking.
- **Real-Time Webcam Processing** with OpenCV.
- **Mediapipe Landmarks** for face and hand tracking.

---

## Technology Stack
- **Python 3.8+**
- **OpenCV** for video capture and image processing.
- **Mediapipe** for face and hand landmark detection.
- **NumPy** for mathematical calculations.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/liveness-certification.git
cd liveness-certification
