# Driver Drowsiness Detection System (Raspberry Pi Compatible)

This project is a real-time **driver eye monitoring system** that detects drowsiness levels using a **deep learning model** and triggers a voice alert when necessary. Designed to run on embedded systems like **Raspberry Pi**, it classifies eye openness into 5 categories using a **CNN model**, enhanced with **Grad-CAM visualization** and **TTS voice warnings**.

---

## Features
- Classifies eye openness into:
  - **Open**
  - **75% Open**
  - **Half Open**
  - **25% Open**
  - **Closed**
- Custom CNN model for eye classification
- Real-time drowsiness detection via webcam or Pi camera
- Voice alerts using `pyttsx3`
- Grad-CAM visualization for model explainability
- Modular design for training, inference, and deployment

---

## Project Structure
![image](https://github.com/user-attachments/assets/36d19dc0-1c2e-46b3-8807-77a18d50439d)

## Setup Instructions
### Requirements: use pip install -r requirements.txt

## Camera Support
This project supports:
Webcam (default: /dev/video0)
Raspberry Pi Camera (requires picamera2 or OpenCV)

## Model Training
python train_model.py.
This creates a model file: model/eye_classifier.pth

## Running the Real-Time Detector
python drowsiness_detector.py
It will:
- Detect eyes in real-time
- Classify eye state
- Display the label on screen
- Trigger a voice alert if eyes remain in 25% open or closed for a threshold number of frames

## Grad-CAM Visualization
To visualize where the model is focusing (helpful for debugging or model introspection):

from gradcam_visualizer import GradCAM
gradcam = GradCAM(model)
heatmap = gradcam.generate(sample_tensor)

## Voice Alerts
Uses pyttsx3 for offline TTS. When drowsiness is detected:
🔊 "Detecting low focus!"
Customize this in voice_alert.py.

## License
This project is licensed under MIT. Feel free to contribute, modify, or integrate into your own systems.

## Author
Built by Anh Phan
Contributions, feedback, and PRs are welcomed.
