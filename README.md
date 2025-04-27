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
![image](https://github.com/user-attachments/assets/0b055eb3-c0be-428a-a435-3365e8db3a36)

## Setup Instructions
### Dependencies

Install these on your system or Raspberry Pi:
pip install torch torchvision opencv-python pyttsx3 matplotlib numpy

## Camera Support
This project supports:
Webcam (default: /dev/video0)
Raspberry Pi Camera (requires picamera2 or OpenCV)

## Model Training
Ensure the dataset is structured like this:

dataset/eye_states/
├── Open/
├── 75/
├── Half/
├── 25/
└── Closed/
Each folder contains images of eyes in that state.
To train: python train_model.py
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

## Future Work
Real-time Grad-CAM overlays
Lightweight MobileNet model for faster inference
Dashboard to log sleepiness scores over time
Integration with cloud alerting

## License
This project is licensed under MIT. Feel free to contribute, modify, or integrate into your own systems.

## Author
Built by Anh Phan
Contributions, feedback, and PRs are welcomed.
