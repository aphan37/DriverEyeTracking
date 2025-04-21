# drowsiness_detector.py
import cv2
import torch
import numpy as np
from torchvision import transforms
from cnn_model import EyeClassifierCNN
from voice_alert import speak_alert

# Load model
model = EyeClassifierCNN()
model.load_state_dict(torch.load('model/eye_classifier.pth'))
model.eval()

categories = ['Open', '75%', 'Half', '25%', 'Closed']
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# OpenCV setup
cap = cv2.VideoCapture(0)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

frame_counter = 0
alert_threshold = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    for (ex, ey, ew, eh) in eyes:
        eye = gray[ey:ey+eh, ex:ex+ew]
        eye = cv2.resize(eye, (48, 48))
        eye_tensor = transform(eye).unsqueeze(0)

        with torch.no_grad():
            output = model(eye_tensor)
            idx = torch.argmax(output)
            label = categories[idx]

        cv2.putText(frame, f"Eye: {label}", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        if label in ['25%', 'Closed']:
            frame_counter += 1
            if frame_counter >= alert_threshold:
                speak_alert()
        else:
            frame_counter = 0

    cv2.imshow('Driver Eye Tracker', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
