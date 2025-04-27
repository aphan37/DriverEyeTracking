# drowsiness_detector.py

import cv2
import torch
import numpy as np
import time
import csv
from torchvision import transforms
from cnn_model import EyeClassifierCNN
from gradcam_visualizer import GradCAM
from voice_alert import speak_alert
from eye_detector import eye_aspect_ratio

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EyeClassifierCNN().to(device)
model.load_state_dict(torch.load('model/eye_classifier.pth', map_location=device))
model.eval()

# Initialize Grad-CAM
gradcam = GradCAM(model)

categories = ['Open', '75%', 'Half', '25%', 'Closed']
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# OpenCV setup
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

frame_counter = 0
alert_threshold = 10
sleepiness_score = 100  # Start fully awake (100)

# Logging setup
log_filename = f"driver_log_{int(time.time())}.csv"
with open(log_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "EAR", "Classification", "Sleepiness Score"])

print(f"🔵 Logging to {log_filename}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_resized = cv2.resize(eye, (48, 48))
            eye_tensor = transform(eye_resized).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(eye_tensor)
                idx = torch.argmax(output)
                label = categories[idx]

            # EAR Calculation (optional)
            eye_pts = np.array([
                [0, eh//2], [ew//4, 0], [3*ew//4, 0],
                [ew, eh//2], [3*ew//4, eh], [ew//4, eh]
            ])
            ear = eye_aspect_ratio(eye_pts)

            # Grad-CAM generation
            eye_tensor.requires_grad = True
            heatmap = gradcam.generate(eye_tensor.cpu())
            heatmap = cv2.resize(heatmap, (ew, eh))
            heatmap = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            eye_color = cv2.cvtColor(eye_resized, cv2.COLOR_GRAY2BGR)
            eye_color = cv2.resize(eye_color, (ew, eh))
            blended = cv2.addWeighted(eye_color, 0.6, heatmap_color, 0.4, 0)

            frame[y+ey:y+ey+eh, x+ex:x+ex+ew] = blended

            # Draw label
            cv2.putText(frame, f"{label}", (x+ex, y+ey-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Sleepiness score logic
            if label in ['25%', 'Closed']:
                frame_counter += 1
                sleepiness_score -= 0.5  # decay faster if drowsy
                if frame_counter >= alert_threshold:
                    speak_alert()
            else:
                frame_counter = 0
                sleepiness_score += 0.2  # recover if awake

            sleepiness_score = max(0, min(100, sleepiness_score))  # keep between 0 and 100

            # Logging
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(log_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, f"{ear:.2f}", label, int(sleepiness_score)])

            # Show Sleepiness Score
            cv2.putText(frame, f"Sleepiness: {int(sleepiness_score)}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Driver Eye Tracking + GradCAM + Sleepiness Score', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
