# drowsiness_detector.py (updated version)

import cv2
import torch
import numpy as np
from torchvision import transforms
from cnn_model import EyeClassifierCNN
from gradcam_visualizer import GradCAM
from voice_alert import speak_alert

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
        eye_resized = cv2.resize(eye, (48, 48))
        eye_tensor = transform(eye_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(eye_tensor)
            idx = torch.argmax(output)
            label = categories[idx]

        # Grad-CAM generation
        eye_tensor.requires_grad = True
        heatmap = gradcam.generate(eye_tensor.cpu())

        # Resize heatmap to match original eye crop
        heatmap = cv2.resize(heatmap, (ew, eh))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Blend heatmap on eye region
        eye_color = cv2.cvtColor(eye_resized, cv2.COLOR_GRAY2BGR)
        eye_color = cv2.resize(eye_color, (ew, eh))
        blended = cv2.addWeighted(eye_color, 0.6, heatmap_color, 0.4, 0)

        # Replace original eye region with blended Grad-CAM visualization
        frame[ey:ey+eh, ex:ex+ew] = blended

        # Draw label
        cv2.putText(frame, f"Eye: {label}", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        if label in ['25%', 'Closed']:
            frame_counter += 1
            if frame_counter >= alert_threshold:
                speak_alert()
        else:
            frame_counter = 0

    cv2.imshow('Driver Eye Tracking + GradCAM', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
