# train_model.py
from cnn_model import EyeClassifierCNN
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Config
data_dir = 'dataset/eye_states'
batch_size = 32
epochs = 10
model_path = 'model/eye_classifier.pth'

# Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Dataset & Loader
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = EyeClassifierCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
print("Training started...")
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f}")

# Save Model
os.makedirs('model', exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
