# cnn_model.py
import torch.nn as nn
import torch.nn.functional as F

class EyeClassifierCNN(nn.Module):
    def __init__(self):
        super(EyeClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 5)  # 5 eye states

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (32, 24, 24)
        x = self.pool(F.relu(self.conv2(x)))  # -> (64, 12, 12)
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
