#!/bin/bash

echo "DriverEyeTracking Setup Script"

# Step 1: Update package list
echo "Updating system packages..."
sudo apt-get update

# Step 2: Install Python3 and pip (if not already installed)
echo "Checking/installing Python3 and pip3..."
sudo apt-get install -y python3 python3-pip

# Step 3: Install necessary system dependencies for pyttsx3
echo "Installing speech synthesis dependency (espeak)..."
sudo apt-get install -y libespeak1

# Step 4: Install Python packages
echo "Installing Python packages from requirements.txt..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Step 5: Final message
echo "Setup Complete! You're ready to run DriverEyeTracking."
echo "To start real-time detection, run:"
echo "python3 drowsiness_detector.py"
