# data_augmentor.py
# python data_augmentor.py
"""
Blending formula:
Synthetic = Open × (1 - α) + Closed × α

For example:

75% Open = 25% Closed → α = 0.25

Half Open = 50% Closed → α = 0.5

25% Open = 75% Closed → α = 0.75

(We "close" the eye more and more through blending.)
"""
import os
import cv2
import random
import numpy as np

def blend_images(open_img, closed_img, alpha):
    """
    Blend two images together with a given alpha.
    """
    return cv2.addWeighted(open_img, 1 - alpha, closed_img, alpha, 0)

def generate_synthetic_images(open_dir, closed_dir, output_base_dir, num_samples_per_class=500):
    """
    Generate synthetic 75%, Half, 25% open eye images.
    """
    open_images = [os.path.join(open_dir, img) for img in os.listdir(open_dir)]
    closed_images = [os.path.join(closed_dir, img) for img in os.listdir(closed_dir)]

    targets = {
        "75": 0.25,     # 75% open = 25% closed
        "Half": 0.5,    # Half open
        "25": 0.75      # 25% open = 75% closed
    }

    for label, alpha in targets.items():
        output_dir = os.path.join(output_base_dir, label)
        os.makedirs(output_dir, exist_ok=True)

        for i in range(num_samples_per_class):
            open_img_path = random.choice(open_images)
            closed_img_path = random.choice(closed_images)

            open_img = cv2.imread(open_img_path, cv2.IMREAD_GRAYSCALE)
            closed_img = cv2.imread(closed_img_path, cv2.IMREAD_GRAYSCALE)

            if open_img is None or closed_img is None:
                continue

            # Resize if needed
            open_img = cv2.resize(open_img, (48, 48))
            closed_img = cv2.resize(closed_img, (48, 48))

            synthetic_img = blend_images(open_img, closed_img, alpha)

            out_path = os.path.join(output_dir, f"{label}_{i}.jpg")
            cv2.imwrite(out_path, synthetic_img)

        print(f"✅ Generated {num_samples_per_class} images for '{label}' class.")

if __name__ == "__main__":
    open_dir = "dataset/eye_states/Open"
    closed_dir = "dataset/eye_states/Closed"
    output_base_dir = "dataset/eye_states"

    generate_synthetic_images(open_dir, closed_dir, output_base_dir)
