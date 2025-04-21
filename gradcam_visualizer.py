# gradcam_visualizer.py
import torch
import numpy as np
import cv2
import torch.nn as nn

def get_last_conv_layer(model):
    for name, module in reversed(model._modules.items()):
        if isinstance(module, nn.Conv2d):
            return name
    return None

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.target_layer = target_layer or get_last_conv_layer(model)

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        for name, module in model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate(self, input_image):
        self.model.eval()
        output = self.model(input_image)
        class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients[0].detach().numpy()
        activations = self.activations[0].detach().numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (48, 48))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam
