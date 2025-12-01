import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np

# --- 1. Image Preprocessing Module ---
def preprocess_image(image: np.ndarray) -> torch.Tensor:
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)

# --- 2. Prompt Encoder Module (Mock) ---
def encode_prompt(prompt: str) -> torch.Tensor:
    if prompt:
        # Return a dummy fixed-size tensor as a mock embedding
        # In a real scenario, this would use a pre-trained transformer model
        return torch.randn(1, 768)  # Example fixed size embedding
    else:
        return torch.zeros(1, 768)

# --- 3. Prompted Segmentation Model (Mock) ---
class MockPromptedSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # In a real model, this would be a U-Net or similar architecture
        # For mock, we just define a dummy layer to accept inputs
        self.dummy_conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, image_tensor: torch.Tensor, prompt_embedding: torch.Tensor) -> torch.Tensor:
        # Acknowledge prompt_embedding, but for this mock, it doesn't deeply influence
        # For simplicity, we'll generate a fake circular mask centered on the image
        # The image_tensor shape is (C, H, W), so we get H, W
        _, H, W = image_tensor.shape
        mask = torch.zeros((1, H, W), dtype=torch.float32)

        # Create a simple circular mask for demonstration
        center_x, center_y = W // 2, H // 2
        radius = min(W, H) // 4 # Adjust radius for a visible 