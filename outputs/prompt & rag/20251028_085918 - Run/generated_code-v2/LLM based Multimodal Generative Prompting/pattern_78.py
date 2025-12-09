import gradio as gr
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from sentence_transformers import SentenceTransformer

# --- 1. Medical Image Preprocessing Module ---
def preprocess_image(image_np):
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).float()
    return img_tensor

# --- 2. Prompt Understanding Module ---
class PromptEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_prompt(self, prompt_text):
        embedding = self.model.encode(prompt_text, convert_to_tensor=True)
        return embedding.unsqueeze(0) # Add batch dimension

prompt_embedder = PromptEmbedder()

# --- 3. Promptable Segmentation Model (Dummy U-Net like) ---
class PromptableSegmentationModel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64 + 384, 32, kernel_size=2, stride=2), nn.ReLU(), # 384 from 768 / 2 (prompt embedding size / 2)
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1), 
            nn.Sigmoid()
        )
        self.prompt_proj = nn.Linear(384, 64 * (128 // 2) * (128 // 2)) # Project prompt to spatial features (simplified)

    def forward(self, image_tensor, prompt_embedding):
        # Dummy downsample prompt_embedding to match a feature map size if needed
        # For simplicity, let's assume prompt_embedding is 1x384 (all-MiniLM-L6-v2)
        # and we need to integrate it into a 1x64x128x128 feature map
        
        # Prompt projection to spatial features (simplified for demonstration)
        # Target size for spatial features after encoding would be something like 64x128x128 if input is 1x256x256
        # Assuming prompt_embedding is [1, 384] (batch_size, embedding_dim)
        prompt_spatial_features = self.prompt_proj(prompt_embedding)
        prompt_spatial_features = prompt_spatial_features.view(prompt_embedding.shape[0], 64, 128, 128)

        enc_features = self.encoder(image_tensor)
        
        # Concatenate prompt features (simplified to match dimensions)
        # This is a very basic way to 