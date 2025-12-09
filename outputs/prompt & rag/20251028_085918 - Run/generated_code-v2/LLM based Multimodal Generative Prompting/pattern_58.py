import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModel
import gradio as gr
import cv2

class PromptEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", output_dim=256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.model.config.hidden_size, output_dim)

    def forward(self, text_prompt):
        inputs = self.tokenizer(text_prompt, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        prompt_embedding = outputs.last_hidden_state[:, 0, :]
        return self.linear(prompt_embedding)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class PromptGuidedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, prompt_embedding_dim=256):
        super().__init__()

        self.inc = ConvBlock(in_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(512, 1024)

        self.prompt_fusion = nn.Linear(prompt_embedding_dim, 1024)

        self.up1 = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2)
        self.conv_up1 = ConvBlock(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = ConvBlock(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = ConvBlock(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, prompt_embedding):
        x1 = self.inc(x)
        x2 = self.conv1(self.down1(x1))
        x3 = self.conv2(self.down2(x2))
        x4 = self.conv3(self.down3(x3))
        x5 = self.conv4(self.down4(x4))

        prompt_features = self.prompt_fusion(prompt_embedding)
        prompt_features = prompt_features.unsqueeze(-1).unsqueeze(-1)
        prompt_features = prompt_features.expand_as(x5)

        x5_fused = torch.cat([x5, prompt_features], dim=1)

        x = self.up1(x5_fused)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        logits = self.outc(x)
        return logits

IMG_SIZE = 256

preprocess_image = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def postprocess_mask(logits):
    mask = torch.sigmoid(logits)
    mask = (mask > 0.5).float()
    return mask.squeeze().cpu().numpy()

def predict_segmentation(image, prompt_text, prompt_encoder_model, segmentation_model):
    pil_image = Image.fromarray(image).convert("RGB")
    input_image_tensor = preprocess_image(pil_image).unsqueeze(0)

    with torch.no_grad():
        prompt_embedding = prompt_encoder_model(prompt_text)

    with torch.no_grad():
        logits = segmentation_model(input_image_tensor, prompt_embedding)

    segmentation_mask = postprocess_mask(logits)

    original_np = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE)))
    mask_colored = np.zeros_like(original_np)
    mask_colored[segmentation_mask > 0] = [255, 0, 0]

    overlay = cv2.addWeighted(original_np, 0.7, mask_colored, 0.3, 0)

    return original_np, overlay, segmentation_mask

prompt_embedding_dim = 256
prompt_encoder = PromptEncoder(output_dim=prompt_embedding_dim)
segmentation_model = PromptGuidedUNet(prompt_embedding_dim=prompt_embedding_dim)

prompt_encoder.eval()
segmentation_model.eval()

def gradio_interface_fn(image, prompt_text):
    original_img_display, overlay_img_display, mask_display = predict_segmentation(
        image, prompt_text, prompt_encoder, segmentation_model
    )
    return original_img_display, overlay_img_display, mask_display

if __name__ == "__main__":
    gr.Interface(
        fn=gradio_interface_fn,
        inputs=[
            gr.Image(type="numpy", label="Upload Medical Image"),
            gr.Textbox(label="Segmentation Prompt (e.g., 'segment the glioblastoma')", placeholder="e.g., 'segment the glioblastoma in the frontal lobe'"),
        ],
        outputs=[
            gr.Image(type="numpy", label="Original Image (Resized)"),
            gr.Image(type="numpy", label="Segmentation Overlay"),
            gr.Image(type="numpy", label="Binary Segmentation Mask"),
        ],
        title="AI-Powered Prompt-Guided Medical Image Segmentation",
        description="Upload a medical image and provide a natural language prompt to segment specific regions. This is a demonstration with randomly initialized models; real-world usage requires pre-trained models.",
        examples=[
            [np.zeros((256, 256, 3), dtype=np.uint8), "segment the tumor"],
            [np.ones((256, 256, 3), dtype=np.uint8) * 128, "highlight all suspicious lesions"],
        ]
    ).launch()