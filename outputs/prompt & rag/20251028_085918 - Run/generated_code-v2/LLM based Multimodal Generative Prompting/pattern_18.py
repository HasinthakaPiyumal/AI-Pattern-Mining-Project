from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
import os

class PromptEncoder:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def encode_prompt(self, text_prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        prompt_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
        return prompt_embedding

class ImageProcessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.grayscale_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def load_image(self, image_path: str) -> torch.Tensor:
        try:
            img = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at {image_path}")
        except Exception as e:
            print(f"Warning: Could not load as RGB, trying grayscale. Error: {e}")
            img = Image.open(image_path).convert("L")
            return self.grayscale_transform(img).unsqueeze(0)

        return self.transform(img).unsqueeze(0)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class PromptGuidedSegmentationModel(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, prompt_embedding_dim=768, initial_features=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.prompt_embedding_dim = prompt_embedding_dim

        self.inc = DoubleConv(n_channels, initial_features)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(initial_features, initial_features * 2)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(initial_features * 2, initial_features * 4)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(initial_features * 4, initial_features * 8)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(initial_features * 8, initial_features * 16)

        self.up1 = nn.ConvTranspose2d(initial_features * 16, initial_features * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(initial_features * 16, initial_features * 8)
        self.up2 = nn.ConvTranspose2d(initial_features * 8, initial_features * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(initial_features * 8, initial_features * 4)
        self.up3 = nn.ConvTranspose2d(initial_features * 4, initial_features * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(initial_features * 4, initial_features * 2)
        self.up4 = nn.ConvTranspose2d(initial_features * 2, initial_features, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(initial_features * 2, initial_features)

        self.outc = nn.Conv2d(initial_features, n_classes, kernel_size=1)

        self.prompt_proj = nn.Linear(prompt_embedding_dim, initial_features * 16)

    def forward(self, image_tensor: torch.Tensor, prompt_embedding: torch.Tensor) -> torch.Tensor:
        x = image_tensor

        projected_prompt = self.prompt_proj(prompt_embedding)
        projected_prompt = projected_prompt.unsqueeze(-1).unsqueeze(-1)

        x1 = self.inc(x)
        x2 = self.conv1(self.down1(x1))
        x3 = self.conv2(self.down2(x2))
        x4 = self.conv3(self.down3(x3))
        x5 = self.conv4(self.down4(x4))

        if x5.shape[0] == projected_prompt.shape[0]:
            x5 = x5 + projected_prompt
        else:
            print("Warning: Prompt embedding batch size does not match image batch size. Skipping direct addition.")

        x = self.up1(x5)
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
        return torch.sigmoid(logits)

def main(image_path: str, prompt_text: str, output_path: str = "output_segmentation.png"):
    print(f"Initializing components...")
    prompt_encoder = PromptEncoder()
    image_processor = ImageProcessor()
    segmentation_model = PromptGuidedSegmentationModel(n_channels=3, n_classes=1, prompt_embedding_dim=768)
    segmentation_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmentation_model.to(device)

    print(f"Processing image: {image_path}")
    try:
        image_tensor = image_processor.load_image(image_path).to(device)
        if image_tensor.shape[1] == 1:
            print("Warning: Loaded grayscale image. Adjusting model input channels to 1.")
            segmentation_model = PromptGuidedSegmentationModel(n_channels=1, n_classes=1, prompt_embedding_dim=768)
            segmentation_model.to(device)
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"Error loading or processing image: {e}")
        return

    print(f"Encoding prompt: \"{prompt_text}\"")
    prompt_embedding = prompt_encoder.encode_prompt(prompt_text).unsqueeze(0).to(device)

    print(f"Performing segmentation...")
    with torch.no_grad():
        segmentation_mask_logits = segmentation_model(image_tensor, prompt_embedding)

    binary_mask = (segmentation_mask_logits > 0.5).float()

    segmentation_np = binary_mask.squeeze().cpu().numpy()
    original_image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_image_np = std * original_image_np + mean
    original_image_np = np.clip(original_image_np, 0, 1)

    print(f"Saving segmentation output to {output_path}")
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image_np)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(original_image_np, cmap='gray')
    plt.imshow(segmentation_np, cmap='jet', alpha=0.5 * (segmentation_np > 0))
    plt.title(f"Segmentation for: \"{prompt_text}\"")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print("Segmentation complete. Output saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Image Analysis Assistant using Prompted Segmentation.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the medical image (e.g., .png, .jpg).")
    parser.add_argument("--prompt", type=str, required=True, help="Natural language prompt for segmentation (e.g., 'segment the left ventricle').")
    parser.add_argument("--output_path", type=str, default="output_segmentation.png", help="Path to save the output segmentation image.")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Creating a dummy image at {args.image_path} for demonstration.")
        dummy_img = Image.new('RGB', (512, 512), color = 'green')
        draw = ImageDraw.Draw(dummy_img)
        draw.ellipse((100, 100, 300, 300), fill='red', outline='red')
        dummy_img.save(args.image_path)
        print("Dummy image created. You can now run the script with this image.")

    main(args.image_path, args.prompt, args.output_path)
