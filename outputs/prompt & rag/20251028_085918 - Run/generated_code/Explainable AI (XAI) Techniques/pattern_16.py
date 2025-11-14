import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lime import lime_image

# --- 1. Dummy Black-box AI Model (Core Diagnostic Engine) ---
class BlackBoxModel(nn.Module):
    def __init__(self, num_classes=2):
        super(BlackBoxModel, self).__init__()
        # A very simple CNN for demonstration purposes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128) # Assuming input image of 224x224, downsampled twice by pool
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56) # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_dummy_model(num_classes=2):
    model = BlackBoxModel(num_classes=num_classes)
    # For demonstration, we'll just initialize weights randomly.
    # In a real scenario, you'd load pre-trained weights: model.load_state_dict(torch.load('model.pth'))
    model.eval() # Set to evaluation mode
    return model

# --- 2. Data Ingestion & Preprocessing Module ---
def preprocess_image(image: Image.Image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0) # Add batch dimension

def deprocess_image(tensor: torch.Tensor):
    # Reverse normalization and convert to numpy for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img = inv_normalize(tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

# --- 3. Explainable AI (XAI) Module - LIME ---
def get_lime_explainer():
    return lime_image.LimeImageExplainer()

def batch_predict(images_np, model, device):
    # images_np is (N, H, W, C) numpy array of images (0-1 range)
    # Convert to tensor and preprocess for model
    tensor_batch = []
    for img_np in images_np:
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        tensor_batch.append(transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(img_pil))
    
    tensor_batch = torch.stack(tensor_batch).to(device)
    with torch.no_grad():
        outputs = model(tensor_batch)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    return probabilities

def visualize_lime_heatmap(original_image_np: np.array, explanation_map: np.array, 
                           positive_only: bool = True, num_features: int = 5, hide_rest: bool = True):
    
    temp, mask = explanation_map
    # Convert mask to 0-255 for cv2
    mask = np.uint8(mask * 255)
    
    # Resize mask to original image size
    mask_resized = cv2.resize(mask, (original_image_np.shape[1], original_image_np.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask_resized_3ch = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

    # Apply colormap to mask
    heatmap = cv2.applyColorMap(mask_resized_3ch, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Overlay heatmap on original image
    overlay = original_image_np.copy()
    alpha = 0.5 # Transparency factor
    cv2.addWeighted(heatmap, alpha, original_image_np, 1 - alpha, 0, overlay)
    
    return overlay

# --- 4. Interactive User Interface (UI) / Dashboard (Streamlit Application) ---
st.set_page_config(layout="wide", page_title="AI Diagnostic Assistant with XAI")
st.title("AI-Powered Diagnostic Assistant with Explainable AI")
st.markdown("Upload a medical image (e.g., X-ray) to get a diagnosis and its explanation.")

# Load dummy model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_model = load_dummy_model(num_classes=2).to(device) # Assuming 2 classes: 'Healthy' and 'Disease'
class_names = ["Healthy", "Disease"]

# Image Uploader
uploaded_file = st.file_uploader("Choose a medical image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Analyzing...")

    # Preprocess image for model
    input_tensor = preprocess_image(image).to(device)
    original_image_np = np.array(image.resize((224, 224))) / 255.0 # For LIME input

    # Make prediction
    with torch.no_grad():
        outputs = dummy_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class_name = class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx] * 100

    st.subheader("AI Diagnosis")
    st.success(f"Prediction: **{predicted_class_name}** with **{confidence:.2f}%** confidence.")

    st.subheader("Explanation (LIME)")
    st.write("LIME highlights image regions that contribute most to the AI's prediction.")

    explainer = get_lime_explainer()
    # LIME expects a numpy array of (H, W, C) where values are 0-1
    # Define a wrapper for model prediction that LIME can use
    def model_predict_fn(images):
        return batch_predict(images, dummy_model, device)

    # Explain the instance
    # num_features: number of superpixels to include in explanation
    # hide_rest: if True, hides the non-contributing superpixels
    explanation = explainer.explain_instance(
        original_image_np, 
        model_predict_fn, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000
    )

    # Get image with explanation
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=True
    )
    
    # Convert temp to numpy array 0-1 range for visualization function
    temp_np = (temp * 255).astype(np.uint8) / 255.0
    
    lime_overlay = visualize_lime_heatmap(temp_np, (temp, mask))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image_np)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(lime_overlay)
    ax[1].set_title(f"LIME Explanation for '{predicted_class_name}'")
    ax[1].axis("off")

    st.pyplot(fig)

    st.markdown("--- More XAI techniques and interactive features would be integrated here ---")
    st.info("This is a simplified demonstration. A full framework would include global explanations (PDP, PFI), counterfactuals, and subgroup analysis for bias detection.")

else:
    st.info("Please upload an image to start the diagnostic process.")