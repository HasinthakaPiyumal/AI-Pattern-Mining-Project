import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image


# 1. Placeholder CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # Assuming input image size 224x224
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)  # Flatten the tensor
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model (using a dummy pre-trained state for demonstration)
model = SimpleCNN(num_classes=2)
# In a real application, you would load a trained model state_dict here
# For now, let's just make it pretend to be trained.
# model.load_state_dict(torch.load('path_to_your_model.pth'))
model.eval() # Set to evaluation mode

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define inverse transformation for LIME visualization
inv_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                         std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.ToPILImage(),
])

# Class labels (example)
class_names = ['Normal', 'Abnormal']

# 2. LIME Explainer Integration
def predict_fn(images):
    # images will be a numpy array from LIME
    # Convert numpy array to PIL Image, then apply transform and predict
    img_tensor_batch = []
    for img_np in images:
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8)) # LIME output is float 0-1
        img_tensor_batch.append(transform(img_pil))
    
    img_tensor_batch = torch.stack(img_tensor_batch)
    
    with torch.no_grad():
        outputs = model(img_tensor_batch)
        probabilities = F.softmax(outputs, dim=1).numpy()
    return probabilities

explainer = lime_image.LimeImageExplainer()

# 3. Streamlit UI
st.set_page_config(layout="wide", page_title="Medical Image Interpretability Platform")
st.title("🧠 Medical Image Interpretability Platform")
st.markdown("Upload a medical image to get a diagnostic prediction and understand the model's reasoning.")

uploaded_file = st.sidebar.file_uploader("Choose a medical image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)
    st.subheader("Model Prediction")
    
    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class = class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx].item() * 100
    
    st.write(f"**Diagnosis:** `{predicted_class}` (Confidence: `{confidence:.2f}%`)")
    
    st.markdown("---")
    
    # Local Interpretability (LIME)
    st.subheader("Local Explanation (LIME)")
    st.write("LIME highlights the regions in the image that are most important for the model's prediction for this specific instance.")
    
    # Convert PIL image to numpy array for LIME
    img_np = np.array(image)

    with st.spinner("Generating LIME explanation..."):
        explanation = explainer.explain_instance(
            img_np,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=1000
        )
    
        temp_fig, plot_ax = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(temp_fig)
        ax.imshow(plot_ax, alpha=0.5, cmap='viridis')
        ax.set_title(f"LIME Explanation for '{class_names[explanation.top_labels[0]]}'")
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # Global Interpretability (Conceptual)
    st.subheader("Global Explanations (Conceptual)")
    st.write("Global interpretability techniques aim to understand the overall behavior of the model, not just individual predictions.")
    st.markdown(
        "**Partial Dependence Plots (PDPs):** Illustrate the marginal effect of one or two features on the predicted outcome of a model. "
        "For image data, this typically involves extracting meaningful features (e.g., handcrafted features, activations from intermediate CNN layers) "
        "and then applying PDPs to these features to see how they influence the overall diagnosis. This can reveal general trends, "
        "such as how the presence of certain textures or shapes consistently impacts the prediction."
    )
    st.markdown(
        "**Permutation Feature Importance:** Measures the importance of a feature by calculating the increase in the model's prediction error "
        "after permuting the feature's values, which breaks the relationship between the feature and the true outcome. "
        "Similar to PDPs, this would be applied to extracted image features or associated metadata, helping identify which high-level "
        "visual characteristics (or clinical data if available) are most crucial for the model's general performance."
    )

    st.markdown("---ادث")

    # Bias and Fairness Diagnostics (Conceptual)
    st.subheader("Bias and Fairness Diagnostics (Conceptual)")
    st.write("Identifying and characterizing data subgroups exhibiting divergent model behaviors is crucial for uncovering biases and ensuring fairness.")
    st.markdown(
        "**DivExplorer-like Analysis:** This conceptual module would involve segmenting the dataset into various subgroups based on demographic "
        "information (e.g., age, gender, ethnicity), imaging modality, or disease stage. The platform would then analyze model performance "
        "metrics (e.g., accuracy, sensitivity, specificity) for each subgroup. Significant discrepancies in performance "
        "across these subgroups would indicate potential biases, prompting further investigation and targeted model improvements. "
        "For instance, if the model consistently misdiagnoses a particular condition for a specific age group or ethnicity, "
        "this module would flag such a discrepancy, enabling medical professionals to address the bias."
    )

else:
    st.info("Please upload a medical image to get started. Example images include X-rays, MRIs, or CT scans.")
