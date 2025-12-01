import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cv2
import streamlit as st
import random

# 1. Behavioral Cloning Model (Deep Learning)
class SurgicalAssistantModel(nn.Module):
    def __init__(self, num_actions, visual_input_shape=(3, 224, 224), telemetry_input_dim=10):
        super(SurgicalAssistantModel, self).__init__()
        
        # Visual Branch (Simplified CNN)
        self.visual_branch = nn.Sequential(
            nn.Conv2d(visual_input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        # Calculate flattened size for visual branch
        # Dummy input to calculate output size
        dummy_visual_input = torch.zeros(1, *visual_input_shape)
        visual_output_size = self.visual_branch(dummy_visual_input).shape[1]
        
        # Telemetry Branch (MLP)
        self.telemetry_branch = nn.Sequential(
            nn.Linear(telemetry_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(visual_output_size + 32, 128),
            nn.ReLU()
        )
        
        # Output Layer
        self.output_layer = nn.Linear(128, num_actions)

    def forward(self, visual_input, telemetry_input):
        visual_features = self.visual_branch(visual_input)
        telemetry_features = self.telemetry_branch(telemetry_input)
        
        combined_features = torch.cat((visual_features, telemetry_features), dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        output = self.output_layer(fused_features)
        return output

# 2. Simulated Data Generation and Preprocessing
def generate_dummy_data(num_samples=100, visual_shape=(3, 224, 224), telemetry_dim=10, num_actions=5):
    images = []
    telemetry = []
    actions = []
    for _ in range(num_samples):
        # Simulate a simple image (e.g., a colored circle or square)
        img = np.zeros(visual_shape[1:], dtype=np.uint8)
        center_x, center_y = random.randint(50, 174), random.randint(50, 174)
        radius = random.randint(10, 30)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if random.random() > 0.5: # Draw circle
            cv2.circle(img, (center_x, center_y), radius, color, -1)
        else: # Draw rectangle
            cv2.rectangle(img, (center_x - radius, center_y - radius), (center_x + radius, center_y + radius), color, -1)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # Ensure 3 channels

        # Resize if necessary (OpenCV handles HWC, PyTorch expects CHW)
        img_resized = cv2.resize(img_rgb, (visual_shape[2], visual_shape[1]))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        images.append(transform(img_resized))
        
        telemetry.append(torch.randn(telemetry_dim))
        actions.append(torch.tensor(random.randint(0, num_actions - 1), dtype=torch.long))
    
    return torch.stack(images), torch.stack(telemetry), torch.stack(actions)

# 3. Training Function
def train_model(model, train_images, train_telemetry, train_actions, num_epochs=10, batch_size=32, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    dataset = torch.utils.data.TensorDataset(train_images, train_telemetry, train_actions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, telemetry, actions) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(images, telemetry)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        st.sidebar.write(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")
    st.sidebar.success("Model training complete!")
    torch.save(model.state_dict(), "surgical_assistant_model.pth")

# 4. Real-time Inference Module
def predict_action(model, visual_input, telemetry_input, action_labels):
    model.eval()
    with torch.no_grad():
        output = model(visual_input.unsqueeze(0), telemetry_input.unsqueeze(0)) # Add batch dimension
        predicted_class = torch.argmax(output, dim=1).item()
    
    # Simulate a warning based on random chance or telemetry values
    warning = ""
    if telemetry_input[0] > 1.5: # Example arbitrary threshold
        warning = "Warning: High Force Detected!"
    elif random.random() < 0.1:
        warning = "Caution: Instrument Angle Suboptimal."

    return action_labels[predicted_class], warning

# Streamlit Application
def streamlit_app():
    st.title("Surgical Assistant AI for Laparoscopic Procedures")
    st.sidebar.header("Configuration")

    num_actions = st.sidebar.slider("Number of Possible Actions", 2, 10, 5)
    telemetry_dim = st.sidebar.slider("Telemetry Data Dimension", 5, 20, 10)
    num_samples = st.sidebar.slider("Training Samples", 50, 500, 100)
    num_epochs = st.sidebar.slider("Training Epochs", 5, 20, 10)

    model = SurgicalAssistantModel(num_actions=num_actions, telemetry_input_dim=telemetry_dim)
    action_labels = [f"Action {i+1}" for i in range(num_actions)]

    if st.sidebar.button("Train Model"): 
        st.sidebar.write("Generating dummy data...")
        train_images, train_telemetry, train_actions = generate_dummy_data(num_samples, telemetry_dim=telemetry_dim, num_actions=num_actions)
        st.sidebar.write("Training model...")
        train_model(model, train_images, train_telemetry, train_actions, num_epochs=num_epochs)
        st.sidebar.success("Model trained and saved!")
    
    if st.sidebar.button("Load Pre-trained Model"): # For demonstration if model was saved previously
        try:
            model.load_state_dict(torch.load("surgical_assistant_model.pth"))
            model.eval()
            st.sidebar.success("Model loaded successfully!")
        except FileNotFoundError:
            st.sidebar.error("No trained model found. Please train the model first.")

    st.header("Live Surgery Simulation")
    st.write("Click 'Simulate Live Surgery' to see real-time guidance.")

    image_placeholder = st.empty()
    telemetry_placeholder = st.empty()
    prediction_placeholder = st.empty()
    warning_placeholder = st.empty()

    if st.button("Simulate Live Surgery"): 
        st.write("Simulating surgical procedure...")
        for i in range(20): # Simulate 20 timesteps
            # Generate dummy live data
            live_image_np = np.zeros((224, 224, 3), dtype=np.uint8)
            center_x, center_y = random.randint(50, 174), random.randint(50, 174)
            radius = random.randint(10, 30)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if random.random() > 0.5:
                cv2.circle(live_image_np, (center_x, center_y), radius, color, -1)
            else:
                cv2.rectangle(live_image_np, (center_x - radius, center_y - radius), (center_x + radius, center_y + radius), color, -1)
            
            live_telemetry_data = torch.randn(telemetry_dim)

            # Preprocess live data
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            live_image_tensor = transform(live_image_np)

            # Get prediction
            predicted_action, warning = predict_action(model, live_image_tensor, live_telemetry_data, action_labels)
            
            # Display results
            image_placeholder.image(live_image_np, caption=f"Live Camera Feed - Step {i+1}", use_column_width=True)
            telemetry_placeholder.write(f"Current Telemetry: {live_telemetry_data[:3].numpy().round(2)}...")
            prediction_placeholder.subheader(f"Predicted Action: {predicted_action}")
            if warning:
                warning_placeholder.warning(warning)
            else:
                warning_placeholder.empty()
            
            import time
            time.sleep(0.5) # Simulate real-time delay

if __name__ == "__main__":
    streamlit_app()