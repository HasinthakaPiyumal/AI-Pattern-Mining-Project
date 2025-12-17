import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os


class PneumoniaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


class PneumoniaNet(nn.Module):
    def __init__(self):
        super(PneumoniaNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


def evaluate_model(model, data_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = running_loss / total_samples
    avg_acc = correct_predictions / total_samples
    return avg_loss, avg_acc


def predict_pneumonia(model, image_path, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        output = model(image)
        prediction = output.item()  # Get the scalar prediction

    return prediction


if __name__ == "__main__":
    # Simulate dummy data for demonstration
    # In a real scenario, these would be actual X-ray image paths and labels
    data_dir = "dummy_xray_data"
    os.makedirs(data_dir, exist_ok=True)

    dummy_image_paths = []
    dummy_labels = []
    for i in range(200):
        img_name = f"xray_{i:03d}.png"
        img_path = os.path.join(data_dir, img_name)
        # Create a dummy image file
        Image.new("RGB", (224, 224), color = (i % 255, (i+50)%255, (i+100)%255)).save(img_path)
        dummy_image_paths.append(img_path)
        # Simulate labels: 0 for normal, 1 for pneumonia
        dummy_labels.append(1 if i % 2 == 0 else 0)

    # Split data into training, validation, and test sets
    total_samples = len(dummy_image_paths)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size

    train_image_paths = dummy_image_paths[:train_size]
    train_labels = dummy_labels[:train_size]
    val_image_paths = dummy_image_paths[train_size:train_size + val_size]
    val_labels = dummy_labels[train_size:train_size + val_size]
    test_image_paths = dummy_image_paths[train_size + val_size:]
    test_labels = dummy_labels[train_size + val_size:]

    # Data transformations
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and dataloaders
    train_dataset = PneumoniaDataset(train_image_paths, train_labels, transform=data_transform)
    val_dataset = PneumoniaDataset(val_image_paths, val_labels, transform=inference_transform)
    test_dataset = PneumoniaDataset(test_image_paths, test_labels, transform=inference_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, criterion, and optimizer
    model = PneumoniaNet()
    criterion = nn.BCELoss()  # Binary Cross-Entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Starting model training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)
    print("Model training complete.")

    # Evaluate on the test set
    print("Evaluating model on test set...")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Perform inference on a new image
    if test_image_paths:
        sample_image_path = test_image_paths[0]
        prediction_prob = predict_pneumonia(model, sample_image_path, inference_transform)
        print(f"\nInference on {sample_image_path}:")
        print(f"Predicted probability of pneumonia: {prediction_prob:.4f}")
        if prediction_prob > 0.5:
            print("Prediction: Pneumonia Detected")
        else:
            print("Prediction: Normal")
        print(f"Actual label: {test_labels[0]} ({'Pneumonia' if test_labels[0] == 1 else 'Normal'})\n")

    # Clean up dummy data (optional)
    import shutil
    shutil.rmtree(data_dir)
    print(f"Cleaned up dummy data directory: {data_dir}")
