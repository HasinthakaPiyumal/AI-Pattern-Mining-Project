import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

# --- Constants ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5  # Example: 0 (No DR) to 4 (Proliferative DR)

# --- 1. Data Acquisition and Preprocessing (Dummy Data Generation) ---
def create_dummy_dataset(num_samples=100):
    print("Creating dummy dataset...")
    X = np.random.rand(num_samples, IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32)
    y = np.random.randint(0, NUM_CLASSES, num_samples)
    # Simulate some basic image preprocessing for dummy images
    X = X * 255 # Scale to 0-255
    for i in range(num_samples):
        # Simulate a very basic 'retinal' pattern or just noise for dummy purposes
        X[i] = cv2.GaussianBlur(X[i], (5, 5), 0)
        # Resize is already handled by numpy array creation, but for real images cv2.resize would be used.
    X = X / 255.0 # Normalize to 0-1
    print(f"Dummy dataset created: {X.shape} images, {y.shape} labels")
    return X, y

# Data augmentation setup
data_augmentor = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# --- 2. Model Architecture (Convolutional Neural Network - CNN) ---
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# --- 3. Training and Evaluation ---
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    # One-hot encode labels for categorical cross-entropy
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )

    print("Model training started...")
    history = model.fit(
        data_augmentor.flow(X_train, y_train_one_hot, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val_one_hot)
    )
    print("Model training finished.")

    print("Model evaluation started...")
    loss, accuracy, precision, recall, auc = model.evaluate(X_val, y_val_one_hot, verbose=2)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation AUC: {auc:.4f}")
    print("Model evaluation finished.")
    return history

# --- 4. Deployment (Inference Function) ---
def predict_retinopathy(model, image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction, axis=1)[0]
    
    class_labels = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }

    return {
        "predicted_class_id": predicted_class,
        "predicted_label": class_labels.get(predicted_class, "Unknown"),
        "confidence": float(confidence),
        "raw_prediction": prediction.tolist()
    }

if __name__ == "__main__":
    # Generate dummy data
    X_dummy, y_dummy = create_dummy_dataset(num_samples=200)
    
    # Split dummy data into training and validation sets
    split_idx = int(len(X_dummy) * 0.8)
    X_train, X_val = X_dummy[:split_idx], X_dummy[split_idx:]
    y_train, y_val = y_dummy[:split_idx], y_dummy[split_idx:]

    # Build the model
    input_shape = (*IMAGE_SIZE, 3)
    model = build_cnn_model(input_shape, NUM_CLASSES)
    model.summary()

    # Train and evaluate the model
    train_and_evaluate_model(model, X_train, y_train, X_val, y_val)

    # Save the trained model
    model_save_path = "diabetic_retinopathy_model.h5"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Example of how to load and use the model for prediction
    print("\n--- Demonstrating Prediction ---")
    loaded_model = tf.keras.models.load_model(model_save_path)

    # Create a dummy image file for demonstration
    dummy_image_path = "dummy_retinal_image.png"
    dummy_img_data = (np.random.rand(*IMAGE_SIZE, 3) * 255).astype(np.uint8)
    cv2.imwrite(dummy_image_path, dummy_img_data)
    print(f"Created dummy image for prediction at {dummy_image_path}")

    prediction_result = predict_retinopathy(loaded_model, dummy_image_path)
    if prediction_result:
        print(f"Prediction for '{dummy_image_path}':")
        print(f"  Class ID: {prediction_result['predicted_class_id']}")
        print(f"  Label: {prediction_result['predicted_label']}")
        print(f"  Confidence: {prediction_result['confidence']:.2f}")
    
    # Clean up dummy image
    os.remove(dummy_image_path)
    print(f"Removed dummy image: {dummy_image_path}")
