import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import cv2

# --- Structured Data Processing and Prediction (Classical ML Component) ---

def generate_structured_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 80, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples),
        'blood_pressure_systolic': np.random.randint(100, 180, num_samples),
        'blood_pressure_diastolic': np.random.randint(60, 120, num_samples),
        'cholesterol': np.random.randint(150, 300, num_samples),
        'glucose': np.random.randint(70, 200, num_samples),
        'smoker': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'disease_outcome': np.random.randint(0, 2, num_samples, p=[0.8, 0.2])
    }
    df = pd.DataFrame(data)
    # Introduce some correlation for disease_outcome
    df.loc[df['cholesterol'] > 250, 'disease_outcome'] = 1
    df.loc[df['blood_pressure_systolic'] > 140, 'disease_outcome'] = 1
    return df

def train_classical_model(df):
    X = df.drop('disease_outcome', axis=1)
    y = df['disease_outcome']

    numerical_features = ['age', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'cholesterol', 'glucose']
    categorical_features = ['gender', 'smoker']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    print("\n--- Classical ML Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model_pipeline

def predict_structured_data(model, new_data_df):
    prediction = model.predict(new_data_df)
    prediction_proba = model.predict_proba(new_data_df)[:, 1]
    return prediction, prediction_proba

# --- Medical Image Processing and Prediction (Deep Learning Component - CNN) ---

def generate_image_data(num_samples=100, img_size=(64, 64), num_classes=2):
    np.random.seed(42)
    images = np.random.rand(num_samples, img_size[0], img_size[1], 1) * 255 # Grayscale images
    labels = np.random.randint(0, num_classes, num_samples)

    # Introduce some 'pattern' for a class for demonstration
    for i in range(num_samples // 2):
        images[i, 10:20, 10:20, :] = 0 # Dark square for class 0
    for i in range(num_samples // 2, num_samples):
        images[i, 30:40, 30:40, :] = 255 # Bright square for class 1

    return images.astype(np.uint8), labels

def preprocess_image_data(images, labels, img_size=(64, 64), num_classes=2):
    processed_images = []
    for img in images:
        resized_img = cv2.resize(img, img_size)
        processed_images.append(resized_img)
    processed_images = np.array(processed_images).astype('float32') / 255.0
    processed_images = np.expand_dims(processed_images, axis=-1) # Add channel dimension for grayscale
    encoded_labels = to_categorical(labels, num_classes=num_classes)
    return processed_images, encoded_labels

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_cnn_model(images, labels, input_shape, num_classes):
    X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = build_cnn_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("\n--- CNN Model Training ---")
    model.fit(X_train_img, y_train_img, epochs=5, batch_size=32, validation_split=0.1, verbose=0)

    loss, accuracy = model.evaluate(X_test_img, y_test_img, verbose=0)
    print(f"CNN Model Test Accuracy: {accuracy:.4f}")
    return model

def predict_image_data(model, new_image_array):
    processed_image = np.array(cv2.resize(new_image_array, (64, 64))).astype('float32') / 255.0
    processed_image = np.expand_dims(processed_image, axis=0) # Add batch dimension
    processed_image = np.expand_dims(processed_image, axis=-1) # Add channel dimension
    prediction = model.predict(processed_image)
    return np.argmax(prediction, axis=1)[0], np.max(prediction)

# --- Main Execution --- 
if __name__ == "__main__":
    print("Starting Predictive Disease Diagnosis System Demonstration...")

    # 1. Classical ML Component for Structured Data
    print("\n=== Classical Machine Learning Component ===")
    structured_data_df = generate_structured_data(num_samples=1000)
    classical_model = train_classical_model(structured_data_df)

    # Demonstrate prediction with new structured data
    print("\nDemonstrating prediction with new structured data:")
    new_patient_data = pd.DataFrame([{
        'age': 55,
        'gender': 'Female',
        'blood_pressure_systolic': 135,
        'blood_pressure_diastolic': 85,
        'cholesterol': 260,
        'glucose': 110,
        'smoker': 0
    }])
    
    prediction_class, prediction_prob = predict_structured_data(classical_model, new_patient_data)
    print(f"New patient diagnosis (0=Healthy, 1=Disease): {prediction_class[0]}")
    print(f"Probability of disease: {prediction_prob[0]:.4f}")

    # 2. Deep Learning Component (CNN) for Medical Image Data
    print("\n=== Deep Learning (CNN) Component ===")
    IMG_SIZE = (64, 64)
    NUM_CLASSES = 2
    images, labels = generate_image_data(num_samples=200, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    processed_images, encoded_labels = preprocess_image_data(images, labels, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)
    cnn_model = train_cnn_model(processed_images, encoded_labels, input_shape, NUM_CLASSES)

    # Demonstrate prediction with a new image
    print("\nDemonstrating prediction with a new image:")
    # Create a dummy new image (e.g., all white, or with a specific pattern)
    new_image_example = np.random.rand(IMG_SIZE[0], IMG_SIZE[1]) * 255 # Random image
    
    # Make it look like class 1 pattern for demonstration (bright square)
    new_image_example[30:40, 30:40] = 255

    pred_class_img, pred_prob_img = predict_image_data(cnn_model, new_image_example)
    print(f"New image diagnosis (0=Class0, 1=Class1): {pred_class_img}")
    print(f"Probability for predicted class: {pred_prob_img:.4f}")

    print("\nPredictive Disease Diagnosis System Demonstration Complete.")
