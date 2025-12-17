import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 5  # Example: No DR, Mild, Moderate, Severe, Proliferative
DATA_DIR = "data/retinal_images" # Placeholder: User needs to provide this directory
MODEL_SAVE_PATH = "best_dr_model.h5"

def create_generators(data_dir, img_height, img_width, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        data_dir + "/train",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir + "/train",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        data_dir + "/test",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return train_generator, validation_generator, test_generator

def build_cnn_model(img_height, img_width, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, train_generator, validation_generator, epochs, model_save_path):
    callbacks = [
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
    ]

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    return history

def evaluate_model(model_save_path, test_generator, class_labels):
    best_model = load_model(model_save_path)
    print("\nEvaluating the best model...")
    
    test_loss, test_accuracy = best_model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    predictions = best_model.predict(test_generator)
    y_pred_classes = np.argmax(predictions, axis=1)
    y_true_classes = test_generator.classes[test_generator.index_array]

    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print(cm)
    
def predict_single_image(image_path, model_save_path, img_height, img_width, class_labels):
    model = load_model(model_save_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]

    print(f"\nPrediction for {image_path}:")
    print(f"Predicted Class: {predicted_class_label}")
    print(f"Confidence: {confidence:.4f}")
    return predicted_class_label, confidence

if __name__ == "__main__":
    # Create dummy data directories for demonstration if they don't exist
    # In a real scenario, you would have actual images here.
    # Expected structure: data/retinal_images/train/class1, data/retinal_images/train/class2, etc.
    #                     data/retinal_images/test/class1, data/retinal_images/test/class2, etc.
    if not os.path.exists(DATA_DIR):
        print(f"Creating dummy data directory structure at {DATA_DIR}. Please replace with actual data.")
        os.makedirs(os.path.join(DATA_DIR, "train", "0_No_DR"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "train", "1_Mild_DR"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "test", "0_No_DR"), exist_ok=True)
        # Create a dummy image file for demonstration
        dummy_image_path = os.path.join(DATA_DIR, "train", "0_No_DR", "dummy.png")
        if not os.path.exists(dummy_image_path):
            dummy_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            cv2.imwrite(dummy_image_path, dummy_img)
            print(f"Created a dummy image at {dummy_image_path}")

    train_gen, val_gen, test_gen = create_generators(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
    class_labels = list(train_gen.class_indices.keys())
    print(f"Detected Class Labels: {class_labels}")

    model = build_cnn_model(IMG_HEIGHT, IMG_WIDTH, len(class_labels))
    model.summary()

    print("\nStarting model training...")
    history = train_model(model, train_gen, val_gen, EPOCHS, MODEL_SAVE_PATH)

    print("\nTraining complete. Evaluating on test set...")
    evaluate_model(MODEL_SAVE_PATH, test_gen, class_labels)

    # Example prediction on a dummy image
    if os.path.exists(dummy_image_path):
        predict_single_image(dummy_image_path, MODEL_SAVE_PATH, IMG_HEIGHT, IMG_WIDTH, class_labels)
    else:
        print("\nCannot perform single image prediction: Dummy image not found.")