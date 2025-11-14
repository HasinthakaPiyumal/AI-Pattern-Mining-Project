import tensorflow as tf
import numpy as np

def create_dummy_medical_image_model(input_shape=(128, 128, 3), num_classes=3):
    """
    Creates a dummy convolutional neural network (CNN) for medical image classification.
    This model is simplified for demonstration purposes and not intended for real medical use.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def preprocess_image(image_path):
    """
    Dummy function to simulate image preprocessing.
    In a real application, this would load and resize images.
    """
    # Simulate loading and preprocessing an image to a fixed size
    dummy_image = np.random.rand(128, 128, 3).astype(np.float32)
    return np.expand_dims(dummy_image, axis=0) # Add batch dimension

def get_model_prediction(model, preprocessed_image):
    """
    Get predictions from the model.
    """
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predictions, predicted_class

# Example usage (for testing purposes)
if __name__ == "__main__":
    dummy_model = create_dummy_medical_image_model()
    dummy_model.summary()

    # Simulate an input image
    sample_image_input = preprocess_image("dummy_path.png")
    
    predictions, predicted_class = get_model_prediction(dummy_model, sample_image_input)
    print(f"Sample image shape: {sample_image_input.shape}")
    print(f"Model predictions: {predictions}")
    print(f"Predicted class index: {predicted_class}")
    class_names = ["No Disease", "Condition A", "Condition B"]
    print(f"Predicted class: {class_names[predicted_class]}")