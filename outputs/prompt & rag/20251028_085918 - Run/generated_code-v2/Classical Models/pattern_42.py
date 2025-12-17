import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# 1. Data Loading & Preprocessing
# Create dummy directories and files for demonstration if they don't exist
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

for d in [train_dir, validation_dir, test_dir]:
    os.makedirs(os.path.join(d, 'disease_A'), exist_ok=True)
    os.makedirs(os.path.join(d, 'disease_B'), exist_ok=True)

# Create some dummy images
def create_dummy_image(path, size=(128, 128), color=(255, 0, 0)):
    img = Image.new('RGB', size, color)
    img.save(path)

if not os.path.exists(os.path.join(train_dir, 'disease_A', 'img1.png')):
    create_dummy_image(os.path.join(train_dir, 'disease_A', 'img1.png'), color=(255, 0, 0))
    create_dummy_image(os.path.join(train_dir, 'disease_A', 'img2.png'), color=(200, 50, 50))
    create_dummy_image(os.path.join(train_dir, 'disease_B', 'img3.png'), color=(0, 255, 0))
    create_dummy_image(os.path.join(train_dir, 'disease_B', 'img4.png'), color=(50, 200, 50))

    create_dummy_image(os.path.join(validation_dir, 'disease_A', 'val_img1.png'), color=(255, 0, 0))
    create_dummy_image(os.path.join(validation_dir, 'disease_B', 'val_img2.png'), color=(0, 255, 0))

    create_dummy_image(os.path.join(test_dir, 'disease_A', 'test_img1.png'), color=(255, 0, 0))
    create_dummy_image(os.path.join(test_dir, 'disease_B', 'test_img2.png'), color=(0, 255, 0))


img_width, img_height = 128, 128
batch_size = 32
num_classes = 2 # Assuming 'disease_A' and 'disease_B'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical' # For multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False # Keep data in order for evaluation
)

# 2. Model Definition (CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax') # num_classes for multi-class
])

# 3. Model Compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 4. Model Training
epochs = 10 # For demonstration, set a small number of epochs

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# 5. Model Evaluation
print("\nEvaluating model on test data...")
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Display training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 6. Prediction Function
def predict_image(model, image_path, target_size=(img_width, img_height)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    img_array = img_array / 255.0 # Rescale

    predictions = model.predict(img_array)
    return predictions

print("\nMaking a prediction on a sample image...")
sample_image_path = os.path.join(test_dir, 'disease_A', 'test_img1.png')
predictions = predict_image(model, sample_image_path)
predicted_class_index = np.argmax(predictions[0])

# Get class labels from the generator
class_labels = list(train_generator.class_indices.keys())
predicted_class_label = class_labels[predicted_class_index]

print(f"Predictions: {predictions}")
print(f"Predicted class for {sample_image_path}: {predicted_class_label}")

sample_image_path_b = os.path.join(test_dir, 'disease_B', 'test_img2.png')
predictions_b = predict_image(model, sample_image_path_b)
predicted_class_index_b = np.argmax(predictions_b[0])
predicted_class_label_b = class_labels[predicted_class_index_b]

print(f"Predictions: {predictions_b}")
print(f"Predicted class for {sample_image_path_b}: {predicted_class_label_b}")