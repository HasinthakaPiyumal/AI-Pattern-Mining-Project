import tensorflow as tf
import numpy as np
import os

# --- Configuration --- #
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2 # Assuming binary classification (No DR, DR) for simplicity

# Create a dummy dataset directory structure for demonstration
# In a real scenario, 'data_dir' would point to your actual dataset.
dummy_data_dir = 'dummy_diabetic_retinopathy_dataset'
if not os.path.exists(dummy_data_dir):
    os.makedirs(os.path.join(dummy_data_dir, 'no_dr'))
    os.makedirs(os.path.join(dummy_data_dir, 'dr'))
    # Create dummy files
    for i in range(50):
        tf.keras.utils.save_img(os.path.join(dummy_data_dir, 'no_dr', f'img_{i}.png'), np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8))
        tf.keras.utils.save_img(os.path.join(dummy_data_dir, 'dr', f'img_{i}.png'), np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8))

data_dir = dummy_data_dir # Replace with your actual dataset path

# --- Data Loading and Preprocessing --- #
def process_image(image, label):
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = tf.cast(image, tf.float32) / 255.0 # Normalize to [0, 1]
    return image, label

# Load datasets
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Apply preprocessing and optimize performance
train_ds = raw_train_ds.map(process_image).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = raw_val_ds.map(process_image).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# --- Model Definition --- #
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax') # Output layer for classification
])

# --- Model Compilation --- #
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# --- Model Training --- #
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --- Model Evaluation (on validation set as no explicit test set was created) --- #
# In a real application, you would typically have a separate test set.
print("\nEvaluating model on the validation set...")
loss, accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
