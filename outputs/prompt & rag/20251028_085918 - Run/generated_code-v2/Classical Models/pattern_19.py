import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import shap
import joblib


# --- 1. Data Simulation ---
# Simulate structured patient data
np.random.seed(42)
num_patients = 1000
data = {
    "age": np.random.randint(20, 80, num_patients),
    "gender": np.random.choice(["Male", "Female"], num_patients),
    "bmi": np.random.normal(25, 5, num_patients),
    "blood_pressure_sys": np.random.randint(100, 180, num_patients),
    "blood_pressure_dia": np.random.randint(60, 110, num_patients),
    "cholesterol": np.random.normal(200, 40, num_patients),
    "glucose": np.random.normal(100, 20, num_patients),
    "smoker": np.random.choice([0, 1], num_patients, p=[0.7, 0.3]),
    "exercise": np.random.choice([0, 1], num_patients, p=[0.4, 0.6]),
    "family_history": np.random.choice([0, 1], num_patients, p=[0.6, 0.4]),
    "symptom_fatigue": np.random.choice([0, 1], num_patients, p=[0.5, 0.5]),
    "symptom_chest_pain": np.random.choice([0, 1], num_patients, p=[0.8, 0.2]),
    "symptom_cough": np.random.choice([0, 1], num_patients, p=[0.6, 0.4]),
    "disease_present": np.random.choice([0, 1], num_patients, p=[0.7, 0.3]) # Target variable
}
patient_df = pd.DataFrame(data)

# Introduce some missing values (for demonstration of imputation)
for col in ["bmi", "cholesterol", "glucose"]:
    patient_df.loc[np.random.choice(patient_df.index, int(num_patients * 0.05), replace=False), col] = np.nan

print("--- Simulated Structured Patient Data ---")
print(patient_df.head())
print(patient_df.info())
print("\n")

# Simulate medical image data (dummy)
img_height, img_width, img_channels = 64, 64, 1 # Grayscale images
num_images = 500
image_data = np.random.rand(num_images, img_height, img_width, img_channels).astype(np.float32)
image_labels = np.random.randint(0, 2, num_images) # Binary classification for images

print("--- Simulated Medical Image Data Shape ---")
print(f"Image data shape: {image_data.shape}")
print(f"Image labels shape: {image_labels.shape}")
print("\n")

# --- 2. Structured Data Preprocessing ---
X = patient_df.drop("disease_present", axis=1)
y = patient_df["disease_present"]

X_train_struct, X_test_struct, y_train_struct, y_test_struct = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify numerical and categorical features
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include="object").columns.tolist()

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

print("--- Structured Data Preprocessing Setup ---")
print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)
print("\n")

# --- 3. Classical ML Model (RandomForestClassifier) ---
print("--- Training Classical ML Model (RandomForestClassifier) ---")

# Create a pipeline with preprocessing and the classifier
classical_model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
classical_model_pipeline.fit(X_train_struct, y_train_struct)

# Evaluate the model
y_pred_struct = classical_model_pipeline.predict(X_test_struct)
y_proba_struct = classical_model_pipeline.predict_proba(X_test_struct)[:, 1]

print("\nClassical Model Performance Report:")
print(classification_report(y_test_struct, y_pred_struct))
print(f"ROC AUC Score: {roc_auc_score(y_test_struct, y_proba_struct):.4f}")

# Save the classical model
joblib.dump(classical_model_pipeline, "classical_disease_predictor.pkl")
print("Classical model saved as classical_disease_predictor.pkl")
print("\n")

# --- 4. Deep Learning Model (CNN for Medical Image Data) ---
print("--- Training Deep Learning Model (CNN for Image Data) ---")

# Split image data
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(image_data, image_labels, test_size=0.2, random_state=42, stratify=image_labels)

# Convert labels to one-hot encoding
y_train_img_onehot = to_categorical(y_train_img, num_classes=2)
y_test_img_onehot = to_categorical(y_test_img, num_classes=2)

# Define a simple CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, img_channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(2, activation="softmax") # 2 classes for binary classification
])

cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("CNN Model Summary:")
cnn_model.summary()

# Train the CNN model (using dummy data, minimal epochs for speed)
history = cnn_model.fit(X_train_img, y_train_img_onehot, epochs=5, batch_size=32, validation_split=0.1, verbose=0)

# Evaluate the CNN model
loss, accuracy = cnn_model.evaluate(X_test_img, y_test_img_onehot, verbose=0)
print(f"\nCNN Model Test Accuracy: {accuracy:.4f}")

# Save the CNN model
cnn_model.save("cnn_image_predictor.h5")
print("CNN model saved as cnn_image_predictor.h5")
print("\n")

# --- 5. Prediction and Inference Layer Demonstration ---
print("--- Prediction and Inference Demonstration ---")

# Simulate new patient data for prediction
new_patient_data = pd.DataFrame({
    "age": [55, 30],
    "gender": ["Male", "Female"],
    "bmi": [31.5, 22.1],
    "blood_pressure_sys": [145, 110],
    "blood_pressure_dia": [90, 70],
    "cholesterol": [230.0, 180.0],
    "glucose": [115.0, 95.0],
    "smoker": [1, 0],
    "exercise": [0, 1],
    "family_history": [1, 0],
    "symptom_fatigue": [1, 0],
    "symptom_chest_pain": [1, 0],
    "symptom_cough": [0, 1],
})

# Load the classical model
loaded_classical_model = joblib.load("classical_disease_predictor.pkl")
classical_predictions = loaded_classical_model.predict(new_patient_data)
classical_probabilities = loaded_classical_model.predict_proba(new_patient_data)[:, 1]

print("New Patient Structured Data:")
print(new_patient_data)
print("Disease Prediction (Classical Model):", classical_predictions)
print("Disease Probability (Classical Model):", [f"{p:.4f}" for p in classical_probabilities])
print("\n")

# Simulate new image data for prediction
new_image_data = np.random.rand(2, img_height, img_width, img_channels).astype(np.float32)

# Load the CNN model
loaded_cnn_model = tf.keras.models.load_model("cnn_image_predictor.h5")
cnn_predictions = loaded_cnn_model.predict(new_image_data)
cnn_predicted_classes = np.argmax(cnn_predictions, axis=1)

print(f"New Image Data (shape): {new_image_data.shape}")
print("Image Disease Probabilities (CNN Model):", [f"Class 0: {p[0]:.4f}, Class 1: {p[1]:.4f}" for p in cnn_predictions])
print("Image Disease Predicted Classes (CNN Model):", cnn_predicted_classes)
print("\n")

# --- 6. Interpretability (SHAP for Classical Model) ---
print("--- Interpretability with SHAP (for Classical Model) ---")

# For SHAP, we need the preprocessed data and the underlying classifier
# It's better to get the preprocessed data directly from the pipeline for SHAP
X_train_processed = classical_model_pipeline.named_steps["preprocessor"].transform(X_train_struct)

# Get feature names after one-hot encoding
onehot_features = classical_model_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols)
all_features = numerical_cols + list(onehot_features)

explainer = shap.TreeExplainer(classical_model_pipeline.named_steps["classifier"])
shap_values = explainer.shap_values(X_train_processed[:50]) # Explain first 50 samples

print(f"SHAP values shape for class 0: {shap_values[0].shape}")
print(f"SHAP values shape for class 1: {shap_values[1].shape}")
print("\nExample SHAP values for the first prediction (disease_present=1):")
# For binary classification, shap_values[1] typically represents the positive class
example_shap = pd.DataFrame(shap_values[1][0], index=all_features, columns=["SHAP Value"])
print(example_shap.sort_values(by="SHAP Value", ascending=False).head(10))

print("\nThis code demonstrates the full workflow for a predictive healthcare analytics system combining classical ML for structured data and a placeholder CNN for image data, including data simulation, preprocessing, model training, evaluation, saving, inference, and a basic interpretability example.")
