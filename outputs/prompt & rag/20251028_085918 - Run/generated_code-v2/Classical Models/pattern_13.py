import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, GRU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib
import os

# --- 1. Data Ingestion & Preprocessing Layer ---

def generate_dummy_structured_data(num_samples=1000):
    data = {
        'age': np.random.randint(20, 80, num_samples),
        'gender': np.random.choice(['M', 'F'], num_samples),
        'cholesterol': np.random.randint(150, 250, num_samples),
        'blood_pressure_systolic': np.random.randint(100, 180, num_samples),
        'blood_pressure_diastolic': np.random.randint(60, 120, num_samples),
        'smoking': np.random.choice([0, 1], num_samples),
        'disease_status': np.random.choice([0, 1], num_samples, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    return df

def preprocess_structured_data(df):
    X = df.drop('disease_status', axis=1)
    y = df['disease_status']

    categorical_features = ['gender']
    numerical_features = [col for col in X.columns if col not in categorical_features]

    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(X[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features), index=X.index)
    
    X_processed = pd.concat([X[numerical_features], encoded_df], axis=1)
    return X_processed, y, scaler, encoder

def generate_dummy_image_data(num_samples=100, img_size=(64, 64), channels=3):
    images = np.random.rand(num_samples, img_size[0], img_size[1], channels).astype(np.float32)
    labels = np.random.randint(0, 2, num_samples) # 0 for healthy, 1 for diseased
    return images, labels

def preprocess_image_data(images):
    # Simple normalization to [0, 1]
    return images / 255.0

def generate_dummy_time_series_data(num_samples=100, seq_length=50, num_features=5):
    data = np.random.rand(num_samples, seq_length, num_features).astype(np.float32)
    labels = np.random.randint(0, 2, num_samples)
    return data, labels

def preprocess_time_series_data(data):
    # Simple Min-Max scaling per feature across all samples and timesteps
    scaler = MinMaxScaler()
    num_samples, seq_length, num_features = data.shape
    reshaped_data = data.reshape(-1, num_features)
    scaled_reshaped_data = scaler.fit_transform(reshaped_data)
    scaled_data = scaled_reshaped_data.reshape(num_samples, seq_length, num_features)
    return scaled_data, scaler

# --- 2. Feature Engineering Layer (for Classical Models) ---

def engineer_structured_features(X_processed):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_processed)
    return X_poly, poly

def engineer_time_series_features(data, window_size=5):
    # Example: Adding rolling mean as a feature for each time series
    engineered_features = []
    for sample in data:
        df_sample = pd.DataFrame(sample)
        rolling_means = df_sample.rolling(window=window_size).mean().fillna(method='bfill')
        engineered_sample = np.concatenate((sample, rolling_means.values), axis=1)
        engineered_features.append(engineered_sample)
    return np.array(engineered_features)

# --- 3. Model Training Layer ---

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_cnn_model(X_train, y_train, input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    return model

def train_lstm_model(X_train, y_train, input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    return model

# --- 4. Model Evaluation & Validation Layer ---

def evaluate_model(model, X_test, y_test, model_type='classical'):
    if model_type == 'classical':
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    elif model_type == 'deep_learning':
        y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()
        y_proba = model.predict(X_test).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc_score': roc_auc
    }
    return metrics

# --- 5. Prediction & Inference Layer ---

def save_model(model, filename):
    if isinstance(model, (LogisticRegression, SVC, RandomForestClassifier)):
        joblib.dump(model, filename)
    elif isinstance(model, tf.keras.Model):
        model.save(filename)

def load_model(filename, model_type='classical'):
    if model_type == 'classical':
        return joblib.load(filename)
    elif model_type == 'deep_learning':
        return tf.keras.models.load_model(filename)


if __name__ == "__main__":
    # Set a random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Structured Data Workflow ---
    print("\n--- Structured Data Workflow ---")
    structured_df = generate_dummy_structured_data(num_samples=1000)
    X_structured, y_structured, structured_scaler, structured_encoder = preprocess_structured_data(structured_df)
    X_structured_fe, structured_poly_features = engineer_structured_features(X_structured)

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_structured_fe, y_structured, test_size=0.2, random_state=42, stratify=y_structured)

    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train_s, y_train_s)
    lr_metrics = evaluate_model(lr_model, X_test_s, y_test_s, model_type='classical')
    print(f"Logistic Regression Metrics: {lr_metrics}")
    save_model(lr_model, "lr_model.joblib")

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train_s, y_train_s)
    rf_metrics = evaluate_model(rf_model, X_test_s, y_test_s, model_type='classical')
    print(f"Random Forest Metrics: {rf_metrics}")
    save_model(rf_model, "rf_model.joblib")

    # --- Image Data Workflow ---
    print("\n--- Image Data Workflow ---")
    images, image_labels = generate_dummy_image_data(num_samples=200)
    images_processed = preprocess_image_data(images)

    X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(images_processed, image_labels, test_size=0.2, random_state=42, stratify=image_labels)

    print("Training CNN for Image Data...")
    img_input_shape = X_train_img.shape[1:]
    cnn_model = train_cnn_model(X_train_img, y_train_img, img_input_shape)
    cnn_metrics = evaluate_model(cnn_model, X_test_img, y_test_img, model_type='deep_learning')
    print(f"CNN Metrics (Image): {cnn_metrics}")
    save_model(cnn_model, "cnn_model.h5")

    # --- Time-Series Data Workflow ---
    print("\n--- Time-Series Data Workflow ---")
    ts_data, ts_labels = generate_dummy_time_series_data(num_samples=150)
    ts_data_processed, ts_scaler = preprocess_time_series_data(ts_data)
    ts_data_fe = engineer_time_series_features(ts_data_processed)

    X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(ts_data_fe, ts_labels, test_size=0.2, random_state=42, stratify=ts_labels)
    
    print("Training LSTM for Time-Series Data...")
    ts_input_shape = X_train_ts.shape[1:]
    lstm_model = train_lstm_model(X_train_ts, y_train_ts, ts_input_shape)
    lstm_metrics = evaluate_model(lstm_model, X_test_ts, y_test_ts, model_type='deep_learning')
    print(f"LSTM Metrics (Time-Series): {lstm_metrics}")
    save_model(lstm_model, "lstm_model.h5")

    # --- Demonstrating Model Loading and Prediction ---
    print("\n--- Demonstrating Model Loading and Prediction ---")

    # Load and predict with a classical model
    loaded_lr_model = load_model("lr_model.joblib", model_type='classical')
    sample_structured_data = X_test_s.iloc[0:1]
    lr_prediction = loaded_lr_model.predict(sample_structured_data)
    print(f"Loaded LR model prediction for sample structured data: {lr_prediction}")

    # Load and predict with a deep learning model
    loaded_cnn_model = load_model("cnn_model.h5", model_type='deep_learning')
    sample_image_data = X_test_img[0:1]
    cnn_prediction = (loaded_cnn_model.predict(sample_image_data) > 0.5).astype("int32").flatten()
    print(f"Loaded CNN model prediction for sample image data: {cnn_prediction}")

    # Clean up generated model files
    os.remove("lr_model.joblib")
    os.remove("rf_model.joblib")
    os.remove("cnn_model.h5")
    os.remove("lstm_model.h5")
    print("\nCleaned up generated model files.")
