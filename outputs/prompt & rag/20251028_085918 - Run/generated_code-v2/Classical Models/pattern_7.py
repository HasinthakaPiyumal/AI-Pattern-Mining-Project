import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# --- 1. Simulate Data (as heart_disease_data.csv is not provided) ---
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 80, num_samples),
        'sex': np.random.randint(0, 2, num_samples), # 0 for female, 1 for male
        'cholesterol': np.random.randint(120, 300, num_samples),
        'blood_pressure': np.random.randint(90, 180, num_samples),
        'max_hr': np.random.randint(100, 200, num_samples),
        'exercise_angina': np.random.randint(0, 2, num_samples),
        'st_depression': np.random.uniform(0.0, 4.0, num_samples),
        'num_major_vessels': np.random.randint(0, 4, num_samples),
        'chest_pain_type': np.random.randint(0, 4, num_samples),
        'fasting_blood_sugar': np.random.randint(0, 2, num_samples), # > 120 mg/dl (1), <= 120 mg/dl (0)
        'target': np.random.randint(0, 2, num_samples) # 0 for no disease, 1 for disease
    }
    df = pd.DataFrame(data)
    # Introduce some correlation for 'target'
    df.loc[df['age'] > 60, 'target'] = np.random.choice([0, 1], size=df[df['age'] > 60].shape[0], p=[0.3, 0.7])
    df.loc[df['cholesterol'] > 240, 'target'] = np.random.choice([0, 1], size=df[df['cholesterol'] > 240].shape[0], p=[0.4, 0.6])
    df.loc[df['blood_pressure'] > 140, 'target'] = np.random.choice([0, 1], size=df[df['blood_pressure'] > 140].shape[0], p=[0.35, 0.65])
    return df

# Generate and save synthetic data
if not os.path.exists('heart_disease_data.csv'):
    print("Generating synthetic heart disease data...")
    synthetic_df = generate_synthetic_data()
    synthetic_df.to_csv('heart_disease_data.csv', index=False)
    print("Synthetic data saved to heart_disease_data.csv")
else:
    print("Using existing heart_disease_data.csv")

# --- 2. Data Loading and Preprocessing ---
print("\nLoading data...")
data = pd.read_csv('heart_disease_data.csv')

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Model Training ---
print("Training Logistic Regression model...")
model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is good for small datasets and binary classification
model.fit(X_train_scaled, y_train)

# --- 4. Model Evaluation ---
print("\nEvaluating model...")
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# --- 5. Model Serialization ---
print("\nSaving model and scaler...")
model_filename = 'logistic_regression_model.joblib'
scaler_filename = 'standard_scaler.joblib'

joblib.dump(model, model_filename)
joblib.dump(scaler, scaler_filename)
print(f"Model saved to {model_filename}")
print(f"Scaler saved to {scaler_filename}")

# --- 6. Prediction API/Function (Conceptual) Demonstration ---
print("\nDemonstrating prediction with the saved model...")

def predict_heart_disease_risk(patient_data: dict, model_path: str, scaler_path: str):
    try:
        loaded_model = joblib.load(model_path)
        loaded_scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        return {"error": "Model or scaler file not found. Please ensure they are saved correctly."}

    # Convert patient data to DataFrame in the same order as training features
    # Ensure the order of features matches the training data (X.columns)
    feature_order = X.columns.tolist() # Use original feature order from X
    patient_df = pd.DataFrame([patient_data], columns=feature_order)
    
    # Scale the input data
    patient_scaled = loaded_scaler.transform(patient_df)
    
    # Make prediction
    prediction = loaded_model.predict(patient_scaled)[0]
    prediction_proba = loaded_model.predict_proba(patient_scaled)[0].tolist()
    
    result = {
        "prediction": int(prediction), # 0: No Heart Disease, 1: Heart Disease
        "prediction_probability": {"No Heart Disease": prediction_proba[0], "Heart Disease": prediction_proba[1]}
    }
    return result

# Example new patient data (ensure features match the training data structure)
new_patient = {
    'age': 55,
    'sex': 1, # Male
    'cholesterol': 280,
    'blood_pressure': 150,
    'max_hr': 140,
    'exercise_angina': 1,
    'st_depression': 2.5,
    'num_major_vessels': 2,
    'chest_pain_type': 3,
    'fasting_blood_sugar': 1
}

prediction_result = predict_heart_disease_risk(new_patient, model_filename, scaler_filename)

print("New Patient Data:", new_patient)
print("Prediction Result:", prediction_result)
if prediction_result.get('prediction') == 1:
    print("Risk of Heart Disease: HIGH")
else:
    print("Risk of Heart Disease: LOW")
