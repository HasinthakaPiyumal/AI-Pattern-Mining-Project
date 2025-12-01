import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- 1. Data Generation and Model Training (Simulated) ---
def generate_synthetic_data(num_samples=100):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'Fever': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
        'Cough': np.random.choice([0, 1], num_samples, p=[0.5, 0.5]),
        'Fatigue': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'BP_Systolic': np.random.randint(90, 180, num_samples),
        'BP_Diastolic': np.random.randint(60, 120, num_samples),
        'Cholesterol': np.random.randint(150, 280, num_samples),
        'Glucose': np.random.randint(70, 200, num_samples),
    }
    df = pd.DataFrame(data)

    # Simulate a target diagnosis (e.g., 'Hypertension', 'Flu', 'Healthy')
    conditions = [
        (df['BP_Systolic'] > 140) | (df['BP_Diastolic'] > 90),
        (df['Fever'] == 1) & (df['Cough'] == 1),
        (df['Age'] > 60) & (df['Cholesterol'] > 240) & (df['BP_Systolic'] > 130)
    ]
    choices = ['Hypertension', 'Flu', 'Cardiac Risk']
    df['Diagnosis'] = np.select(conditions, choices, default='Healthy')
    return df

patient_data_raw = generate_synthetic_data(500)

# Preprocessing for the model
le_gender = LabelEncoder()
patient_data_raw['Gender_Encoded'] = le_gender.fit_transform(patient_data_raw['Gender'])

le_diagnosis = LabelEncoder()
patient_data_raw['Diagnosis_Encoded'] = le_diagnosis.fit_transform(patient_data_raw['Diagnosis'])

features = ['Age', 'Gender_Encoded', 'Fever', 'Cough', 'Fatigue', 'BP_Systolic', 'BP_Diastolic', 'Cholesterol', 'Glucose']
X = patient_data_raw[features]
y = patient_data_raw['Diagnosis_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simulated black-box model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 2. Explanation Module (LACE-like Simulation) ---
class LaceExplainer:
    def __init__(self, model, feature_names, class_names):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names

    def explain_instance(self, instance_df):
        prediction_proba = self.model.predict_proba(instance_df)[0]
        predicted_class_idx = np.argmax(prediction_proba)
        predicted_class_name = self.class_names[predicted_class_idx]
        confidence = prediction_proba[predicted_class_idx]

        # Simplified feature importance based on model's global feature importances
        # and instance values for a LACE-like feel
        feature_importances = self.model.feature_importances_
        feature_contributions = {}
        explanation_text = f