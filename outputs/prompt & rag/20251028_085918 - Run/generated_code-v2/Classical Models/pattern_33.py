import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 80, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples),
        'blood_pressure_systolic': np.random.randint(90, 180, num_samples),
        'blood_pressure_diastolic': np.random.randint(60, 120, num_samples),
        'cholesterol': np.random.randint(150, 300, num_samples),
        'glucose': np.random.randint(70, 200, num_samples),
        'bmi': np.random.uniform(18.0, 40.0, num_samples),
        'family_history_heart_disease': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'smoker': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        'diabetes': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]) # Target variable
    }
    df = pd.DataFrame(data)
    
    # Introduce some correlation for diabetes
    df.loc[df['glucose'] > 140, 'diabetes'] = 1
    df.loc[(df['age'] > 50) & (df['bmi'] > 30), 'diabetes'] = 1
    df.loc[(df['cholesterol'] > 240) & (df['family_history_heart_disease'] == 1), 'diabetes'] = 1
    
    # Ensure binary target
    df['diabetes'] = df['diabetes'].apply(lambda x: 1 if x > 0.5 else 0)

    return df


def train_and_evaluate_model(df):
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    categorical_features = ['gender']
    numerical_features = [
        'age', 'blood_pressure_systolic', 'blood_pressure_diastolic',
        'cholesterol', 'glucose', 'bmi', 'family_history_heart_disease', 'smoker'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    return model

def predict_new_patient(model, patient_data):
    # Ensure patient_data is a DataFrame with the same columns as training data
    patient_df = pd.DataFrame([patient_data])
    prediction = model.predict(patient_df)
    prediction_proba = model.predict_proba(patient_df)[:, 1]
    return prediction[0], prediction_proba[0]

if __name__ == "__main__":
    print("Generating synthetic patient data...")
    synthetic_df = generate_synthetic_data(num_samples=2000)
    print("Data generated successfully. First 5 rows:")
    print(synthetic_df.head())

    print("\nTraining and evaluating the disease prediction model...")
    trained_model = train_and_evaluate_model(synthetic_df)

    model_filename = 'disease_prediction_model.joblib'
    print(f"\nSaving the trained model to {model_filename}...")
    joblib.dump(trained_model, model_filename)
    print("Model saved successfully.")

    print(f"\nLoading the model from {model_filename}...")
    loaded_model = joblib.load(model_filename)
    print("Model loaded successfully.")

    print("\nDemonstrating prediction for a new patient...")
    new_patient_data = {
        'age': 65,
        'gender': 'Female',
        'blood_pressure_systolic': 150,
        'blood_pressure_diastolic': 95,
        'cholesterol': 280,
        'glucose': 180,
        'bmi': 32.5,
        'family_history_heart_disease': 1,
        'smoker': 0
    }
    
    prediction, probability = predict_new_patient(loaded_model, new_patient_data)
    
    disease_status = "likely to have diabetes" if prediction == 1 else "unlikely to have diabetes"
    print(f"New patient data: {new_patient_data}")
    print(f"Prediction: Patient is {disease_status} (Probability: {probability:.4f})")

    new_patient_data_healthy = {
        'age': 30,
        'gender': 'Male',
        'blood_pressure_systolic': 110,
        'blood_pressure_diastolic': 70,
        'cholesterol': 170,
        'glucose': 90,
        'bmi': 22.0,
        'family_history_heart_disease': 0,
        'smoker': 0
    }
    prediction_healthy, probability_healthy = predict_new_patient(loaded_model, new_patient_data_healthy)
    disease_status_healthy = "likely to have diabetes" if prediction_healthy == 1 else "unlikely to have diabetes"
    print(f"\nNew patient data (healthy example): {new_patient_data_healthy}")
    print(f"Prediction: Patient is {disease_status_healthy} (Probability: {probability_healthy:.4f})")