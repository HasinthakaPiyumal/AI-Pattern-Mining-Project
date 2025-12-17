
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# 1. Data Source Simulation
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(25, 80, num_samples),
        'Cholesterol': np.random.randint(150, 300, num_samples),
        'BloodPressure': np.random.randint(90, 180, num_samples),
        'HeartRate': np.random.randint(60, 100, num_samples),
        'Glucose': np.random.randint(70, 200, num_samples),
        'HeartDisease': np.random.randint(0, 2, num_samples) # 0 for no disease, 1 for disease
    }
    df = pd.DataFrame(data)
    # Introduce some correlation for HeartDisease
    df.loc[df['Age'] > 50, 'HeartDisease'] = np.random.randint(0, 2, df[df['Age'] > 50].shape[0], p=[0.3, 0.7])
    df.loc[df['Cholesterol'] > 220, 'HeartDisease'] = np.random.randint(0, 2, df[df['Cholesterol'] > 220].shape[0], p=[0.2, 0.8])
    df.loc[df['BloodPressure'] > 140, 'HeartDisease'] = np.random.randint(0, 2, df[df['BloodPressure'] > 140].shape[0], p=[0.25, 0.75])

    return df

# Main execution block
if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    print("Data generated. First 5 rows:")
    print(df.head())

    # 2. Data Preprocessing
    print("\nPreprocessing data...")
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    numerical_cols = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_cols)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # 3. Model Training
    print("\nTraining RandomForestClassifier model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Model Evaluation
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # 5. Model Persistence
    print("\nSaving trained model and scaler...")
    joblib.dump(model, 'random_forest_heart_disease_model.joblib')
    joblib.dump(scaler, 'standard_scaler_heart_disease.joblib')
    print("Model and scaler saved.")

    # 6. Prediction Service/Function Demonstration
    print("\nDemonstrating prediction service...")

    def predict_heart_disease_risk(patient_data: dict, model_path='random_forest_heart_disease_model.joblib', scaler_path='standard_scaler_heart_disease.joblib'):
        loaded_model = joblib.load(model_path)
        loaded_scaler = joblib.load(scaler_path)

        df_patient = pd.DataFrame([patient_data])
        patient_scaled = loaded_scaler.transform(df_patient)

        risk_prediction = loaded_model.predict(patient_scaled)[0]
        risk_probability = loaded_model.predict_proba(patient_scaled)[0][1]

        return {
            'prediction': 'High Risk' if risk_prediction == 1 else 'Low Risk',
            'probability': f'{risk_probability:.2f}'
        }

    # Example new patient data
    new_patient = {
        'Age': 60,
        'Cholesterol': 250,
        'BloodPressure': 150,
        'HeartRate': 75,
        'Glucose': 130
    }

    prediction_result = predict_heart_disease_risk(new_patient)
    print(f"New Patient Data: {new_patient}")
    print(f"Heart Disease Risk Prediction: {prediction_result['prediction']} (Probability: {prediction_result['probability']})")

    new_patient_low_risk = {
        'Age': 30,
        'Cholesterol': 180,
        'BloodPressure': 110,
        'HeartRate': 68,
        'Glucose': 85
    }
    prediction_result_low = predict_heart_disease_risk(new_patient_low_risk)
    print(f"\nNew Patient Data: {new_patient_low_risk}")
    print(f"Heart Disease Risk Prediction: {prediction_result_low['prediction']} (Probability: {prediction_result_low['probability']})")
