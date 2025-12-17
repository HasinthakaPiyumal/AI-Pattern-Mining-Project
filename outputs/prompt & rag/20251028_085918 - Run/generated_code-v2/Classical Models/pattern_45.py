import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import joblib

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 70, num_samples),
        'BMI': np.random.uniform(18.0, 40.0, num_samples),
        'Glucose': np.random.randint(70, 200, num_samples),
        'BloodPressure': np.random.randint(60, 120, num_samples),
        'Insulin': np.random.randint(15, 300, num_samples),
        'Smoking': np.random.randint(0, 2, num_samples),
        'PhysicalActivity': np.random.uniform(0.1, 10.0, num_samples),
        'FamilyHistory': np.random.randint(0, 2, num_samples),
        'Outcome': np.random.randint(0, 2, num_samples) # 0 for no disease, 1 for disease
    }
    df = pd.DataFrame(data)
    # Introduce some correlation for 'Outcome'
    df['Outcome'] = ((df['Age'] > 50).astype(int) + 
                     (df['BMI'] > 30).astype(int) + 
                     (df['Glucose'] > 140).astype(int) + 
                     (df['Smoking'] == 1).astype(int) + 
                     (df['FamilyHistory'] == 1).astype(int) + 
                     np.random.randint(0, 2, num_samples)) > 2.5 # Threshold for disease
    df['Outcome'] = df['Outcome'].astype(int)
    return df

def train_model(dataframe):
    X = dataframe.drop('Outcome', axis=1)
    y = dataframe['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, solver='liblinear')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, scaler, X.columns.tolist()

def predict_disease(model, scaler, feature_names, patient_data):
    patient_df = pd.DataFrame([patient_data], columns=feature_names)
    patient_scaled = scaler.transform(patient_df)
    prediction_proba = model.predict_proba(patient_scaled)[0]
    prediction_class = model.predict(patient_scaled)[0]
    
    disease_status = "Positive (High Risk)" if prediction_class == 1 else "Negative (Low Risk)"
    return {
        "prediction_class": prediction_class,
        "disease_status": disease_status,
        "probability_negative": prediction_proba[0],
        "probability_positive": prediction_proba[1]
    }

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data(num_samples=2000)
    # print(df.head())
    # print(df['Outcome'].value_counts())

    print("\nTraining the disease prediction model...")
    trained_model, data_scaler, features = train_model(df)
    
    # Save the trained model and scaler
    joblib.dump(trained_model, 'disease_prediction_model.pkl')
    joblib.dump(data_scaler, 'feature_scaler.pkl')
    joblib.dump(features, 'model_features.pkl')
    print("\nModel, scaler, and features saved.")

    # Example usage for prediction
    print("\nLoading saved model for prediction...")
    loaded_model = joblib.load('disease_prediction_model.pkl')
    loaded_scaler = joblib.load('feature_scaler.pkl')
    loaded_features = joblib.load('model_features.pkl')

    sample_patient_data = {
        'Age': 55,
        'BMI': 32.5,
        'Glucose': 160,
        'BloodPressure': 90,
        'Insulin': 150,
        'Smoking': 1,
        'PhysicalActivity': 3.5,
        'FamilyHistory': 1
    }

    print(f"\nPredicting for sample patient data: {sample_patient_data}")
    prediction_result = predict_disease(loaded_model, loaded_scaler, loaded_features, sample_patient_data)
    print(f"Prediction Result: {prediction_result['disease_status']}")
    print(f"Probability of No Disease: {prediction_result['probability_negative']:.4f}")
    print(f"Probability of Disease: {prediction_result['probability_positive']:.4f}")

    sample_patient_data_low_risk = {
        'Age': 30,
        'BMI': 22.0,
        'Glucose': 85,
        'BloodPressure': 70,
        'Insulin': 50,
        'Smoking': 0,
        'PhysicalActivity': 8.0,
        'FamilyHistory': 0
    }
    print(f"\nPredicting for sample patient data (low risk): {sample_patient_data_low_risk}")
    prediction_result_low_risk = predict_disease(loaded_model, loaded_scaler, loaded_features, sample_patient_data_low_risk)
    print(f"Prediction Result: {prediction_result_low_risk['disease_status']}")
    print(f"Probability of No Disease: {prediction_result_low_risk['probability_negative']:.4f}")
    print(f"Probability of Disease: {prediction_result_low_risk['probability_positive']:.4f}")
