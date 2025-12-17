import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def prepare_data(df):
    numerical_cols = ['Age', 'BMI', 'BloodPressure', 'Cholesterol', 'Glucose']
    categorical_cols = ['Gender']

    # Handle missing values (simple imputation for demonstration)
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].mean())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

    # Combine preprocessed features
    X = pd.concat([df[numerical_cols], encoded_df], axis=1)
    y = df['Disease']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler, encoder

def train_model(X_train, y_train, X_test, y_test):
    # Initialize and train Logistic Regression model
    model = LogisticRegression(random_state=42, solver='liblinear')
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)
    
    return model

def make_inference(model, scaler, encoder, new_patient_data):
    numerical_cols = ['Age', 'BMI', 'BloodPressure', 'Cholesterol', 'Glucose']
    categorical_cols = ['Gender']
    
    # Preprocess new data using the fitted scaler and encoder
    new_patient_df = pd.DataFrame([new_patient_data])
    
    new_patient_df[numerical_cols] = scaler.transform(new_patient_df[numerical_cols])
    
    encoded_features = encoder.transform(new_patient_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    
    X_new = pd.concat([new_patient_df[numerical_cols], encoded_df], axis=1)
    
    # Make prediction
    prediction = model.predict(X_new)
    prediction_proba = model.predict_proba(X_new)[:, 1] # Probability of disease (class 1)
    
    return prediction[0], prediction_proba[0]

if __name__ == '__main__':
    # 1. Simulate Data Ingestion (dummy data)
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 70, 100),
        'Gender': np.random.choice(['M', 'F'], 100),
        'BMI': np.random.uniform(18.0, 35.0, 100),
        'BloodPressure': np.random.randint(90, 180, 100),
        'Cholesterol': np.random.randint(150, 250, 100),
        'Glucose': np.random.randint(70, 150, 100),
        'Disease': np.random.randint(0, 2, 100) # 0: No Disease, 1: Disease
    }
    df = pd.DataFrame(data)

    print("--- Data Preparation ---")
    X_train, X_test, y_train, y_test, scaler, encoder = prepare_data(df)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Save scaler and encoder for inference
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoder, 'encoder.pkl')

    print("\n--- Model Training ---")
    model = train_model(X_train, y_train, X_test, y_test)
    # Save the trained model
    joblib.dump(model, 'logistic_regression_model.pkl')
    print("Model and preprocessors saved.")

    print("\n--- Model Inference ---")
    # Load the saved model and preprocessors for inference
    loaded_model = joblib.load('logistic_regression_model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_encoder = joblib.load('encoder.pkl')

    # Example new patient data for inference
    new_patient_data = {
        'Age': 55,
        'Gender': 'F',
        'BMI': 28.5,
        'BloodPressure': 130,
        'Cholesterol': 210,
        'Glucose': 110
    }
    
    prediction, probability = make_inference(loaded_model, loaded_scaler, loaded_encoder, new_patient_data)
    print(f"\nNew Patient Data: {new_patient_data}")
    print(f"Predicted Disease: {'Present' if prediction == 1 else 'Absent'}")
    print(f"Probability of Disease: {probability:.4f}")