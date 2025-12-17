import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# --- 1. Data Generation and Preprocessing Module ---

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'Fever': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'Cough': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
        'Fatigue': np.random.choice([0, 1], num_samples, p=[0.5, 0.5]),
        'BloodPressure': np.random.normal(120, 15, num_samples),
        'Cholesterol': np.random.normal(200, 30, num_samples),
        'Diagnosis': np.random.choice(['Healthy', 'Disease_A', 'Disease_B'], num_samples, p=[0.6, 0.25, 0.15])
    }
    df = pd.DataFrame(data)

    # Introduce some missing values for demonstration
    for col in ['BloodPressure', 'Cholesterol']:
        df.loc[df.sample(frac=0.05).index, col] = np.nan
    df.loc[df.sample(frac=0.02).index, 'Gender'] = np.nan

    return df

def preprocess_data(df):
    # Define categorical and numerical features
    categorical_features = ['Gender'] # Other binary features like Fever, Cough, Fatigue are already numerical (0/1)
    numerical_features = ['Age', 'BloodPressure', 'Cholesterol']

    # Preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough' # Keep other columns (like Fever, Cough, Fatigue)
    )

    return preprocessor

# --- 2. Model Training Module ---

def train_model(X_train, y_train, preprocessor):
    # Combine preprocessor and classifier in a pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model_pipeline.fit(X_train, y_train)
    return model_pipeline

# --- 3. Model Evaluation Module ---

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# --- 4. Prediction Interface Module ---

def predict_diagnosis(new_patient_data, model_path='disease_diagnosis_model.joblib'):
    if not os.path.exists(model_path):
        return "Error: Model not found. Please train the model first."

    model = joblib.load(model_path)

    # Ensure new_patient_data is a DataFrame with the same columns as training data
    if isinstance(new_patient_data, dict):
        new_patient_df = pd.DataFrame([new_patient_data])
    elif isinstance(new_patient_data, pd.DataFrame):
        new_patient_df = new_patient_data
    else:
        return "Error: Invalid input format. Expected dict or pandas DataFrame."

    # Make prediction
    prediction = model.predict(new_patient_df)
    prediction_proba = model.predict_proba(new_patient_df)

    # Get class labels from the model's classes_
    class_labels = model.classes_ # Access classes_ from the final estimator in the pipeline

    predicted_disease = prediction[0]
    probabilities = dict(zip(class_labels, prediction_proba[0]))

    return {"predicted_disease": predicted_disease, "probabilities": probabilities}

# --- Main Workflow (simulating train_model.py and predict_diagnosis.py) ---
if __name__ == "__main__":
    # --- Training Phase ---
    print("\n--- Training Phase ---")
    df = generate_synthetic_data(num_samples=2000)

    # Separate features (X) and target (y)
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and fit the preprocessor
    from sklearn.impute import SimpleImputer # Moved here for local scope or ensuring it's available
    preprocessor = preprocess_data(X_train) # Pass X_train to infer feature types for preprocessor setup
    
    # Train the model
    print("Training the RandomForestClassifier...")
    trained_model = train_model(X_train, y_train, preprocessor)
    print("Model training complete.")

    # Evaluate the model
    print("\n--- Model Evaluation ---")
    evaluate_model(trained_model, X_test, y_test)

    # Save the trained model (which includes the preprocessor)
    model_filename = 'disease_diagnosis_model.joblib'
    joblib.dump(trained_model, model_filename)
    print(f"\nModel saved as {model_filename}")

    # --- Prediction Phase ---
    print("\n--- Prediction Phase ---")
    # Simulate new patient data
    new_patient_data_example = {
        'Age': 55,
        'Gender': 'Female',
        'Fever': 1,
        'Cough': 1,
        'Fatigue': 0,
        'BloodPressure': 145,
        'Cholesterol': 230
    }

    print("\nMaking prediction for a new patient:")
    print(new_patient_data_example)
    prediction_result = predict_diagnosis(new_patient_data_example, model_path=model_filename)
    print("Prediction Result:", prediction_result)

    new_patient_data_healthy = {
        'Age': 30,
        'Gender': 'Male',
        'Fever': 0,
        'Cough': 0,
        'Fatigue': 0,
        'BloodPressure': 110,
        'Cholesterol': 170
    }
    print("\nMaking prediction for another patient (likely healthy):")
    print(new_patient_data_healthy)
    prediction_result_healthy = predict_diagnosis(new_patient_data_healthy, model_path=model_filename)
    print("Prediction Result:", prediction_result_healthy)