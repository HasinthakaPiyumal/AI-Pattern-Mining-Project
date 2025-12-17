import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

# 1. Data Management: Simulate Data Loading
def load_data(filepath='simulated_patient_data.csv'):
    # In a real scenario, this would load from a CSV file.
    # For demonstration, let's create a synthetic dataset.
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 70, 100),
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'BMI': np.random.uniform(18, 35, 100),
        'BloodPressure_Systolic': np.random.randint(100, 180, 100),
        'Cholesterol': np.random.randint(150, 250, 100),
        'FamilyHistory_Diabetes': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        'Smoking': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
        'ExerciseFrequency': np.random.randint(0, 7, 100),
        'Disease': np.random.choice([0, 1], 100, p=[0.6, 0.4]) # Target variable
    }
    df = pd.DataFrame(data)
    
    # Introduce some missing values for demonstration
    for col in ['BMI', 'BloodPressure_Systolic', 'Cholesterol']:
        df.loc[df.sample(frac=0.05).index, col] = np.nan
    df.loc[df.sample(frac=0.02).index, 'Gender'] = np.nan

    print(f"Loaded {len(df)} samples.")
    return df

# 2. Data Preprocessing & Feature Engineering
def preprocess_data(df):
    # Define categorical and numerical features
    categorical_features = ['Gender']
    numerical_features = ['Age', 'BMI', 'BloodPressure_Systolic', 'Cholesterol', 'FamilyHistory_Diabetes', 'Smoking', 'ExerciseFrequency']

    # Preprocessing Pipelines
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
        ])
    
    # Separate features (X) and target (y)
    X = df.drop('Disease', axis=1)
    y = df['Disease']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training data size: {len(X_train)}, Testing data size: {len(X_test)}")
    return preprocessor, X_train, X_test, y_train, y_test

# 3. Model Selection & Training
def train_model(preprocessor, X_train, y_train):
    # Create a full pipeline including preprocessing and model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    model_pipeline.fit(X_train, y_train)
    print("Model trained successfully.")
    return model_pipeline

# 4. Model Evaluation
def evaluate_model(model_pipeline, X_test, y_test):
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1] # Probability of positive class

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("Confusion Matrix:\n", cm)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc_score': roc_auc,
        'confusion_matrix': cm.tolist()
    }

# 5. Prediction Interface
def predict_disease(model_pipeline, new_patient_data):
    # new_patient_data should be a pandas DataFrame with the same columns as the training data (excluding target)
    prediction = model_pipeline.predict(new_patient_data)
    prediction_proba = model_pipeline.predict_proba(new_patient_data)[:, 1]
    return prediction, prediction_proba

# Main execution flow
if __name__ == "__main__":
    # Load Data
    df = load_data()

    # Preprocess Data
    preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Train Model
    trained_model = train_model(preprocessor, X_train, y_train)

    # Evaluate Model
    evaluation_results = evaluate_model(trained_model, X_test, y_test)

    # Save the trained model and preprocessor for future use
    joblib.dump(trained_model, 'disease_prediction_model.joblib')
    print("\nModel and preprocessor saved as 'disease_prediction_model.joblib'")

    # --- Demonstrate Prediction with a new patient --- 
    print("\n--- Demonstrating Prediction on New Data ---")
    # Simulate new patient data (ensure columns match original dataset)
    new_patient = pd.DataFrame({
        'Age': [55],
        'Gender': ['Female'],
        'BMI': [28.5],
        'BloodPressure_Systolic': [145],
        'Cholesterol': [210],
        'FamilyHistory_Diabetes': [1],
        'Smoking': [0],
        'ExerciseFrequency': [2]
    })

    # Load the saved model for prediction (in a real app, this would be done separately)
    loaded_model = joblib.load('disease_prediction_model.joblib')
    
    predicted_class, predicted_proba = predict_disease(loaded_model, new_patient)

    print(f"New Patient Data:\n{new_patient.to_string(index=False)}")
    print(f"Predicted Disease Class (0: No, 1: Yes): {predicted_class[0]}")
    print(f"Predicted Probability of Disease: {predicted_proba[0]:.4f}")