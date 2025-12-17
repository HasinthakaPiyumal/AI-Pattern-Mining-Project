import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# --- 1. Data Acquisition and Preparation Module ---

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(dataframe, scaler=None, fit_scaler=True):
    # Identify columns that often have '0' as missing values in diabetes datasets
    cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_impute:
        dataframe[col] = dataframe[col].replace(0, np.nan)
        dataframe[col] = dataframe[col].fillna(dataframe[col].mean())

    # Features are all columns except 'Outcome'
    X = dataframe.drop('Outcome', axis=1)
    y = dataframe['Outcome']

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided if fit_scaler is False")
        X_scaled = scaler.transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y, scaler

def split_data(features, target, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state, stratify=target)
    return X_train, X_test, y_train, y_test

# --- 2. Model Training Module ---

def train_model(model_type, X_train, y_train):
    if model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=42, solver='liblinear')
    elif model_type == 'SVC':
        model = SVC(random_state=42, probability=True)
    elif model_type == 'DecisionTree':
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

# --- 3. Model Persistence Module ---

def save_model(model, filename):
    joblib.dump(model, filename)
    
def load_model(filename):
    return joblib.load(filename)

# --- 4. Prediction Module ---

def predict_diabetes(model, new_data_scaled):
    return model.predict(new_data_scaled)

# --- Main Application Logic ---

if __name__ == "__main__":
    # Create a dummy diabetes.csv for demonstration if it doesn't exist
    # In a real scenario, you would have your actual dataset
    if not os.path.exists('diabetes.csv'):
        print("Creating a dummy 'diabetes.csv' for demonstration.")
        # Based on Pima Indians Diabetes Database structure
        data = {
            'Pregnancies': np.random.randint(0, 17, 100),
            'Glucose': np.random.randint(44, 199, 100),
            'BloodPressure': np.random.randint(24, 122, 100),
            'SkinThickness': np.random.randint(7, 99, 100),
            'Insulin': np.random.randint(14, 846, 100),
            'BMI': np.random.uniform(18.2, 67.1, 100),
            'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, 100),
            'Age': np.random.randint(21, 81, 100),
            'Outcome': np.random.randint(0, 2, 100) # 0 for no diabetes, 1 for diabetes
        }
        dummy_df = pd.DataFrame(data)
        dummy_df.loc[dummy_df['Glucose'] < 70, 'Glucose'] = np.random.randint(70, 199, len(dummy_df[dummy_df['Glucose'] < 70]))
        dummy_df.loc[dummy_df['BloodPressure'] < 50, 'BloodPressure'] = np.random.randint(50, 122, len(dummy_df[dummy_df['BloodPressure'] < 50]))
        dummy_df.loc[dummy_df['BMI'] < 18.5, 'BMI'] = np.random.uniform(18.5, 67.1, len(dummy_df[dummy_df['BMI'] < 18.5]))
        dummy_df.to_csv('diabetes.csv', index=False)
    
    data_filepath = 'diabetes.csv'
    print(f"\n--- Starting Diabetes Prediction System ---")

    # 1. Data Acquisition and Preparation
    print("Loading and preprocessing data...")
    df = load_data(data_filepath)
    X, y, scaler = preprocess_data(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Data split: Train samples = {len(X_train)}, Test samples = {len(X_test)}")

    # 2. Model Training and Evaluation
    models = {
        'LogisticRegression': None,
        'SVC': None,
        'DecisionTree': None
    }
    evaluations = {}
    best_model_name = ''
    best_accuracy = -1
    best_model = None

    print("\nTraining and evaluating models...")
    for model_name in models.keys():
        print(f"  Training {model_name}...")
        model = train_model(model_name, X_train, y_train)
        models[model_name] = model
        metrics = evaluate_model(model, X_test, y_test)
        evaluations[model_name] = metrics
        print(f"  {model_name} Metrics: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-score={metrics['f1_score']:.4f}")

        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_model_name = model_name
            best_model = model
    
    print(f"\nBest performing model: {best_model_name} with Accuracy = {best_accuracy:.4f}")

    # 3. Model Persistence
    model_filename = f'best_diabetes_model_{best_model_name}.joblib'
    scaler_filename = 'scaler.joblib'
    save_model(best_model, model_filename)
    save_model(scaler, scaler_filename)
    print(f"Best model saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")

    # Load the best model and scaler for demonstration
    loaded_model = load_model(model_filename)
    loaded_scaler = load_model(scaler_filename)
    print(f"Loaded model: {type(loaded_model).__name__}")

    # 4. Prediction Module - Demonstrate with new hypothetical data
    print("\nDemonstrating prediction with new hypothetical data...")
    # Example new patient data (ensure order of features is same as training data)
    # Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    new_patient_data = pd.DataFrame([[
        6, 148, 72, 35, 0, 33.6, 0.627, 50
    ]], columns=X.columns) # X.columns preserves the original feature order

    # Preprocess new data using the *trained* scaler (fit_scaler=False)
    # Note: `preprocess_data` expects a DataFrame with 'Outcome' column, but for new data
    # we only have features. So, we'll manually apply the scaler.
    
    # Handle '0' values in new data similar to training data imputation
    for col in cols_to_impute:
        if col in new_patient_data.columns:
            new_patient_data[col] = new_patient_data[col].replace(0, df[col].mean()) # Use original df's mean

    new_patient_data_scaled = loaded_scaler.transform(new_patient_data)
    
    prediction = predict_diabetes(loaded_model, new_patient_data_scaled)

    print(f"New patient data:\n{new_patient_data}")
    if prediction[0] == 1:
        print("Prediction: The patient is likely to have diabetes.")
    else:
        print("Prediction: The patient is likely NOT to have diabetes.")

    print("\n--- Diabetes Prediction System Finished ---")
