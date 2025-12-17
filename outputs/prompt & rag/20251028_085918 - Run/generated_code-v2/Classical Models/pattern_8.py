import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib
import numpy as np

# --- 1. Data Ingestion and Preprocessing ---

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    
    # Separate target variable
    X = data.drop('Disease', axis=1)
    y = data['Disease']
    
    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
    # Create preprocessor pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features

# --- 2. Model Training ---

def train_model(X_train, y_train, preprocessor, model_type='logistic_regression'):
    if model_type == 'logistic_regression':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='liblinear', random_state=42))
        ])
    elif model_type == 'random_forest':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
    else:
        raise ValueError("model_type must be 'logistic_regression' or 'random_forest'")
        
    model.fit(X_train, y_train)
    return model

# --- 3. Model Evaluation ---

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of positive class
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    evaluation_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    return evaluation_metrics

# --- 4. Prediction Interface ---

def predict_disease_likelihood(model, patient_data_df):
    # patient_data_df should be a DataFrame with the same columns as the training data
    prediction_proba = model.predict_proba(patient_data_df)[:, 1]
    return prediction_proba

# --- 5. Model Persistence ---

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

# --- Example Usage (simulated data) ---
if __name__ == "__main__":
    # Create a dummy dataset for demonstration
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 70, 100),
        'BMI': np.random.uniform(18, 35, 100),
        'Glucose': np.random.uniform(70, 200, 100),
        'BloodPressure': np.random.uniform(80, 140, 100),
        'Smoking': np.random.choice(['Yes', 'No'], 100),
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'Disease': np.random.choice([0, 1], 100, p=[0.7, 0.3]) # 0: No Disease, 1: Disease
    }
    df = pd.DataFrame(data)
    df.to_csv('medical_data.csv', index=False)
    
    print("--- Data Loading and Preprocessing ---")
    X_train, X_test, y_train, y_test, preprocessor, num_features, cat_features = load_and_preprocess_data('medical_data.csv')
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    print(f"Numerical Features: {num_features}")
    print(f"Categorical Features: {cat_features}")
    
    print("\n--- Training Logistic Regression Model ---")
    lr_model = train_model(X_train, y_train, preprocessor, model_type='logistic_regression')
    print("Logistic Regression Model Trained.")
    
    print("\n--- Evaluating Logistic Regression Model ---")
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    for metric, value in lr_metrics.items():
        print(f"Logistic Regression {metric.replace('_', ' ').title()}: {value:.4f}")
        
    print("\n--- Training Random Forest Model ---")
    rf_model = train_model(X_train, y_train, preprocessor, model_type='random_forest')
    print("Random Forest Model Trained.")
    
    print("\n--- Evaluating Random Forest Model ---")
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    for metric, value in rf_metrics.items():
        print(f"Random Forest {metric.replace('_', ' ').title()}: {value:.4f}")

    print("\n--- Saving and Loading Models ---")
    save_model(lr_model, 'logistic_regression_model.joblib')
    loaded_lr_model = load_model('logistic_regression_model.joblib')
    
    save_model(rf_model, 'random_forest_model.joblib')
    loaded_rf_model = load_model('random_forest_model.joblib')
    
    print("\n--- Making a Prediction with Loaded Model ---")
    # Create a dummy patient data for prediction
    new_patient_data = pd.DataFrame({
        'Age': [55],
        'BMI': [28.5],
        'Glucose': [130],
        'BloodPressure': [100],
        'Smoking': ['No'],
        'Gender': ['Female']
    })
    
    lr_prediction_proba = predict_disease_likelihood(loaded_lr_model, new_patient_data)
    rf_prediction_proba = predict_disease_likelihood(loaded_rf_model, new_patient_data)
    
    print(f"New patient data:\n{new_patient_data}")
    print(f"Predicted likelihood of disease (Logistic Regression): {lr_prediction_proba[0]:.4f}")
    print(f"Predicted likelihood of disease (Random Forest): {rf_prediction_proba[0]:.4f}")

    # Clean up dummy data
    import os
    os.remove('medical_data.csv')
    os.remove('logistic_regression_model.joblib')
    os.remove('random_forest_model.joblib')