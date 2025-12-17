import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

# --- 1. Data Ingestion and Preprocessing Layer ---

def load_and_preprocess_data(df):
    numerical_features = ['age', 'bmi', 'blood_pressure', 'cholesterol']
    categorical_features = ['smoking', 'exercise_habit', 'family_history']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Separate features and target
    X = df.drop('cardio_disease', axis=1)
    y = df['cardio_disease']

    return X, y, preprocessor


# --- 2. Classical Machine Learning Model (for Structured Data) ---

def train_and_evaluate_model(X_train, X_test, y_train, y_test, preprocessor):
    
    # Logistic Regression Pipeline
    log_reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
    ])
    log_reg_pipeline.fit(X_train, y_train)
    y_pred_log_reg = log_reg_pipeline.predict(X_test)
    y_proba_log_reg = log_reg_pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Logistic Regression Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_log_reg):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_log_reg):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_log_reg):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_log_reg):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))

    # Random Forest Pipeline
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Random Forest Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    
    return log_reg_pipeline, rf_pipeline


# --- 5. Prediction/Inference API (Simplified) ---

def predict_risk(model_pipeline, new_patient_data):
    # Ensure new_patient_data is a DataFrame
    if isinstance(new_patient_data, dict):
        new_patient_data = pd.DataFrame([new_patient_data])
        
    prediction = model_pipeline.predict(new_patient_data)
    probability = model_pipeline.predict_proba(new_patient_data)[:, 1]
    return prediction[0], probability[0]


# --- Main Execution Block ---
if __name__ == "__main__":
    # Generate a synthetic dataset for demonstration
    np.random.seed(42)
    n_samples = 1000
    data = {
        'age': np.random.randint(20, 80, n_samples),
        'bmi': np.random.uniform(18.0, 40.0, n_samples),
        'blood_pressure': np.random.randint(90, 180, n_samples),
        'cholesterol': np.random.randint(150, 300, n_samples),
        'smoking': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'exercise_habit': np.random.choice(['Active', 'Moderate', 'Sedentary'], n_samples, p=[0.4, 0.3, 0.3]),
        'family_history': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
        'cardio_disease': np.random.randint(0, 2, n_samples) # 0 for no disease, 1 for disease
    }
    # Introduce some correlation for 'cardio_disease'
    data['cardio_disease'] = (
        (data['age'] > 50).astype(int) +
        (data['bmi'] > 28).astype(int) +
        (data['blood_pressure'] > 140).astype(int) +
        (data['cholesterol'] > 220).astype(int) +
        (data['smoking'] == 'Yes').astype(int) +
        (data['family_history'] == 'Yes').astype(int)
    ) > 2 # If more than 2 risk factors, likely disease
    data['cardio_disease'] = data['cardio_disease'].astype(int)
    
    df = pd.DataFrame(data)

    print("Generated Synthetic Dataset Head:")
    print(df.head())
    print("\nTarget distribution:")
    print(df['cardio_disease'].value_counts())

    # Prepare data for modeling
    X, y, preprocessor = load_and_preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train and evaluate models
    logistic_regression_model, random_forest_model = train_and_evaluate_model(X_train, X_test, y_train, y_test, preprocessor)

    # Save the trained models
    joblib.dump(logistic_regression_model, 'logistic_regression_model.pkl')
    joblib.dump(random_forest_model, 'random_forest_model.pkl')
    print("\nModels saved as 'logistic_regression_model.pkl' and 'random_forest_model.pkl'")

    # Load a model and make a prediction on new data
    loaded_lr_model = joblib.load('logistic_regression_model.pkl')
    print("\nLoaded Logistic Regression model.")

    new_patient = {
        'age': 65,
        'bmi': 32.5,
        'blood_pressure': 150,
        'cholesterol': 240,
        'smoking': 'Yes',
        'exercise_habit': 'Sedentary',
        'family_history': 'Yes'
    }

    disease_prediction, disease_probability = predict_risk(loaded_lr_model, new_patient)
    print(f"\nPrediction for new patient (Logistic Regression): {'Disease Detected' if disease_prediction == 1 else 'No Disease'}")
    print(f"Probability of Disease: {disease_probability:.4f}")

    new_patient_2 = {
        'age': 30,
        'bmi': 22.0,
        'blood_pressure': 110,
        'cholesterol': 180,
        'smoking': 'No',
        'exercise_habit': 'Active',
        'family_history': 'No'
    }
    disease_prediction_2, disease_probability_2 = predict_risk(loaded_lr_model, new_patient_2)
    print(f"\nPrediction for another new patient (Logistic Regression): {'Disease Detected' if disease_prediction_2 == 1 else 'No Disease'}")
    print(f"Probability of Disease: {disease_probability_2:.4f}")