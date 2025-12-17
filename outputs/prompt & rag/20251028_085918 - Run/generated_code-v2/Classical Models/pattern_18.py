import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. Simulated Data Generation ---
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'BloodPressure_Systolic': np.random.randint(100, 180, num_samples),
        'BloodPressure_Diastolic': np.random.randint(60, 120, num_samples),
        'Cholesterol': np.random.randint(150, 300, num_samples),
        'Glucose': np.random.randint(70, 200, num_samples),
        'Smoking': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'Alcohol': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        'ExerciseHoursWeekly': np.random.uniform(0.5, 10, num_samples),
        'BMI': np.random.uniform(18.0, 35.0, num_samples),
        'Diagnosis': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]) # 0: No Disease, 1: Disease
    }
    df = pd.DataFrame(data)
    
    # Introduce some missing values for demonstration
    for col in ['BloodPressure_Systolic', 'Cholesterol', 'Glucose']:
        df.loc[df.sample(frac=0.05).index, col] = np.nan
        
    return df

# --- 2. Data Preprocessing and Feature Engineering Pipeline ---
def create_preprocessing_pipeline(data_df):
    numeric_features = data_df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = data_df.select_dtypes(include='object').columns.tolist()
    
    # Remove target from features if present
    if 'Diagnosis' in numeric_features:
        numeric_features.remove('Diagnosis')

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# --- 3. Model Training and Selection ---
def train_model(X, y, preprocessor):
    # Create a full pipeline including preprocessing and the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Hyperparameter tuning (example, can be more extensive)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None]
    }
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation ROC AUC: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    return best_model

# --- 4. Model Evaluation ---
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:\n", cm)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc, "confusion_matrix": cm.tolist()}

# --- Main script to train and save the model ---
if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Creating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(X_train)
    
    print("Training the model...")
    trained_model_pipeline = train_model(X_train, y_train, preprocessor)
    
    print("Evaluating the model...")
    evaluation_results = evaluate_model(trained_model_pipeline, X_test, y_test)
    
    print("Saving the trained model and preprocessor...")
    joblib.dump(trained_model_pipeline, 'disease_diagnosis_model.joblib')
    print("Model saved as 'disease_diagnosis_model.joblib'")

    # --- 6. Prediction/Inference Service (FastAPI) ---
    app = FastAPI(title="Disease Diagnosis Prediction API")

    class PatientData(BaseModel):
        Age: int
        Gender: str
        BloodPressure_Systolic: float = None # Allow None for missing values
        BloodPressure_Diastolic: float
        Cholesterol: float = None
        Glucose: float = None
        Smoking: int
        Alcohol: int
        ExerciseHoursWeekly: float
        BMI: float

    # Load the trained model and preprocessor for inference
    try:
        loaded_model_pipeline = joblib.load('disease_diagnosis_model.joblib')
        print("Model loaded successfully for API.")
    except FileNotFoundError:
        print("Error: 'disease_diagnosis_model.joblib' not found. Please run the script to train and save the model first.")
        loaded_model_pipeline = None

    @app.post("/predict")
    async def predict_diagnosis(patient: PatientData):
        if loaded_model_pipeline is None:
            raise HTTPException(status_code=500, detail="Model not loaded. Please ensure the model is trained and saved.")

        # Convert incoming patient data to a pandas DataFrame
        input_df = pd.DataFrame([patient.dict()])
        
        # Ensure columns are in the same order as training data and handle missing/new columns
        # This is crucial for ColumnTransformer. It expects the same column names as during fit.
        # We need to explicitly handle potential missing columns in the input_df compared to training X
        # For this example, we'll assume the input_df has all the expected columns.
        # In a real application, you might need a more robust way to align columns.
        
        # Make prediction
        prediction = loaded_model_pipeline.predict(input_df)[0]
        prediction_proba = loaded_model_pipeline.predict_proba(input_df)[0].tolist()
        
        diagnosis_label = "Disease Present" if prediction == 1 else "No Disease"

        return {
            "diagnosis": diagnosis_label,
            "prediction_probability": {"No Disease": prediction_proba[0], "Disease Present": prediction_proba[1]}
        }

    print("\nTo run the FastAPI service, save this code as a .py file (e.g., disease_diagnosis_app.py) and run:\n")
    print("\tuvicorn disease_diagnosis_app:app --reload\n")
    print("Then, access the API at http://127.0.0.1:8000/docs for interactive documentation.")
