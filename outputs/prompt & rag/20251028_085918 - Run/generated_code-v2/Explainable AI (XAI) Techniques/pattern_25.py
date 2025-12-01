# requirements.txt
# pandas
# scikit-learn
# numpy
# fastapi
# uvicorn
# pydantic
# matplotlib
# seaborn
# joblib

# To run this project:
# 1. Save the entire code block below into a single file, e.g., `app.py`.
# 2. Make sure you have the required libraries installed: `pip install -r requirements.txt` (if you save the first part as requirements.txt)
#    Alternatively, manually install: `pip install pandas scikit-learn numpy fastapi uvicorn pydantic matplotlib seaborn joblib`
# 3. Create directories: `mkdir data models`
# 4. Run the training script part once:
#    If you separated the train_model.py part, run `python train_model.py`.
#    If all in one file (`app.py`), find the `if __name__ == "__main__" and "TRAIN_MODEL_FLAG"` section and uncomment/execute it.
#    For simplicity, I will integrate the training logic to run automatically if the model/data files are missing when the FastAPI app starts.
# 5. Run the FastAPI application: `uvicorn app:app --reload` (assuming you named the file `app.py`)
# 6. Access API at `http://127.0.0.1:8000/docs`

# --- Start of Combined Code (app.py) ---

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import joblib
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Define paths
DATA_DIR = "data"
MODELS_DIR = "models"
PATIENT_DATA_FILE = os.path.join(DATA_DIR, "patient_data.csv")
TRAINED_MODEL_FILE = os.path.join(MODELS_DIR, "readmission_model.joblib")
TEST_DATA_FEATURES_FILE = os.path.join(DATA_DIR, "test_data_features.csv")
TEST_DATA_TARGET_FILE = os.path.join(DATA_DIR, "test_data_target.csv")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Simulate train_model.py logic ---
def train_and_save_model():
    # Generate synthetic patient data
    np.random.seed(42)
    n_samples = 1000

    data = {
        "age": np.random.randint(18, 90, n_samples),
        "comorbidity_score": np.random.randint(0, 10, n_samples),
        "num_admissions_prev_year": np.random.randint(0, 5, n_samples),
        "length_of_stay": np.random.randint(1, 30, n_samples),
        "insurance_type_A": np.random.randint(0, 2, n_samples),
        "insurance_type_B": np.random.randint(0, 2, n_samples),
        "insurance_type_C": np.random.randint(0, 2, n_samples),
        "diagnosis_cardiac": np.random.randint(0, 2, n_samples),
        "diagnosis_respiratory": np.random.randint(0, 2, n_samples),
        "medication_count": np.random.randint(1, 15, n_samples),
        "readmitted": np.random.randint(0, 2, n_samples) # Target variable
    }
    df = pd.DataFrame(data)

    # Introduce some correlation for 'readmitted'
    df["readmitted"] = (
        0.3 * df["age"]
        + 0.5 * df["comorbidity_score"]
        + 0.7 * df["num_admissions_prev_year"]
        + 0.2 * df["length_of_stay"]
        + -0.4 * df["insurance_type_A"]
        + np.random.normal(0, 2, n_samples) > 20
    ).astype(int)

    # Save full dataset (optional, mainly for reference)
    df.to_csv(PATIENT_DATA_FILE, index=False)

    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train a black-box model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, TRAINED_MODEL_FILE)
    print(f"Model trained and saved to {TRAINED_MODEL_FILE}")

    # Save test features and target for explanation
    X_test.to_csv(TEST_DATA_FEATURES_FILE, index=False)
    y_test.to_csv(TEST_DATA_TARGET_FILE, index=False)
    print(f"Test data for explanation saved to {TEST_DATA_FEATURES_FILE} and {TEST_DATA_TARGET_FILE}")

# --- Simulate model.py logic ---
def load_readmission_model():
    if not os.path.exists(TRAINED_MODEL_FILE):
        print("Trained model not found. Training a new model...")
        train_and_save_model()
    return joblib.load(TRAINED_MODEL_FILE)

def calculate_permutation_importance_for_model(model, X_df, y_series, metric=roc_auc_score):
    result = permutation_importance(model, X_df, y_series, scoring=metric, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()[::-1] # Sort in descending order of importance

    importance_scores = []
    for i in sorted_idx:
        importance_scores.append({
            "feature": X_df.columns[i],
            "importance_mean": result.importances_mean[i],
            "importance_std": result.importances_std[i]
        })
    return importance_scores

# --- Simulate main.py logic (FastAPI Application) ---
app = FastAPI(
    title="Patient Readmission Risk Explainer",
    description="API for predicting patient readmission risk and explaining feature importance using Permutation Feature Importance.",
    version="1.0.0"
)

# Pydantic models for request and response
class PatientInput(BaseModel):
    age: int
    comorbidity_score: int
    num_admissions_prev_year: int
    length_of_stay: int
    insurance_type_A: int
    insurance_type_B: int
    insurance_type_C: int
    diagnosis_cardiac: int
    diagnosis_respiratory: int
    medication_count: int

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

class FeatureImportanceItem(BaseModel):
    feature: str
    importance_mean: float
    importance_std: float

class FeatureImportanceResponse(BaseModel):
    feature_importances: List[FeatureImportanceItem]

@app.on_event("startup")
async def startup_event():
    # Ensure model is trained and data for explanation exists on startup
    if not os.path.exists(TRAINED_MODEL_FILE) or not os.path.exists(TEST_DATA_FEATURES_FILE):
        print("Model or test data not found. Running initial training/data generation.")
        train_and_save_model()
    else:
        print("Model and test data found. Skipping initial training.")

@app.post("/predict", response_model=PredictionResponse, summary="Predict Patient Readmission Risk")
async def predict_readmission_risk(patient_data: PatientInput):
    model = load_readmission_model()
    
    # Convert input to DataFrame matching training features
    input_df = pd.DataFrame([patient_data.dict()])
    
    # Ensure all original features are present, fill missing with 0 if necessary
    # (This assumes the input PatientInput matches the training features.
    # For a robust app, you'd handle feature order and missing features more carefully.)
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[:, 1][0] # Probability of readmission (class 1)
    
    return PredictionResponse(prediction=int(prediction), probability=float(probability))

@app.get("/explain_importance", response_model=FeatureImportanceResponse, summary="Get Global Feature Importance")
async def get_global_feature_importance():
    model = load_readmission_model()

    if not os.path.exists(TEST_DATA_FEATURES_FILE) or not os.path.exists(TEST_DATA_TARGET_FILE):
        raise HTTPException(status_code=500, detail="Test data for explanation not found. Please ensure the model training script has run successfully.")
    
    X_test = pd.read_csv(TEST_DATA_FEATURES_FILE)
    y_test = pd.read_csv(TEST_DATA_TARGET_FILE).squeeze() # .squeeze() to convert single column DataFrame to Series

    if X_test.empty or y_test.empty:
        raise HTTPException(status_code=500, detail="Test data for explanation is empty.")

    importance_scores = calculate_permutation_importance_for_model(model, X_test, y_test)
    
    return FeatureImportanceResponse(feature_importances=importance_scores)

# To run this FastAPI application:
# Save the code as `app.py`.
# Then run `uvicorn app:app --reload` in your terminal.
# Access the API documentation at `http://127.0.0.1:8000/docs`