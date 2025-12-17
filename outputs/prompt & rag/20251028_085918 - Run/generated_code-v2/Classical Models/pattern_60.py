import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

# --- Part 1: Training and Evaluation Script ---

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 90, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'admission_type': np.random.choice(['Emergency', 'Urgent', 'Elective'], n_samples),
        'num_diagnoses': np.random.randint(1, 10, n_samples),
        'num_medications': np.random.randint(5, 30, n_samples),
        'time_in_hospital': np.random.randint(1, 15, n_samples),
        'medical_specialty': np.random.choice(['Cardiology', 'Internal Medicine', 'Surgery', 'Pediatrics', 'Oncology', 'Other'], n_samples),
        'num_lab_procedures': np.random.randint(10, 80, n_samples),
        'num_procedures': np.random.randint(0, 6, n_samples),
        'glucose_level': np.random.randint(70, 200, n_samples),
        'blood_pressure_systolic': np.random.randint(90, 180, n_samples),
        'blood_pressure_diastolic': np.random.randint(60, 110, n_samples),
        'readmitted': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]) # 0: No readmission, 1: Readmission
    }
    df = pd.DataFrame(data)
    
    # Introduce some missing values for demonstration
    for col in ['medical_specialty', 'glucose_level']:
        df.loc[df.sample(frac=0.05).index, col] = np.nan

    return df

def train_model():
    print("Generating synthetic data...")
    df = generate_synthetic_data()

    # Define target and features
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']

    # Split data into training and testing sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', pd.DataFrame.fillna, {'value': X_train[numerical_features].mean()}), # Simple mean imputation
        ('scaler', StandardScaler())
    ])
    # For OneHotEncoder, handle_unknown='ignore' to avoid errors during prediction if unseen category appears
    categorical_transformer = Pipeline(steps=[
        ('imputer', pd.DataFrame.fillna, {'value': 'missing'}), # Simple mode imputation for categorical
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Manually fit and transform for demonstration, as pipeline needs to handle Series/DataFrame properly
    # For numerical features, a custom imputer lambda might be needed or fit on X_train directly
    # For simplicity, let's use a function that correctly handles DataFrames or Series.
    
    # Define a custom imputer for numerical data within the pipeline to handle DataFrames
    class CustomNumericalImputer():
        def __init__(self):
            self.mean_values = None
        
        def fit(self, X, y=None):
            self.mean_values = X.mean()
            return self
        
        def transform(self, X):
            return X.fillna(self.mean_values)

    numerical_transformer = Pipeline(steps=[
        ('imputer', CustomNumericalImputer()),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the full pipeline with preprocessor and XGBoost classifier
    print("Building and training model pipeline...")
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Save the trained model and preprocessor
    model_filename = 'readmission_model.joblib'
    joblib.dump(model_pipeline, model_filename)
    print(f"\nModel and preprocessor saved as {model_filename}")
    
    return model_pipeline, X_train.columns.tolist() # Return trained pipeline and original column order

# Run the training process if this script is executed directly
if __name__ == '__main__':
    _, _ = train_model()


# --- Part 2: FastAPI Application (for deployment) ---

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load the trained model and preprocessor (these should be available after running train_model)
try:
    loaded_model_pipeline = joblib.load('readmission_model.joblib')
    # Extract feature names from the trained preprocessor to ensure correct input order
    # This part can be tricky with ColumnTransformer if feature names are not preserved.
    # For simplicity, we assume the order of features passed to the API matches the training order.
    # In a real application, you might save feature names explicitly or use a more robust preprocessor.
    print("Model loaded successfully for API.")
except FileNotFoundError:
    print("Error: Model file 'readmission_model.joblib' not found. Please run the training script first.")
    loaded_model_pipeline = None

app = FastAPI(
    title="Patient Readmission Risk Prediction API",
    description="API for predicting patient readmission risk using a trained ML model.",
    version="1.0.0",
)

# Define the input data model for the API
class PatientData(BaseModel):
    age: int
    gender: str
    admission_type: str
    num_diagnoses: int
    num_medications: int
    time_in_hospital: int
    medical_specialty: str
    num_lab_procedures: int
    num_procedures: int
    glucose_level: float = None # Allow None for missing values
    blood_pressure_systolic: int
    blood_pressure_diastolic: int

@app.get("/", summary="Root")
async def root():
    return {"message": "Welcome to the Patient Readmission Risk Prediction API!"}

@app.post("/predict", summary="Predict Patient Readmission Risk")
async def predict_readmission(patient_data: PatientData):
    if loaded_model_pipeline is None:
        return {"error": "Model not loaded. Please ensure 'readmission_model.joblib' exists."}

    # Convert input data to a pandas DataFrame in the correct order
    input_df = pd.DataFrame([patient_data.dict()])

    # Make prediction
    prediction_proba = loaded_model_pipeline.predict_proba(input_df)[:, 1][0]
    prediction = loaded_model_pipeline.predict(input_df)[0]

    return {
        "readmission_risk_score": float(prediction_proba),
        "readmitted_prediction": int(prediction),
        "explanation": "0 indicates no readmission, 1 indicates readmission."
    }

# To run the FastAPI application, save this code as 'app.py' and execute 'uvicorn app:app --reload'
# For demonstration purposes, if you were to run the FastAPI part directly from this combined file:
# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=8000)


