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
import os

# FastAPI related imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- 1. Data Generation (for demonstration purposes) ---
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(29, 77, num_samples),
        'sex': np.random.choice([0, 1], num_samples), # 0: female, 1: male
        'cp': np.random.randint(0, 4, num_samples), # chest pain type
        'trestbps': np.random.randint(90, 200, num_samples), # resting blood pressure
        'chol': np.random.randint(120, 564, num_samples), # serum cholestoral in mg/dl
        'fbs': np.random.choice([0, 1], num_samples), # fasting blood sugar > 120 mg/dl
        'restecg': np.random.randint(0, 3, num_samples), # resting electrocardiographic results
        'thalach': np.random.randint(71, 202, num_samples), # maximum heart rate achieved
        'exang': np.random.choice([0, 1], num_samples), # exercise induced angina
        'oldpeak': np.round(np.random.uniform(0, 6.2, num_samples), 1), # ST depression induced by exercise relative to rest
        'slope': np.random.randint(0, 3, num_samples), # the slope of the peak exercise ST segment
        'ca': np.random.randint(0, 5, num_samples), # number of major vessels (0-4) colored by flourosopy
        'thal': np.random.randint(0, 4, num_samples), # 0: normal; 1: fixed defect; 2: reversible defect
        'target': np.random.choice([0, 1], num_samples, p=[0.5, 0.5]) # 0: no disease, 1: disease
    }
    df = pd.DataFrame(data)
    return df

# --- 2. Data Preprocessing and Model Training --- 
def train_and_save_model(data_path='heart_disease_data.csv', model_filename='heart_disease_model.joblib', preprocessor_filename='heart_disease_preprocessor.joblib'):
    if not os.path.exists(data_path):
        print("Generating synthetic data...")
        df = generate_synthetic_data()
        df.to_csv(data_path, index=False)
    else:
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define categorical and numerical features
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create a pipeline with preprocessing and a model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    print("Training model...")
    model_pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nModel Evaluation (RandomForestClassifier):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:\n", cm)

    # Save the trained pipeline (model + preprocessor)
    joblib.dump(model_pipeline, model_filename)
    print(f"Model saved as {model_filename}")
    # Note: The preprocessor is part of the pipeline, so no need to save separately
    return model_pipeline

# --- 3. FastAPI Application for Prediction ---
app = FastAPI()

# Load the pre-trained model and preprocessor
MODEL_FILENAME = 'heart_disease_model.joblib'
PREPROCESSOR_FILENAME = 'heart_disease_preprocessor.joblib' # Not strictly needed if preprocessor is in pipeline

model = None

@app.on_event("startup")
async def load_model():
    global model
    if not os.path.exists(MODEL_FILENAME):
        # If model doesn't exist, try to train it
        print(f"Model file '{MODEL_FILENAME}' not found. Attempting to train and save model...")
        model = train_and_save_model()
    else:
        print(f"Loading model from {MODEL_FILENAME}...")
        try:
            model = joblib.load(MODEL_FILENAME)
            print("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Could not load model: {e}")
    if model is None:
        raise RuntimeError("Model could not be loaded or trained at startup.")


class PatientData(BaseModel):
    age: int
    sex: int # 0 for female, 1 for male
    cp: int # chest pain type (0-3)
    trestbps: int # resting blood pressure
    chol: int # serum cholestoral
    fbs: int # fasting blood sugar > 120 mg/dl (1=true; 0=false)
    restecg: int # resting electrocardiographic results (0-2)
    thalach: int # maximum heart rate achieved
    exang: int # exercise induced angina (1=yes; 0=no)
    oldpeak: float # ST depression induced by exercise relative to rest
    slope: int # the slope of the peak exercise ST segment (0-2)
    ca: int # number of major vessels (0-4) colored by flourosopy
    thal: int # 0: normal; 1: fixed defect; 2: reversible defect (this varies, common values 1, 2, 3)

@app.get("/")
async def read_root():
    return {"message": "Heart Disease Prediction API. Go to /docs for API details."}

@app.post("/predict/")
async def predict_heart_disease(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please restart the application or check model training.")

    # Convert input data to DataFrame for prediction
    input_df = pd.DataFrame([data.dict()])

    try:
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1] # Probability of heart disease

        return {
            "prediction": int(prediction),
            "probability_of_disease": round(prediction_proba, 4),
            "interpretation": "Heart disease detected" if prediction == 1 else "No heart disease detected"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}. Check input data format.")

# To run the FastAPI app (typically from terminal: uvicorn heart_disease_predictor_app:app --reload)
# Or, uncomment the following block to run directly within the script (for demonstration):
if __name__ == "__main__":
    # Ensure model is trained and saved before starting API, or handle it in startup event
    if not os.path.exists(MODEL_FILENAME):
        print("No existing model found, initiating training...")
        train_and_save_model()
    else:
        print("Existing model found, skipping explicit training step here (will load on API startup).")

    print("\nStarting FastAPI application. Access API docs at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)