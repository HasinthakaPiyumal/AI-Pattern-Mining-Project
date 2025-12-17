import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- Start of main.py content ---

def run_main():
    print("Running main.py: Training and evaluating classical models...")

    # 1. Data Ingestion and Preprocessing Module
    # Create a dummy dataset for demonstration
    data = {
        'Age': [45, 62, 34, 58, 71, 39, 50, 67, 42, 55, 48, 60, 30, 65, 52, 40, 59, 70, 33, 53],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Female', 'Male'],
        'BMI': [28.5, 31.2, 22.1, 29.8, 33.5, 25.0, 27.6, 30.1, 23.9, 26.7, 27.0, 32.0, 21.0, 30.5, 28.0, 24.5, 31.5, 34.0, 23.0, 29.0],
        'Cholesterol': [200, 240, 180, 220, 260, 190, 210, 250, 175, 205, 200, 230, 170, 255, 215, 185, 225, 270, 165, 210],
        'Glucose': [100, 130, 85, 115, 150, 90, 105, 140, 80, 110, 95, 125, 75, 145, 100, 88, 120, 155, 70, 112],
        'BloodPressure': [120, 140, 110, 130, 150, 115, 125, 145, 105, 135, 118, 138, 108, 148, 128, 112, 132, 152, 102, 122],
        'Smoking': ['No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes'],
        'Exercise': ['Regular', 'Irregular', 'Regular', 'Irregular', 'None', 'Regular', 'Irregular', 'Regular', 'Regular', 'Irregular', 'Regular', 'Irregular', 'Regular', 'None', 'Regular', 'Irregular', 'Regular', 'None', 'Regular', 'Irregular'],
        'Diagnosis': ['No Disease', 'Heart Disease', 'No Disease', 'Diabetes', 'Heart Disease', 'No Disease', 'Diabetes', 'Heart Disease', 'No Disease', 'Diabetes', 'No Disease', 'Heart Disease', 'No Disease', 'Diabetes', 'No Disease', 'No Disease', 'Diabetes', 'Heart Disease', 'No Disease', 'Diabetes']
    }
    df = pd.DataFrame(data)

    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(include='object').columns

    # Create preprocessing pipelines for numerical and categorical features
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

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Model Training Module
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Support Vector Machine': SVC(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    best_model_name = None
    best_accuracy = 0
    best_pipeline = None

    print("\nTraining and evaluating models:")
    for name, model in models.items():
        print(f"\n--- {name} ---")
        # Create a full pipeline (preprocessor + model)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # 3. Model Evaluation Module
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest performing model: {best_model_name} with accuracy: {best_accuracy:.4f}")

    # 4. Model Persistence Module
    model_filename = "best_disease_predictor_pipeline.joblib"
    joblib.dump(best_pipeline, model_filename)
    print(f"\nSaved best model pipeline to {model_filename}")

    # Demonstrate a simple prediction using the loaded model
    print("\n--- Demonstrating Prediction with Loaded Model ---")
    loaded_pipeline = joblib.load(model_filename)

    # Example new patient data for prediction
    new_patient_data = pd.DataFrame([{ 
        'Age': 50,
        'Gender': 'Male',
        'BMI': 29.0,
        'Cholesterol': 210,
        'Glucose': 110,
        'BloodPressure': 130,
        'Smoking': 'No',
        'Exercise': 'Regular'
    }])

    prediction = loaded_pipeline.predict(new_patient_data)
    print(f"New patient data:\n{new_patient_data}")
    print(f"Predicted Diagnosis: {prediction[0]}")

# --- End of main.py content ---


# --- Start of prediction_api.py content ---
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os

API_MODEL_PATH = "best_disease_predictor_pipeline.joblib"

# Initialize FastAPI app
app = FastAPI(title="Healthcare Disease Prediction API", description="Predicts disease likelihood using classical ML models.")

# Load the saved pipeline (preprocessor + model) once when the application starts
# This will attempt to load the model saved by run_main()
loaded_api_pipeline = None
try:
    if os.path.exists(API_MODEL_PATH):
        loaded_api_pipeline = joblib.load(API_MODEL_PATH)
        print(f"Successfully loaded model for API from {API_MODEL_PATH}")
    else:
        print(f"Warning: Model file {API_MODEL_PATH} not found for API. Please run the training part first (run_main()).")
except Exception as e:
    print(f"Error loading model for API: {e}")


# Define the input data model using Pydantic
class PatientData(BaseModel):
    Age: int
    Gender: str
    BMI: float
    Cholesterol: int
    Glucose: int
    BloodPressure: int
    Smoking: str
    Exercise: str

@app.post("/predict")
async def predict_disease(patient: PatientData):
    if loaded_api_pipeline is None:
        return {"error": "Model not loaded. Please ensure the model file exists and is accessible by running the training script first."}

    # Convert incoming Pydantic model data to a pandas DataFrame
    input_df = pd.DataFrame([patient.dict()])

    # Make prediction
    try:
        prediction = loaded_api_pipeline.predict(input_df)[0]
        return {"prediction": prediction}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# --- End of prediction_api.py content ---


if __name__ == "__main__":
    # This block allows you to run the training/evaluation part
    # or start the FastAPI server depending on your need.
    # To run the training and save the model:
    run_main()
    
    # To run the FastAPI server (after running run_main() once to save the model):
    print("\n--------------------------------------------------------------")
    print("To run the FastAPI server, you would typically run it in a separate process/terminal.")
    print("Ensure you have 'uvicorn' installed (`pip install uvicorn`).")
    print("Then, from your terminal, navigate to this file's directory and execute:")
    print("uvicorn disease_prediction_system:app --host 0.0.0.0 --port 8000 --reload")
    print("--------------------------------------------------------------")
    
    # For demonstration, if you want to explicitly start the API here (will block):
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
