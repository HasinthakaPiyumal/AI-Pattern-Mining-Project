import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "age": np.random.randint(20, 80, num_samples),
        "gender": np.random.choice(["Male", "Female"], num_samples),
        "bmi": np.random.normal(25, 5, num_samples),
        "blood_pressure_systolic": np.random.normal(120, 15, num_samples),
        "blood_pressure_diastolic": np.random.normal(80, 10, num_samples),
        "cholesterol": np.random.normal(200, 30, num_samples),
        "glucose": np.random.normal(100, 20, num_samples),
        "smoking": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        "exercise_hours_week": np.random.randint(0, 10, num_samples),
        "chronic_disease": np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
    }
    df = pd.DataFrame(data)
    df.loc[df["age"] > 60, "chronic_disease"] = np.random.choice([0, 1], df[df["age"] > 60].shape[0], p=[0.4, 0.6])
    df.loc[df["bmi"] > 30, "chronic_disease"] = np.random.choice([0, 1], df[df["bmi"] > 30].shape[0], p=[0.3, 0.7])
    df.loc[df["glucose"] > 120, "chronic_disease"] = np.random.choice([0, 1], df[df["glucose"] > 120].shape[0], p=[0.2, 0.8])
    return df

if not os.path.exists("models"):
    os.makedirs("models")

df = generate_synthetic_data()

X = df.drop("chronic_disease", axis=1)
y = df["chronic_disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

categorical_features = ["gender"]
numerical_features = ["age", "bmi", "blood_pressure_systolic", "blood_pressure_diastolic", "cholesterol", "glucose", "smoking", "exercise_hours_week"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

joblib.dump(preprocessor, "models/preprocessor.pkl")
joblib.dump(X_train_processed, "models/X_train_processed.pkl")
joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(X_test_processed, "models/X_test_processed.pkl")
joblib.dump(y_test, "models/y_test.pkl")

X_train_processed = joblib.load("models/X_train_processed.pkl")
y_train = joblib.load("models/y_train.pkl")
X_test_processed = joblib.load("models/X_test_processed.pkl")
y_test = joblib.load("models/y_test.pkl")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)
y_proba = model.predict_proba(X_test_processed)[:, 1]

joblib.dump(model, "models/random_forest_model.pkl")

app = FastAPI()

try:
    loaded_model = joblib.load("models/random_forest_model.pkl")
    loaded_preprocessor = joblib.load("models/preprocessor.pkl")
except FileNotFoundError:
    print("Error: Model or preprocessor files not found. Ensure the script has been run to generate them.")
    exit()

class PatientData(BaseModel):
    age: int
    gender: str
    bmi: float
    blood_pressure_systolic: int
    blood_pressure_diastolic: int
    cholesterol: int
    glucose: int
    smoking: int
    exercise_hours_week: int

@app.post("/predict")
def predict_chronic_disease(patient: PatientData):
    input_df = pd.DataFrame([patient.dict()])
    processed_input = loaded_preprocessor.transform(input_df)
    
    prediction_proba = loaded_model.predict_proba(processed_input)[:, 1]
    prediction_label = loaded_model.predict(processed_input)[0]

    return {
        "chronic_disease_likelihood": float(prediction_proba[0]),
        "chronic_disease_prediction": int(prediction_label),
        "explanation": "0 means low likelihood, 1 means high likelihood of chronic disease."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)