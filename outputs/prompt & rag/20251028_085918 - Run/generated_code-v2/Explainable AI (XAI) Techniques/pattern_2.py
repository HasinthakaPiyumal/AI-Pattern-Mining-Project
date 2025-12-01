import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import streamlit as st
import requests
import json
import joblib
import os

class PatientData(BaseModel):
    age: int
    symptom_A: int
    symptom_B: int
    lab_result_X: float
    lab_result_Y: float
    genetic_marker_Z: int

class PredictionResult(BaseModel):
    predicted_disease: str
    probability: float

class ExplanationFeature(BaseModel):
    feature: str
    importance: float
    value: Any

class ExplanationResult(BaseModel):
    explanation: List[ExplanationFeature]

class RareDiseaseClassifier:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self._train_model()

    def _train_model(self):
        np.random.seed(42)
        n_samples = 1000
        data = {
            "age": np.random.randint(10, 80, n_samples),
            "symptom_A": np.random.randint(0, 2, n_samples),
            "symptom_B": np.random.randint(0, 2, n_samples),
            "lab_result_X": np.random.normal(50, 10, n_samples),
            "lab_result_Y": np.random.normal(10, 2, n_samples),
            "genetic_marker_Z": np.random.randint(0, 3, n_samples),
        }
        df = pd.DataFrame(data)

        df["Rare_Disease"] = ((df["symptom_A"] == 1) & (df["lab_result_X"] > 60) & (df["age"] > 50)) | \
                             ((df["symptom_B"] == 1) & (df["genetic_marker_Z"] == 2) & (df["lab_result_Y"] < 8))
        df["Rare_Disease"] = df["Rare_Disease"].astype(int)

        X = df.drop("Rare_Disease", axis=1)
        y = df["Rare_Disease"]
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def predict_proba(self, data: pd.DataFrame):
        return self.model.predict_proba(data)

    def get_feature_importances(self):
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None

class ExplanationService:
    def __init__(self, classifier: RareDiseaseClassifier):
        self.classifier = classifier

    def explain(self, patient_data: Dict[str, Any]):
        feature_importances = self.classifier.get_feature_importances()
        feature_names = self.classifier.feature_names

        explanation_features = []
        if feature_importances is not None:
            sorted_indices = np.argsort(feature_importances)[::-1]
            for idx in sorted_indices:
                explanation_features.append({
                    "feature": feature_names[idx],
                    "importance": feature_importances[idx],
                    "value": patient_data.get(feature_names[idx])
                })
        return explanation_features

app = FastAPI()

ml_classifier = RareDiseaseClassifier()
explanation_service = ExplanationService(ml_classifier)

@app.post("/predict", response_model=PredictionResult)
async def predict_rare_disease(patient_data: PatientData):
    df_data = pd.DataFrame([patient_data.dict()])
    proba = ml_classifier.predict_proba(df_data)[0]
    if proba[1] > proba[0]:
        predicted_disease = "Rare Disease"
        probability = proba[1]
    else:
        predicted_disease = "No Rare Disease"
        probability = proba[0]
    return PredictionResult(predicted_disease=predicted_disease, probability=probability)

@app.post("/explain", response_model=ExplanationResult)
async def get_explanation(patient_data: PatientData):
    explanation_features = explanation_service.explain(patient_data.dict())
    return ExplanationResult(explanation=explanation_features)

BACKEND_URL = "http://127.0.0.1:8000"

def run_streamlit_frontend():
    st.title("Interactive AI-powered Medical Diagnosis Explainer for Rare Disease Prediction")

    st.sidebar.header("Patient Data Input")

    with st.sidebar.form("patient_form"):
        age = st.slider("Age", 10, 80, 45)
        symptom_A = st.checkbox("Symptom A Present", value=False)
        symptom_B = st.checkbox("Symptom B Present", value=False)
        lab_result_X = st.number_input("Lab Result X", min_value=1.0, max_value=100.0, value=55.0, step=0.1)
        lab_result_Y = st.number_input("Lab Result Y", min_value=1.0, max_value=20.0, value=10.0, step=0.1)
        genetic_marker_Z = st.selectbox("Genetic Marker Z", options=[0, 1, 2], index=0)

        submitted = st.form_submit_button("Get Prediction and Explanation")

    if submitted:
        patient_data = {
            "age": age,
            "symptom_A": 1 if symptom_A else 0,
            "symptom_B": 1 if symptom_B else 0,
            "lab_result_X": lab_result_X,
            "lab_result_Y": lab_result_Y,
            "genetic_marker_Z": genetic_marker_Z
        }

        try:
            predict_response = requests.post(f"{BACKEND_URL}/predict", json=patient_data)
            predict_response.raise_for_status()
            prediction = predict_response.json()

            st.subheader("Prediction Result")
            st.write(f"**Predicted Disease:** {prediction['predicted_disease']}")
            st.write(f"**Probability:** {prediction['probability']:.4f}")

            explain_response = requests.post(f"{BACKEND_URL}/explain", json=patient_data)
            explain_response.raise_for_status()
            explanation = explain_response.json()

            st.subheader("Explanation (Feature Importance)")
            st.write("Features contributing most to the prediction:")
            explanation_df = pd.DataFrame(explanation["explanation"])
            st.dataframe(explanation_df.round(4))

            st.subheader("What-If Analysis")
            st.write("Modify inputs in the sidebar and click 'Get Prediction and Explanation' again to see how changes affect the outcome.")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend server. Please ensure the FastAPI server is running at " + BACKEND_URL)
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")
        except json.JSONDecodeError:
            st.error("Failed to decode JSON response from the backend.")

if __name__ == "__main__":
    pass

if "streamlit" in os.environ.get("STREAMLIT_SERVER_URL", "").lower() or "streamlit" in os.environ.get("_STREAMLIT_SCRIPT_RUNNING_CONTEXT", "").lower():
     run_streamlit_frontend()