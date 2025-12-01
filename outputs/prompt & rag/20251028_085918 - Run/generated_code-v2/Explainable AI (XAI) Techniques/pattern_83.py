import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import json

app = FastAPI()

class PatientData(BaseModel):
    age: int
    gender: int
    bp_systolic: int
    bp_diastolic: int
    cholesterol: int
    glucose: int
    smoker: int
    alcohol: int
    active: int

models = {}
feature_names = ['age', 'gender', 'bp_systolic', 'bp_diastolic', 'cholesterol', 'glucose', 'smoker', 'alcohol', 'active']

def train_dummy_model(model_name):
    np.random.seed(42)
    num_samples = 1000
    data = {
        'age': np.random.randint(20, 80, num_samples),
        'gender': np.random.randint(0, 2, num_samples),
        'bp_systolic': np.random.randint(90, 180, num_samples),
        'bp_diastolic': np.random.randint(60, 120, num_samples),
        'cholesterol': np.random.randint(100, 300, num_samples),
        'glucose': np.random.randint(70, 200, num_samples),
        'smoker': np.random.randint(0, 2, num_samples),
        'alcohol': np.random.randint(0, 2, num_samples),
        'active': np.random.randint(0, 2, num_samples),
    }
    df = pd.DataFrame(data)
    df['target_A'] = ((df['bp_systolic'] > 140) | (df['cholesterol'] > 200) | (df['age'] > 60)).astype(int)
    df['target_B'] = ((df['glucose'] > 120) | (df['smoker'] == 1) | (df['active'] == 0)).astype(int)

    X_A = df[feature_names]
    y_A = df['target_A']
    X_B = df[feature_names]
    y_B = df['target_B']

    scaler_A = StandardScaler()
    X_A_scaled = scaler_A.fit_transform(X_A)
    model_A = LogisticRegression(random_state=42)
    model_A.fit(X_A_scaled, y_A)

    scaler_B = StandardScaler()
    X_B_scaled = scaler_B.fit_transform(X_B)
    model_B = LogisticRegression(random_state=42)
    model_B.fit(X_B_scaled, y_B)

    models[f"{model_name}_A"] = {'model': model_A, 'scaler': scaler_A, 'target_label': 'Disease A'}
    models[f"{model_name}_B"] = {'model': model_B, 'scaler': scaler_B, 'target_label': 'Disease B'}

train_dummy_model("Model_X")
train_dummy_model("Model_Y")

def generate_lace_explanation(model_name: str, patient_data: PatientData):
    features = list(patient_data.dict().values())
    np.random.seed(hash(frozenset(patient_data.dict().items())) % (2**32 - 1))

    feature_importances = {
        name: round(np.random.uniform(0.1, 0.9), 2)
        for name in feature_names
    }
    
    local_rules = []
    if patient_data.age > 60 and patient_data.cholesterol > 200:
        local_rules.append("High age and high cholesterol are significant factors.")
    if patient_data.smoker == 1 and patient_data.bp_systolic > 140:
        local_rules.append("Smoking combined with high systolic BP contributes to risk.")
    if not local_rules:
        local_rules.append("No specific strong local rules identified for this instance.")
        
    return {
        "feature_importances": feature_importances,
        "local_rules": local_rules
    }

class PredictionResponse(BaseModel):
    prediction_label: str
    probability: float

class ExplanationResponse(BaseModel):
    feature_importances: dict
    local_rules: list

class WhatIfResponse(BaseModel):
    original_prediction: PredictionResponse
    original_explanation: ExplanationResponse
    what_if_prediction: PredictionResponse
    what_if_explanation: ExplanationResponse

class SummaryRequest(BaseModel):
    explanations: list

class SummaryResponse(BaseModel):
    common_attributes: list
    prevalent_rules: list
    overall_insight: str

@app.post("/predict/{model_id}", response_model=PredictionResponse)
async def predict(model_id: str, data: PatientData):
    if model_id not in models:
        return {"prediction_label": "Error", "probability": 0.0, "detail": "Model not found"}

    model_info = models[model_id]
    model = model_info['model']
    scaler = model_info['scaler']
    target_label = model_info['target_label']

    input_df = pd.DataFrame([data.dict()])
    scaled_input = scaler.transform(input_df)
    
    prediction_proba = model.predict_proba(scaled_input)[0, 1]
    prediction_label = target_label if prediction_proba > 0.5 else f"No {target_label}"

    return PredictionResponse(prediction_label=prediction_label, probability=round(float(prediction_proba), 4))

@app.post("/explain/{model_id}", response_model=ExplanationResponse)
async def explain(model_id: str, data: PatientData):
    if model_id not in models:
        return {"feature_importances": {}, "local_rules": ["Model not found"]}
    
    explanation = generate_lace_explanation(model_id, data)
    return ExplanationResponse(**explanation)

@app.post("/what-if/{model_id}", response_model=WhatIfResponse)
async def what_if_analysis(model_id: str, original_data: PatientData, tweaked_data: PatientData):
    original_pred_resp = await predict(model_id, original_data)
    original_expl_resp = await explain(model_id, original_data)

    what_if_pred_resp = await predict(model_id, tweaked_data)
    what_if_expl_resp = await explain(model_id, tweaked_data)

    return WhatIfResponse(
        original_prediction=original_pred_resp,
        original_explanation=original_expl_resp,
        what_if_prediction=what_if_pred_resp,
        what_if_explanation=what_if_expl_resp
    )

@app.post("/summarize_explanations", response_model=SummaryResponse)
async def summarize_explanations(req: SummaryRequest):
    all_importances = {}
    all_rules = []

    for expl in req.explanations:
        for feature, importance in expl['feature_importances'].items():
            all_importances[feature] = all_importances.get(feature, 0) + importance
        all_rules.extend(expl['local_rules'])

    sorted_attributes = sorted(all_importances.items(), key=lambda item: item[1], reverse=True)
    common_attributes = [attr for attr, _ in sorted_attributes[:3]]

    rule_counts = pd.Series(all_rules).value_counts()
    prevalent_rules = rule_counts[rule_counts > 1].index.tolist()

    overall_insight = "Based on aggregated explanations, key attributes and rules influencing predictions have been identified."

    return SummaryResponse(
        common_attributes=common_attributes,
        prevalent_rules=prevalent_rules,
        overall_insight=overall_insight
    )

FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide", page_title="MedXplain: Interactive Diagnostic Assistant")

st.title("MedXplain: Interactive Diagnostic Assistant")
st.write("Understand and validate predictions from black-box medical AI models.")

st.sidebar.header("Patient Data Input")
selected_model_id = st.sidebar.selectbox("Select AI Model", list(models.keys()))

with st.sidebar.form("patient_data_form"):
    age = st.slider("Age", 20, 100, 50)
    gender = st.selectbox("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
    bp_systolic = st.slider("Systolic Blood Pressure", 80, 200, 120)
    bp_diastolic = st.slider("Diastolic Blood Pressure", 50, 140, 80)
    cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
    glucose = st.slider("Glucose (mg/dL)", 60, 300, 100)
    smoker = st.selectbox("Smoker", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    alcohol = st.selectbox("Alcohol Consumer", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    active = st.selectbox("Physically Active", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    submit_button = st.form_submit_button("Get Prediction & Explanation")

current_patient_data = {
    "age": age, "gender": gender, "bp_systolic": bp_systolic, "bp_diastolic": bp_diastolic,
    "cholesterol": cholesterol, "glucose": glucose, "smoker": smoker, "alcohol": alcohol, "active": active
}

if submit_button:
    st.session_state["current_patient_data"] = current_patient_data
    st.session_state["selected_model_id"] = selected_model_id
    
    try:
        predict_response = requests.post(
            f"{FASTAPI_URL}/predict/{selected_model_id}", json=current_patient_data
        )
        predict_response.raise_for_status()
        prediction = predict_response.json()
        st.session_state["prediction"] = prediction
        
        explain_response = requests.post(
            f"{FASTAPI_URL}/explain/{selected_model_id}", json=current_patient_data
        )
        explain_response.raise_for_status()
        explanation = explain_response.json()
        st.session_state["explanation"] = explanation

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the FastAPI backend. Please ensure it is running.")
    except requests.exceptions.HTTPError as e:
        st.error(f"Error from backend: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

if "prediction" in st.session_state and "explanation" in st.session_state:
    st.header(f"Prediction for {models[st.session_state['selected_model_id']]['target_label']}")
    st.metric(
        label=f"Diagnosis: {st.session_state['prediction']['prediction_label']}",
        value=f"{st.session_state['prediction']['probability']:.2f}"
    )

    st.subheader("Explanation (LACE based)")
    st.write("**Feature Importances:**")
    df_importances = pd.DataFrame(
        st.session_state["explanation"]["feature_importances"].items(),
        columns=["Feature", "Importance"]
    ).sort_values(by="Importance", ascending=False)
    st.bar_chart(df_importances.set_index("Feature"))

    st.write("**Local Rules:**")
    for rule in st.session_state["explanation"]["local_rules"]:
        st.markdown(f"- {rule}")

    st.markdown("---")
    st.header("What-If Analysis")
    st.write("Modify patient attributes to see how the prediction and explanation change.")

    if "current_patient_data" in st.session_state:
        original_data = st.session_state["current_patient_data"]
        
        with st.expander("Tweak Patient Attributes for What-If"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Data")
                st.json(original_data)
            with col2:
                st.subheader("Tweaked Data")
                tweaked_age = st.slider("Tweak Age", 20, 100, original_data["age"], key="tweaked_age")
                tweaked_bp_systolic = st.slider("Tweak Systolic Blood Pressure", 80, 200, original_data["bp_systolic"], key="tweaked_bp_systolic")
                tweaked_cholesterol = st.slider("Tweak Cholesterol", 100, 400, original_data["cholesterol"], key="tweaked_cholesterol")
                tweaked_smoker = st.selectbox("Tweak Smoker", options=[("No", 0), ("Yes", 1)], index=original_data["smoker"], format_func=lambda x: x[0], key="tweaked_smoker")[1]
                
                tweaked_patient_data = original_data.copy()
                tweaked_patient_data["age"] = tweaked_age
                tweaked_patient_data["bp_systolic"] = tweaked_bp_systolic
                tweaked_patient_data["cholesterol"] = tweaked_cholesterol
                tweaked_patient_data["smoker"] = tweaked_smoker
                st.json(tweaked_patient_data)
            
            what_if_button = st.button("Run What-If Analysis")

            if what_if_button:
                try:
                    what_if_response = requests.post(
                        f"{FASTAPI_URL}/what-if/{st.session_state['selected_model_id']}",
                        json={"original_data": original_data, "tweaked_data": tweaked_patient_data}
                    )
                    what_if_response.raise_for_status()
                    what_if_result = what_if_response.json()
                    st.session_state["what_if_result"] = what_if_result
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Please ensure it is running.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"Error from backend: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

        if "what_if_result" in st.session_state:
            st.subheader("What-If Comparison")
            col_orig, col_tweak = st.columns(2)
            with col_orig:
                st.write("--- Original Scenario ---")
                st.metric(
                    label=f"Diagnosis: {st.session_state['what_if_result']['original_prediction']['prediction_label']}",
                    value=f"{st.session_state['what_if_result']['original_prediction']['probability']:.2f}"
                )
                st.write("**Feature Importances:**")
                df_orig_importances = pd.DataFrame(
                    st.session_state["what_if_result"]["original_explanation"]["feature_importances"].items(),
                    columns=["Feature", "Importance"]
                ).sort_values(by="Importance", ascending=False)
                st.dataframe(df_orig_importances)
                st.write("**Local Rules:**")
                for rule in st.session_state["what_if_result"]["original_explanation"]["local_rules"]:
                    st.markdown(f"- {rule}")

            with col_tweak:
                st.write("--- Tweaked Scenario ---")
                st.metric(
                    label=f"Diagnosis: {st.session_state['what_if_result']['what_if_prediction']['prediction_label']}",
                    value=f"{st.session_state['what_if_result']['what_if_prediction']['probability']:.2f}"
                )
                st.write("**Feature Importances:**")
                df_tweak_importances = pd.DataFrame(
                    st.session_state["what_if_result"]["what_if_explanation"]["feature_importances"].items(),
                    columns=["Feature", "Importance"]
                ).sort_values(by="Importance", ascending=False)
                st.dataframe(df_tweak_importances)
                st.write("**Local Rules:**")
                for rule in st.session_state["what_if_result"]["what_if_explanation"]["local_rules"]:
                    st.markdown(f"- {rule}")

    st.markdown("---")
    st.header("Explanation Summarization (Global Insights)")
    st.write("Collect multiple explanations and summarize them for global insights.")

    if "collected_explanations" not in st.session_state:
        st.session_state["collected_explanations"] = []

    if st.button("Add Current Explanation to Summary"):
        if "explanation" in st.session_state:
            st.session_state["collected_explanations"].append(st.session_state["explanation"])
            st.success("Explanation added to summary collection!")
        else:
            st.warning("Please get a prediction and explanation first.")

    if st.session_state["collected_explanations"]:
        st.write(f"Collected {len(st.session_state['collected_explanations'])} explanations.")
        if st.button("Generate Summary of Collected Explanations"):
            try:
                summary_response = requests.post(
                    f"{FASTAPI_URL}/summarize_explanations",
                    json={"explanations": st.session_state["collected_explanations"]}
                )
                summary_response.raise_for_status()
                summary_result = summary_response.json()
                st.session_state["summary_result"] = summary_result
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Please ensure it is running.")
            except requests.exceptions.HTTPError as e:
                st.error(f"Error from backend: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        
        if "summary_result" in st.session_state:
            st.subheader("Summary Insights")
            st.write(f"**Overall Insight:** {st.session_state['summary_result']['overall_insight']}")
            st.write(f"**Common Influential Attributes:** {', '.join(st.session_state['summary_result']['common_attributes'])}")
            st.write("**Prevalent Local Rules:**")
            for rule in st.session_state['summary_result']['prevalent_rules']:
                st.markdown(f"- {rule}")