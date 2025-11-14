
import pandas as pd
import numpy as np
import joblib
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Interpretability Libraries
import lime
import lime.lime_tabular
import shap

# FastAPI for Backend
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Streamlit for Frontend
import streamlit as st
import requests # To make requests from Streamlit to FastAPI
import json # For handling JSON responses

# --- Configuration --- #
FASTAPI_HOST = "localhost"
FASTAPI_PORT = 8000
FASTAPI_URL = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"

# --- 1. Data Layer & Model Training (Dummy Data for Demonstration) --- #
def generate_dummy_data():
    np.random.seed(42)
    data_size = 1000
    
    # Features (simplified medical data)
    age = np.random.randint(20, 80, data_size)
    gender = np.random.choice(["Male", "Female"], data_size, p=[0.55, 0.45])
    cholesterol = np.random.randint(150, 250, data_size)
    blood_pressure = np.random.randint(90, 180, data_size)
    smoker = np.random.choice([0, 1], data_size, p=[0.7, 0.3])
    exercise_hours = np.random.randint(1, 10, data_size)
    
    # Target variable: Disease (0 = No Disease, 1 = Disease)
    # Make it somewhat dependent on features for a realistic model
    disease = (0.2 * age + 0.1 * cholesterol + 0.3 * blood_pressure + 
               0.5 * smoker - 0.1 * exercise_hours + 
               np.random.normal(0, 50, data_size)) > 250
    disease = disease.astype(int)

    df = pd.DataFrame({"Age": age, "Gender": gender, "Cholesterol": cholesterol,
                       "Blood_Pressure": blood_pressure, "Smoker": smoker,
                       "Exercise_Hours": exercise_hours, "Disease": disease})
    return df

def preprocess_data(df):
    df_processed = df.copy()
    
    # Encode categorical features
    gender_encoder = LabelEncoder()
    df_processed["Gender_encoded"] = gender_encoder.fit_transform(df_processed["Gender"])
    
    feature_cols = ["Age", "Gender_encoded", "Cholesterol", "Blood_Pressure", "Smoker", "Exercise_Hours"]
    X = df_processed[feature_cols]
    y = df_processed["Disease"]
    
    return X, y, feature_cols, {"Gender": gender_encoder}

def train_and_save_model(X, y, feature_cols):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "diagnostic_model.joblib")
    joblib.dump(feature_cols, "feature_cols.joblib")
    return model

# Generate and prepare data, train model if not already done
try:
    model = joblib.load("diagnostic_model.joblib")
    feature_names = joblib.load("feature_cols.joblib")
    print("Model and feature columns loaded.")
except FileNotFoundError:
    print("Training new model...")
    data = generate_dummy_data()
    X, y, feature_names, encoders = preprocess_data(data)
    model = train_and_save_model(X, y, feature_names)
    print("Model trained and saved.")

# We need a global X_train for interpretability methods (e.g., LIME background data)
dummy_data_full = generate_dummy_data()
X_processed_full, _, _, _ = preprocess_data(dummy_data_full)
X_train_global = X_processed_full
class_names = ["No Disease", "Disease"]

# --- 4. Backend Layer (FastAPI) --- #
app = FastAPI(
    title="Healthcare Diagnostic AI Interpretability Platform Backend",
    description="API for medical AI model predictions and interpretability explanations."
)

# Pydantic models for request/response bodies
class PatientData(BaseModel):
    age: int
    gender: str  # "Male" or "Female"
    cholesterol: int
    blood_pressure: int
    smoker: int  # 0 or 1
    exercise_hours: int

class PredictionResponse(BaseModel):
    prediction: str
    probability: float

class LIMEExplanation(BaseModel):
    feature_contributions: dict
    fidelity: float

class SHAPExplanation(BaseModel):
    shap_values: list
    expected_value: float

class CounterfactualExplanation(BaseModel):
    original_prediction: str
    counterfactual_instance: dict
    counterfactual_prediction: str
    description: str

class PlotData(BaseModel):
    plot_base64: str
    description: str

def _encode_plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig) # Close the plot to free memory
    return img_base64

# Helper to preprocess single patient data consistent with training
def _preprocess_single_patient(patient_data: PatientData):
    # Re-create the encoder or use a fixed one if saved. For this demo, let's assume
    # 'Gender' is encoded to 0 for 'Female' and 1 for 'Male' based on initial data.
    gender_map = {"Female": 0, "Male": 1}
    gender_encoded = gender_map.get(patient_data.gender, 0) # Default to 0 if unknown

    patient_features = np.array([
        patient_data.age,
        gender_encoded,
        patient_data.cholesterol,
        patient_data.blood_pressure,
        patient_data.smoker,
        patient_data.exercise_hours
    ]).reshape(1, -1)
    
    # Ensure feature order matches trained model's feature_names
    # (Already handled by fixed order above, but good to note for more complex cases)
    
    return patient_features


@app.post("/predict", response_model=PredictionResponse)
async def predict_diagnosis(patient_data: PatientData):
    processed_data = _preprocess_single_patient(patient_data)
    prediction_proba = model.predict_proba(processed_data)[0]
    predicted_class_idx = np.argmax(prediction_proba)
    
    return PredictionResponse(
        prediction=class_names[predicted_class_idx],
        probability=prediction_proba[predicted_class_idx]
    )

@app.post("/explain/lime", response_model=LIMEExplanation)
async def explain_lime(patient_data: PatientData):
    processed_data = _preprocess_single_patient(patient_data)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_global.values,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

    explanation = explainer.explain_instance(
        data_row=processed_data[0],
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    
    return LIMEExplanation(
        feature_contributions=dict(explanation.as_list()),
        fidelity=explanation.score
    )

@app.post("/explain/shap", response_model=SHAPExplanation)
async def explain_shap(patient_data: PatientData):
    processed_data = _preprocess_single_patient(patient_data)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(processed_data)
    
    # For binary classification, shap_values is a list of arrays for each class
    # We'll return the shap values for the positive class (Disease)
    if isinstance(shap_values, list):
        shap_values_for_positive_class = shap_values[1][0].tolist() # Assuming binary and class 1 is positive
    else:
        shap_values_for_positive_class = shap_values[0].tolist()

    # Calculate expected value (base value for shap)
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1] # For positive class

    return SHAPExplanation(
        shap_values=shap_values_for_positive_class,
        expected_value=expected_value
    )

@app.post("/explain/pdp", response_model=PlotData)
async def explain_pdp(feature_name: str):
    if feature_name not in feature_names:
        raise HTTPException(status_code=400, detail=f"Feature '{feature_name}' not found.")

    # Create a figure and axes object
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # PartialDependenceDisplay works with trained model, X data, and features
    # It expects features as a list of feature names or indices
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_train_global, # Use the global training data for PDP calculation
        features=[feature_name], # Can be a list of features or a single feature
        feature_names=feature_names,
        target=1, # For positive class (Disease)
        kind="average", # For PDP
        ax=ax
    )
    ax.set_title(f"Partial Dependence Plot for {feature_name}")
    ax.set_ylabel("Partial Dependence")
    ax.set_xlabel(feature_name)

    img_base64 = _encode_plot_to_base64(fig)
    return PlotData(plot_base64=img_base64, description=f"Partial Dependence Plot for {feature_name}")


@app.post("/explain/ice", response_model=PlotData)
async def explain_ice(feature_name: str):
    if feature_name not in feature_names:
        raise HTTPException(status_code=400, detail=f"Feature '{feature_name}' not found.")

    fig, ax = plt.subplots(figsize=(8, 6))

    PartialDependenceDisplay.from_estimator(
        model,
        X_train_global, # Use global training data
        features=[feature_name],
        feature_names=feature_names,
        target=1, # For positive class
        kind="individual", # For ICE plots
        ax=ax, 
        random_state=42 # for reproducibility if sampling is involved
    )
    ax.set_title(f"Individual Conditional Expectation (ICE) Plot for {feature_name}")
    ax.set_ylabel("Partial Dependence (ICE)")
    ax.set_xlabel(feature_name)

    img_base64 = _encode_plot_to_base64(fig)
    return PlotData(plot_base64=img_base64, description=f"Individual Conditional Expectation (ICE) Plot for {feature_name}")


@app.post("/explain/pfi", response_model=PlotData)
async def explain_pfi():
    # Calculate permutation importance on the global training data
    result = permutation_importance(model, X_train_global, y, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(feature_names)[sorted_idx])
    ax.set_title("Permutation Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    img_base64 = _encode_plot_to_base64(fig)
    return PlotData(plot_base64=img_base64, description="Permutation Feature Importance Plot")


@app.post("/explain/counterfactual", response_model=CounterfactualExplanation)
async def explain_counterfactual(patient_data: PatientData):
    # This is a simplified/placeholder implementation for counterfactuals.
    # A full integration with `dice_ml` would be more complex and require
    # setting up `dice_ml.Data` and `dice_ml.Model` objects.
    # For demonstration, we'll simulate a simple counterfactual change.

    original_processed_data = _preprocess_single_patient(patient_data)
    original_prediction_proba = model.predict_proba(original_processed_data)[0]
    original_predicted_class_idx = np.argmax(original_prediction_proba)
    original_prediction_label = class_names[original_predicted_class_idx]

    # Simulate a counterfactual: e.g., if smoker status changes
    counterfactual_patient_data = patient_data.copy()
    
    # Attempt to change the prediction by altering a key feature (e.g., Smoker status)
    # If currently a smoker (1), try changing to non-smoker (0)
    # If currently non-smoker (0), try changing to smoker (1) - to see if it causes disease
    changed_feature = None
    if patient_data.smoker == 1: # If currently a smoker
        counterfactual_patient_data.smoker = 0
        changed_feature = "Smoker (from Yes to No)"
    elif patient_data.smoker == 0 and original_prediction_label == "No Disease":
        # If not a smoker and no disease, maybe becoming a smoker causes disease
        counterfactual_patient_data.smoker = 1
        changed_feature = "Smoker (from No to Yes)"
    else:
        # If smoker status change isn't impactful enough, try exercise_hours
        # Increase exercise to see if it reverses disease
        if original_prediction_label == "Disease":
            counterfactual_patient_data.exercise_hours += 5 # More exercise
            changed_feature = "Exercise_Hours (increased by 5)"
        else: # If no disease, maybe less exercise causes it
            counterfactual_patient_data.exercise_hours = max(1, counterfactual_patient_data.exercise_hours - 3)
            changed_feature = "Exercise_Hours (decreased by 3)"


    cf_processed_data = _preprocess_single_patient(counterfactual_patient_data)
    cf_prediction_proba = model.predict_proba(cf_processed_data)[0]
    cf_predicted_class_idx = np.argmax(cf_prediction_proba)
    cf_prediction_label = class_names[cf_predicted_class_idx]

    description_text = (
        f"Original prediction: {original_prediction_label}. "
        f"If {changed_feature}, prediction changes to: {cf_prediction_label}. "
        f"This is a simplified counterfactual; a full implementation would use `dice_ml` for diverse and actionable counterfactuals."
    )

    return CounterfactualExplanation(
        original_prediction=original_prediction_label,
        counterfactual_instance=counterfactual_patient_data.dict(),
        counterfactual_prediction=cf_prediction_label,
        description=description_text
    )


# --- 5. Frontend Layer (Streamlit Application) --- #
# This section will only run when executed with `streamlit run your_script_name.py`

def main_streamlit_app():
    st.set_page_config(layout="wide", page_title="Healthcare AI Interpretability")
    st.title("🩺 Healthcare Diagnostic AI Interpretability Platform")

    st.sidebar.header("Patient Data Input")

    with st.sidebar.form("patient_form"):
        age = st.slider("Age", 20, 80, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 300, 200)
        blood_pressure = st.slider("Blood Pressure (mmHg)", 80, 200, 120)
        smoker = st.selectbox("Smoker", {0: "No", 1: "Yes"}, format_func=lambda x: "Yes" if x == 1 else "No")
        exercise_hours = st.slider("Exercise Hours/Week", 0, 20, 5)
        
        submit_button = st.form_submit_button("Get Prediction & Explanations")

    patient_data_input = {
        "age": age,
        "gender": gender,
        "cholesterol": cholesterol,
        "blood_pressure": blood_pressure,
        "smoker": smoker,
        "exercise_hours": exercise_hours
    }

    if submit_button:
        st.subheader("Model Prediction")
        try:
            response = requests.post(f"{FASTAPI_URL}/predict", json=patient_data_input)
            response.raise_for_status() # Raise an exception for HTTP errors
            prediction_result = response.json()
            st.success(f"Predicted Diagnosis: **{prediction_result['prediction']}** "
                       f"(Probability: {prediction_result['probability']:.2f})")
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to FastAPI backend at {FASTAPI_URL}. Please ensure the backend is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling prediction API: {e}")

        st.markdown("--- Jardar --- Jardar --- Jardar ---")

        st.subheader("Interpretability Explanations")
        
        # Tabs for different interpretability methods
        tab_lime, tab_shap, tab_pdp, tab_ice, tab_pfi, tab_cf = st.tabs([
            "LIME (Local)", "SHAP (Local)", "PDP (Global)", "ICE (Local)", "PFI (Global)", "Counterfactuals (Local)"
        ])

        with tab_lime:
            st.write("### LIME Explanation")
            st.info("LIME provides local, instance-specific explanations by approximating the black-box model locally with an interpretable model.")
            try:
                response = requests.post(f"{FASTAPI_URL}/explain/lime", json=patient_data_input)
                response.raise_for_status()
                lime_exp = response.json()
                st.write("**Feature Contributions:**")
                st.json(lime_exp["feature_contributions"])
                st.write(f"**Model Fidelity (R-squared of local approximation):** {lime_exp['fidelity']:.3f}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error getting LIME explanation: {e}")

        with tab_shap:
            st.write("### SHAP Explanation")
            st.info("SHAP (SHapley Additive exPlanations) attributes the contribution of each feature to the prediction for a specific instance.")
            try:
                response = requests.post(f"{FASTAPI_URL}/explain/shap", json=patient_data_input)
                response.raise_for_status()
                shap_exp = response.json()
                st.write("**SHAP Values (for 'Disease' class):**")
                # Display SHAP values for each feature
                shap_df = pd.DataFrame({"Feature": feature_names, "SHAP Value": shap_exp["shap_values"]})
                shap_df["Absolute SHAP Value"] = shap_df["SHAP Value"].abs()
                shap_df = shap_df.sort_values(by="Absolute SHAP Value", ascending=False).drop(columns="Absolute SHAP Value")
                st.table(shap_df)
                st.write(f"**Base Value (Expected Value):** {shap_exp['expected_value']:.3f}")
                st.write("Positive SHAP values indicate features pushing towards 'Disease', negative towards 'No Disease'.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error getting SHAP explanation: {e}")

        with tab_pdp:
            st.write("### Partial Dependence Plots (PDP)")
            st.info("PDPs show the average marginal effect of one or two features on the predicted outcome of a model. Helps understand global relationships.")
            selected_pdp_feature = st.selectbox("Select Feature for PDP", feature_names, key="pdp_feature_select")
            if st.button(f"Generate PDP for {selected_pdp_feature}", key="generate_pdp_btn"):
                try:
                    response = requests.post(f"{FASTAPI_URL}/explain/pdp?feature_name={selected_pdp_feature}")
                    response.raise_for_status()
                    plot_data = response.json()
                    st.image(base64.b64decode(plot_data["plot_base64"]), caption=plot_data["description"], use_column_width=True)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error getting PDP: {e}")

        with tab_ice:
            st.write("### Individual Conditional Expectation (ICE) Plots")
            st.info("ICE plots display the prediction for each instance as a function of a feature, allowing to identify heterogeneous relationships that PDPs might obscure.")
            selected_ice_feature = st.selectbox("Select Feature for ICE", feature_names, key="ice_feature_select")
            if st.button(f"Generate ICE for {selected_ice_feature}", key="generate_ice_btn"):
                try:
                    response = requests.post(f"{FASTAPI_URL}/explain/ice?feature_name={selected_ice_feature}")
                    response.raise_for_status()
                    plot_data = response.json()
                    st.image(base64.b64decode(plot_data["plot_base64"]), caption=plot_data["description"], use_column_width=True)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error getting ICE: {e}")

        with tab_pfi:
            st.write("### Permutation Feature Importance (PFI)")
            st.info("PFI measures the increase in prediction error after permuting a feature's values, showing how much the model relies on that feature for its predictions.")
            if st.button("Generate PFI Plot", key="generate_pfi_btn"):
                try:
                    response = requests.post(f"{FASTAPI_URL}/explain/pfi")
                    response.raise_for_status()
                    plot_data = response.json()
                    st.image(base64.b64decode(plot_data["plot_base64"]), caption=plot_data["description"], use_column_width=True)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error getting PFI: {e}")
        
        with tab_cf:
            st.write("### Counterfactual Explanations")
            st.info("Counterfactual explanations show the smallest change to an instance's features that would change the prediction to a desired outcome. 'What if' scenarios.")
            st.warning("**Note:** This is a simplified counterfactual simulation for demonstration. A full implementation would use libraries like `dice_ml` to generate diverse and actionable counterfactuals more robustly.")
            if st.button("Generate Counterfactual Explanation", key="generate_cf_btn"):
                try:
                    response = requests.post(f"{FASTAPI_URL}/explain/counterfactual", json=patient_data_input)
                    response.raise_for_status()
                    cf_exp = response.json()
                    st.write(f"**Original Prediction:** {cf_exp['original_prediction']}")
                    st.write("**Counterfactual Instance:**")
                    st.json(cf_exp['counterfactual_instance'])
                    st.write(f"**Counterfactual Prediction:** {cf_exp['counterfactual_prediction']}")
                    st.write(f"**Description:** {cf_exp['description']}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error getting Counterfactual Explanation: {e}")

# --- Execution Logic --- #
if __name__ == "__main__":
    # This block allows running either FastAPI or Streamlit depending on how the script is invoked.
    # To run FastAPI: uvicorn healthcare_interpretability_platform:app --reload --host 0.0.0.0 --port 8000
    # To run Streamlit: streamlit run healthcare_interpretability_platform.py
    
    # Streamlit will be executed if 'streamlit run' is used
    # If the script is run directly (e.g., `python healthcare_interpretability_platform.py`), it does nothing or you could add a default message.
    
    # Check if the script is being run by streamlit
    if "streamlit" in st.__dict__:
         main_streamlit_app()
    else:
        import uvicorn
        st.write("This script contains both a FastAPI backend and a Streamlit frontend.")
        st.write(f"To run the FastAPI backend: `uvicorn healthcare_interpretability_platform:app --reload --host {FASTAPI_HOST} --port {FASTAPI_PORT}`")
        st.write("To run the Streamlit frontend: `streamlit run healthcare_interpretability_platform.py`")
        st.write("Please ensure the FastAPI backend is running before starting the Streamlit frontend.")

