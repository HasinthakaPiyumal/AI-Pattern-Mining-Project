
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils import get_lime_explanation # Assuming utils.py is in the same directory

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Oncology Diagnosis Interpretability")

# --- Load Model and Data ---
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load("cancer_prediction_model.joblib")
        feature_names = joblib.load("feature_names.joblib")
        
        # Generate a small synthetic dataset for LIME's training_data_df
        # In a real application, you'd load a representative sample of your actual training data
        np.random.seed(42)
        num_samples = 100
        data = {
            'Age': np.random.randint(20, 80, num_samples),
            'Tumor_Size_mm': np.random.uniform(5, 50, num_samples),
            'Genetic_Marker_1': np.random.randint(0, 2, num_samples),
            'Biopsy_Result_Score': np.random.uniform(1, 10, num_samples),
            'Inflammation_Level': np.random.uniform(0.1, 5.0, num_samples),
        }
        training_data_for_lime = pd.DataFrame(data)
        return model, feature_names, training_data_for_lime
    except FileNotFoundError:
        st.error("Model files not found. Please run `model_training.py` first.")
        st.stop()

model, feature_names, training_data_for_lime = load_model_and_data()
class_names = ["No Cancer", "Cancer"]

# --- Streamlit App Layout ---
st.title("🔬 Predictive Diagnosis Interpretability Platform for Oncology")
st.markdown("Understand and debug AI models for cancer prediction.")

st.sidebar.header("Patient Data Input")

# --- User Input Fields ---
input_data = {}
with st.sidebar.form("patient_input_form"):
    input_data["Age"] = st.slider("Age", 20, 80, 55)
    input_data["Tumor_Size_mm"] = st.slider("Tumor Size (mm)", 5.0, 50.0, 25.0, step=0.5)
    input_data["Genetic_Marker_1"] = st.selectbox("Genetic Marker 1", [0, 1], format_func=lambda x: "Present" if x == 1 else "Absent")
    input_data["Biopsy_Result_Score"] = st.slider("Biopsy Result Score", 1.0, 10.0, 6.5, step=0.1)
    input_data["Inflammation_Level"] = st.slider("Inflammation Level", 0.1, 5.0, 2.5, step=0.1)
    
    submit_button = st.form_submit_button("Get Prediction & Explanation")

if submit_button:
    st.subheader("Prediction Results")
    # Create a DataFrame for the single instance
    input_df = pd.DataFrame([input_data])
    
    # Ensure the order of columns matches the training data
    input_df = input_df[feature_names]
    
    prediction_proba = model.predict_proba(input_df)[0]
    predicted_class = np.argmax(prediction_proba)
    predicted_class_name = class_names[predicted_class]

    st.write(f"#### Predicted Diagnosis: **{predicted_class_name}**")
    st.write(f"Probability of No Cancer: `{prediction_proba[0]:.2f}`")
    st.write(f"Probability of Cancer: `{prediction_proba[1]:.2f}`")

    st.markdown("--- Jardar --- Jardar --- Jardar --- Jardar --- Jardar --- Jardar --- Jardar --- Jardar --- Jardar --- Jardar")
    st.subheader("Local Explanation (LIME)")
    st.write("The following explanation shows which features contributed most to this specific prediction.")

    # Generate LIME explanation
    try:
        explanation = get_lime_explanation(
            model=model,
            instance=input_df.iloc[0], # Pass as Series
            feature_names=feature_names,
            class_names=class_names,
            training_data_df=training_data_for_lime
        )
        
        # Display LIME explanation
        # LIME's as_html() generates HTML, which Streamlit can render
        st.components.v1.html(explanation.as_html(), height=600, scrolling=True)

        st.subheader("Explanation Details:")
        # You can also iterate through the explanation.as_list() for a more controlled display
        st.markdown("**Top features contributing to the prediction:**")
        for feature, weight in explanation.as_list():
            st.write(f"- {feature}: {weight:.4f}")

    except Exception as e:
        st.error(f"Error generating LIME explanation: {e}")
        st.info("LIME requires a representative sample of training data to work effectively. "
                "Ensure `training_data_for_lime` is populated correctly.")

st.markdown("--- Jardar --- Jardar --- Jardar --- Jardar --- Jardar --- Jardar --- Jardar --- Jardar --- Jardar --- Jardar")
st.subheader("Global Interpretability & Debugging Tools (Coming Soon!)")
st.write("This section will feature advanced tools for global feature importance, partial dependence plots, "
         "subgroup analysis (e.g., DivExplorer), and counterfactual explanations to provide deeper insights "
         "into the model's overall behavior and performance across different patient cohorts.")

st.markdown("**Future Enhancements:**")
st.markdown("- **Global Feature Importance:** Understand which features are generally most important across all predictions.")
st.markdown("- **Partial Dependence Plots (PDP) / ICE Plots:** Visualize the marginal effect of one or two features on the predicted outcome.")
st.markdown("- **Subgroup Analysis (DivExplorer-like):** Identify and explore data subgroups where the model performs differently or makes divergent predictions.")
st.markdown("- **Counterfactual Explanations:** Discover the smallest changes to input features that would flip a prediction.")
st.markdown("- **Inherently Interpretable Models:** Where suitable, integrate models like Decision Trees or Rule-based systems.")
