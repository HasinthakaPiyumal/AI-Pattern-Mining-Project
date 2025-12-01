import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Mock Black-box Medical Diagnosis Model ---
# In a real scenario, this would be a pre-trained, complex model.
class MockMedicalDiagnosisModel:
    def __init__(self):
        # Simulate a dataset for training
        np.random.seed(42)
        self.features = ['Age', 'Fever', 'Cough', 'Fatigue', 'Blood_Pressure', 'Cholesterol']
        self.X = pd.DataFrame({
            'Age': np.random.randint(20, 80, 100),
            'Fever': np.random.randint(0, 2, 100), # 0: No, 1: Yes
            'Cough': np.random.randint(0, 2, 100),
            'Fatigue': np.random.randint(0, 2, 100),
            'Blood_Pressure': np.random.randint(90, 180, 100),
            'Cholesterol': np.random.randint(150, 250, 100),
        })
        # Simulate a target variable (e.g., 'Disease_X_Positive')
        self.y = np.random.randint(0, 2, 100) # 0: Negative, 1: Positive

        # Introduce some correlation to make the model somewhat meaningful
        self.y = np.where((self.X['Fever'] == 1) & (self.X['Cough'] == 1) & (self.X['Age'] > 50), 1, self.y)
        self.y = np.where((self.X['Cholesterol'] > 200) & (self.X['Blood_Pressure'] > 140), 1, self.y)

        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X, self.y)

    def predict_proba(self, patient_data):
        # patient_data is a dictionary/DataFrame row
        # Ensure the order of features is consistent
        patient_df = pd.DataFrame([patient_data], columns=self.features)
        return self.model.predict_proba(patient_df)[0]

# --- Mock LACE Explainer --- 
# Simulates LACE by providing feature contributions based on a simple heuristic
# In a real LACE implementation, this would involve a more sophisticated algorithm.
class MockLACEExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def explain_instance(self, instance, target_class=1): # Explaining for 'Disease_X_Positive'
        # Simple heuristic: Features with values strongly associated with the target class
        # will have higher 'contributions'. This is a simplified proxy for LACE.
        explanation = {}
        # For demonstration, we'll just say higher values of certain features
        # or presence of binary features contribute more to the positive class.
        
        # Example of a very basic heuristic for demonstration purposes:
        if instance['Fever'] == 1:
            explanation['Fever'] = 'Presence of fever significantly contributes to positive diagnosis.'
        if instance['Cough'] == 1:
            explanation['Couch'] = 'Presence of cough contributes to positive diagnosis.'
        if instance['Age'] > 60:
            explanation['Age'] = 'Higher age is a contributing factor.'
        if instance['Blood_Pressure'] > 140:
            explanation['Blood_Pressure'] = 'Elevated blood pressure is a factor.'
        if instance['Cholesterol'] > 220:
            explanation['Cholesterol'] = 'High cholesterol levels are a factor.'
        
        # If no specific rule triggered, provide a generic message
        if not explanation:
            explanation['General'] = 'No strong individual factors identified for this instance based on current rules.'
            
        return explanation

# --- Streamlit UI --- 
st.set_page_config(layout="wide", page_title="xPlain for Medical Diagnosis")

st.title("xPlain for Medical Diagnosis")
st.markdown("### Interactive Human-in-the-Loop Explanation Framework")

# Initialize model and explainer
@st.cache_resource
def load_resources():
    model = MockMedicalDiagnosisModel()
    explainer = MockLACEExplainer(model, model.features)
    return model, explainer

medical_model, lace_explainer = load_resources()

st.sidebar.header("Patient Data Input")

# Input widgets for patient attributes
patient_input = {}
patient_input['Age'] = st.sidebar.slider('Age', 20, 90, 55)
patient_input['Fever'] = st.sidebar.selectbox('Fever', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', index=0)
patient_input['Cough'] = st.sidebar.selectbox('Cough', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', index=0)
patient_input['Fatigue'] = st.sidebar.selectbox('Fatigue', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', index=0)
patient_input['Blood_Pressure'] = st.sidebar.slider('Blood Pressure (mmHg)', 80, 200, 120)
patient_input['Cholesterol'] = st.sidebar.slider('Cholesterol (mg/dL)', 100, 300, 180)

st.subheader("Diagnosis and Explanation")

# Make prediction
patient_data_for_prediction = pd.DataFrame([patient_input])
prediction_proba = medical_model.predict_proba(patient_input)

# Display prediction
st.write(f"#### Likelihood of Disease X: {prediction_proba[1]*100:.2f}%")
st.progress(prediction_proba[1], text=f"Disease X Probability: {prediction_proba[1]*100:.2f}%")

st.write("### LACE Explanation for this Diagnosis")
explanation = lace_explainer.explain_instance(patient_input)

if explanation:
    for feature, desc in explanation.items():
        st.markdown(f"- **{feature}**: {desc}")
else:
    st.info("No specific contributing factors identified by the explainer for this instance.")

st.markdown("___")
st.subheader("Interactive Features (Placeholders)")
st.info("This section would house features like 'Comparison across models/diseases', 'What-if Analysis (beyond basic input tweaks)', and 'User-defined Rules'.")

st.markdown("___")
st.subheader("Explanation Metadata (Global Insights - Placeholder)")
st.info("In a full implementation, this section would summarize multiple local explanations into global insights (e.g., most common contributing factors, population-level risk rules).")

st.markdown("\n\nTo run this app: `streamlit run xplain_medical_diagnosis_app.py`")