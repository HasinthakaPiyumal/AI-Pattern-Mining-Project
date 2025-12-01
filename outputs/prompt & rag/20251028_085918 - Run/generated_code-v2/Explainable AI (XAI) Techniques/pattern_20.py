
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# --- 1. Data Layer: Dummy Data Generation and Preprocessing ---

@st.cache_data
def load_and_preprocess_data():
    # Generate synthetic patient data
    np.random.seed(42)
    num_samples = 1000
    data = {
        'Age': np.random.randint(20, 80, num_samples),
        'Fever': np.random.randint(0, 2, num_samples), # 0: No, 1: Yes
        'Cough': np.random.randint(0, 2, num_samples),
        'Fatigue': np.random.randint(0, 2, num_samples),
        'Headache': np.random.randint(0, 2, num_samples),
        'SoreThroat': np.random.randint(0, 2, num_samples),
        'MusclePain': np.random.randint(0, 2, num_samples),
        'Diagnosis': np.random.choice(['Flu', 'Cold', 'Allergy'], num_samples, p=[0.4, 0.4, 0.2])
    }
    df = pd.DataFrame(data)

    # Simulate some correlations for a more meaningful model
    df.loc[df['Fever'] == 1, 'Diagnosis'] = np.random.choice(['Flu', 'Cold'], sum(df['Fever'] == 1), p=[0.7, 0.3])
    df.loc[df['Cough'] == 1, 'Diagnosis'] = np.random.choice(['Flu', 'Cold', 'Allergy'], sum(df['Cough'] == 1), p=[0.5, 0.3, 0.2])
    df.loc[(df['Fever'] == 0) & (df['Cough'] == 0), 'Diagnosis'] = np.random.choice(['Allergy', 'Cold'], sum((df['Fever'] == 0) & (df['Cough'] == 0)), p=[0.7, 0.3])

    # Encode target variable
    df['Diagnosis_encoded'] = df['Diagnosis'].astype('category').cat.codes
    feature_cols = ['Age', 'Fever', 'Cough', 'Fatigue', 'Headache', 'SoreThroat', 'MusclePain']
    X = df[feature_cols]
    y = df['Diagnosis_encoded']
    class_names = df['Diagnosis'].astype('category').cat.categories.tolist()

    return X, y, feature_cols, class_names, df['Diagnosis'].astype('category').cat.codes.unique()

X, y, feature_cols, class_names, unique_diagnosis_codes = load_and_preprocess_data()

# --- 2. Backend: AI Model & XAI Explanation Engine ---

# Split data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_model(X_train_data, y_train_data):
    # Train a RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_data, y_train_data)
    return model

model = train_model(X_train, y_train)

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_cols,
    class_names=class_names,
    mode='classification'
)

# --- 3. Frontend: Interactive User Interface (Streamlit Application) ---

st.set_page_config(layout="wide", page_title="xPlain: Clinical Diagnosis Explainer")
st.title("🩺 xPlain: Interactive Clinical Diagnosis Explainer")
st.markdown("Understand and debug black-box disease predictions with explanations.")

st.sidebar.header("Patient Data Input")

# Input widgets for patient features
input_age = st.sidebar.slider("Age", min_value=1, max_value=100, value=35)
input_fever = st.sidebar.selectbox("Fever", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
input_cough = st.sidebar.selectbox("Cough", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
input_fatigue = st.sidebar.selectbox("Fatigue", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
input_headache = st.sidebar.selectbox("Headache", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
input_sorethroat = st.sidebar.selectbox("Sore Throat", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
input_musclepain = st.sidebar.selectbox("Muscle Pain", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)

# Create a DataFrame for the current input
input_data = pd.DataFrame([[input_age, input_fever, input_cough, input_fatigue, input_headache, input_sorethroat, input_musclepain]],
                            columns=feature_cols)

st.sidebar.subheader("Action")
if st.sidebar.button("Diagnose & Explain"): # Explicit button for better control in what-if
    st.subheader("Diagnosis Results")

    # Make prediction
    prediction_proba = model.predict_proba(input_data)[0]
    predicted_class_idx = np.argmax(prediction_proba)
    predicted_diagnosis = class_names[predicted_class_idx]

    st.success(f"Predicted Diagnosis: **{predicted_diagnosis}** (Confidence: {prediction_proba[predicted_class_idx]*100:.2f}%) ")

    st.subheader("Explanation for the Prediction (LIME)")

    # Generate LIME explanation
    # We explain the prediction for the predicted class
    explanation = explainer.explain_instance(
        data_row=input_data.values[0],
        predict_fn=model.predict_proba,
        num_features=len(feature_cols),
        num_samples=1000 # Number of perturbations
    )

    # Visualize explanation
    fig = explanation.as_pyplot_figure()
    st.pyplot(fig)

    st.markdown("**How to interpret:** The plot shows which features (e.g., 'Fever', 'Age') contribute positively (towards the predicted class) or negatively (away from the predicted class) to the model's decision.")

    # Display raw explanation components for detail
    st.subheader("Detailed Explanation Components")
    exp_list = explanation.as_list()
    for feature, weight in exp_list:
        st.write(f"- **{feature}**: {weight:.4f}")

    st.markdown("### What-If Analysis")
    st.info("Modify the patient attributes in the sidebar and click 'Diagnose & Explain' again to see how the prediction and explanation change.")

else:
    st.info("Enter patient details in the sidebar and click 'Diagnose & Explain' to get a prediction and its explanation.")

