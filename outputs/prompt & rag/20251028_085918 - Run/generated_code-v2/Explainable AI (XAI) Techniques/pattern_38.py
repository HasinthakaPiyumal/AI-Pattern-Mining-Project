import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from lime import lime_tabular
import matplotlib.pyplot as plt


@st.cache_resource
def load_and_train_models():
    np.random.seed(42)
    num_samples = 1000

    data = {
        "Age": np.random.randint(20, 80, num_samples),
        "Blood_Pressure_Systolic": np.random.randint(90, 180, num_samples),
        "Blood_Pressure_Diastolic": np.random.randint(60, 120, num_samples),
        "Cholesterol": np.random.randint(150, 300, num_samples),
        "Symptom_A": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        "Symptom_B": np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
        "Symptom_C": np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        "Family_History": np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
    }
    df = pd.DataFrame(data)

    df["Diagnosis_X"] = ((df["Age"] > 50).astype(int) +
                       (df["Blood_Pressure_Systolic"] > 140).astype(int) +
                       (df["Cholesterol"] > 220).astype(int) +
                       df["Symptom_A"] + df["Symptom_B"] +
                       (df["Family_History"] == 1).astype(int) > 3).astype(int)
    
    df["Diagnosis_Y"] = ((df["Age"] < 40).astype(int) +
                         (df["Blood_Pressure_Diastolic"] < 80).astype(int) +
                         (df["Symptom_C"] == 1).astype(int) +
                         np.random.choice([0, 1], num_samples, p=[0.8, 0.2])).astype(int)
    
    X_raw = df.drop(["Diagnosis_X", "Diagnosis_Y"], axis=1)
    y1 = df["Diagnosis_X"]
    y2 = df["Diagnosis_Y"]

    feature_names = X_raw.columns.tolist()
    class_names_model1 = ["Low Risk (Disease X)", "High Risk (Disease X)"]
    class_names_model2 = ["Low Risk (Disease Y)", "High Risk (Disease Y)"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    model1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model1.fit(X_scaled, y1)

    model2 = RandomForestClassifier(n_estimators=100, random_state=24)
    model2.fit(X_scaled, y2)

    return model1, model2, scaler, feature_names, class_names_model1, class_names_model2, X_raw.values

model1, model2, scaler, feature_names, class_names_model1, class_names_model2, X_train_raw_values = load_and_train_models()

explainer1 = lime_tabular.LimeTabularExplainer(
    training_data=X_train_raw_values,
    feature_names=feature_names,
    class_names=class_names_model1,
    mode='classification'
)

explainer2 = lime_tabular.LimeTabularExplainer(
    training_data=X_train_raw_values,
    feature_names=feature_names,
    class_names=class_names_model2,
    mode='classification'
)

def get_explanation(model, explainer, scaler, instance_raw, num_features=5):
    def predict_proba_fn(raw_data_rows):
        scaled_data_rows = scaler.transform(raw_data_rows)
        return model.predict_proba(scaled_data_rows)

    predicted_probs = predict_proba_fn(instance_raw.reshape(1, -1))[0]
    top_predicted_label = np.argmax(predicted_probs)

    explanation = explainer.explain_instance(
        data_row=instance_raw,
        predict_fn=predict_proba_fn,
        num_features=num_features,
        labels=(top_predicted_label,)
    )
    return explanation.as_list()

st.set_page_config(layout="wide")
st.title("MediExplain: Interactive Diagnostic Assistant")
st.markdown("Understand, debug, and compare black-box AI diagnostic model predictions for individual patients.")

st.sidebar.header("Patient Attributes Input")

col1, col2 = st.sidebar.columns(2)
with col1:
    age = st.slider("Age", 20, 80, 45)
    bp_systolic = st.slider("Blood Pressure (Systolic)", 90, 180, 130)
    cholesterol = st.slider("Cholesterol", 150, 300, 200)
with col2:
    bp_diastolic = st.slider("Blood Pressure (Diastolic)", 60, 120, 85)
    symptom_a = st.checkbox("Symptom A Present", value=False)
    symptom_b = st.checkbox("Symptom B Present", value=False)
    symptom_c = st.checkbox("Symptom C Present", value=False)
    family_history = st.checkbox("Family History of Disease", value=False)

patient_input_raw_df = pd.DataFrame({
    "Age": [age],
    "Blood_Pressure_Systolic": [bp_systolic],
    "Blood_Pressure_Diastolic": [bp_diastolic],
    "Cholesterol": [cholesterol],
    "Symptom_A": [int(symptom_a)],
    "Symptom_B": [int(symptom_b)],
    "Symptom_C": [int(symptom_c)],
    "Family_History": [int(family_history)],
})

patient_input_raw_df = patient_input_raw_df[feature_names]
patient_instance_raw_array = patient_input_raw_df.values[0]

st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.radio("Select Diagnostic Model:", ["Disease X Risk Model", "Disease Y Risk Model"])

current_model = model1 if "Disease X" in selected_model_name else model2
current_explainer = explainer1 if "Disease X" in selected_model_name else explainer2
current_class_names = class_names_model1 if "Disease X" in selected_model_name else class_names_model2

st.header("Model Prediction & Explanation")

patient_input_scaled = scaler.transform(patient_input_raw_df)
prediction_proba = current_model.predict_proba(patient_input_scaled)[0]
predicted_class_idx = np.argmax(prediction_proba)
predicted_class_name = current_class_names[predicted_class_idx]
confidence = prediction_proba[predicted_class_idx] * 100

st.subheader(f"Prediction for {selected_model_name}:")
st.success(f"**Predicted Diagnosis:** {predicted_class_name} with **{confidence:.2f}% confidence.**")

st.subheader("Explanation (LIME - Local Feature Importance)")

explanation_list = get_explanation(current_model, current_explainer, scaler, patient_instance_raw_array)

if explanation_list:
    features = [item[0] for item in explanation_list]
    weights = [item[1] for item in explanation_list]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if w > 0 else 'red' for w in weights]
    ax.barh(features, weights, color=colors)
    ax.set_xlabel("Contribution to Prediction")
    ax.set_title(f"LIME Explanation for '{predicted_class_name}' Prediction")
    ax.invert_yaxis()
    st.pyplot(fig)

    st.markdown("---")
    st.write("### Explanation Details:")
    for feature, weight in explanation_list:
        st.write(f"- **{feature}**: {weight:.4f}")
else:
    st.info("No explanation generated.")

st.subheader("Interactive 'What-If' Analysis")
st.write("Change the patient attributes in the sidebar to see how the prediction and explanation change in real-time.")

st.subheader("User-Defined Rules/Explanation Metadata")
st.write("This section would allow medical professionals to define or view custom rules, or summarize global insights from multiple local explanations.")
st.text_area("Example User Rule Input (Placeholder):", "If Age > 60 AND Cholesterol > 250, THEN High Risk (override model).", height=100)
