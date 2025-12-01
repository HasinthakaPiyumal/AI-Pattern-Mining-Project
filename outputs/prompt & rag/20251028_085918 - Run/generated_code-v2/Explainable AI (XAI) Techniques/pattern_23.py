import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
import joblib
import os

# --- 1. Data Ingestion and Preprocessing Module ---
def load_and_preprocess_data(df):
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df[col].replace(0, df[col].median())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled_df, y, scaler, X.columns.tolist()

# --- 2. Model Training Module ---
def train_model(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

# --- 3. Prediction and Interpretation Module ---
def get_decision_path_explanation(model, sample_scaled_df, sample_original_df, feature_names):
    node_indicator = model.decision_path(sample_scaled_df)
    leaf_id = model.apply(sample_scaled_df)

    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    explanation_rules = []
    for i, node_id in enumerate(node_index):
        if node_id == leaf_id[0]:
            break

        feature_idx = model.tree_.feature[node_id]
        threshold_scaled = model.tree_.threshold[node_id]
        feature_name = feature_names[feature_idx]

        original_value = sample_original_df.iloc[0][feature_name]
        scaled_value = sample_scaled_df.iloc[0, feature_idx]

        next_node_in_path = node_index[i + 1] if (i + 1) < len(node_index) else -1

        rule_text = ""
        if next_node_in_path == model.tree_.children_left[node_id]:
            rule_text = f"{feature_name} (patient value: {original_value:.2f}) is considered relatively low based on a threshold."
        elif next_node_in_path == model.tree_.children_right[node_id]:
            rule_text = f"{feature_name} (patient value: {original_value:.2f}) is considered relatively high based on a threshold."
        else:
            rule_text = f"Decision related to {feature_name} (patient value: {original_value:.2f})."

        explanation_rules.append(rule_text)
    return explanation_rules[:5]

def predict_and_explain(model, scaler, patient_data_dict, feature_names):
    patient_df_original = pd.DataFrame([patient_data_dict])
    patient_df_original = patient_df_original[feature_names]

    patient_df_scaled = scaler.transform(patient_df_original)
    patient_df_scaled = pd.DataFrame(patient_df_scaled, columns=feature_names)

    prediction = model.predict(patient_df_scaled)[0]
    prediction_proba = model.predict_proba(patient_df_scaled)[0]
    diabetes_probability = prediction_proba[1]

    risk_category = ""
    if diabetes_probability >= 0.7:
        risk_category = "High Risk"
    elif diabetes_probability >= 0.4:
        risk_category = "Medium Risk"
    else:
        risk_category = "Low Risk"

    explanation_rules = get_decision_path_explanation(model, patient_df_scaled, patient_df_original, feature_names)

    explanation = "The model predicted " + risk_category.lower() + " with a probability of " \
                  f"{diabetes_probability:.2f} for diabetes. This decision was primarily influenced by the following factors:\n"
    if explanation_rules:
        for i, rule in enumerate(explanation_rules):
            explanation += f"- {rule}\n"
    else:
        explanation += "- The model reached a decision with minimal splits for this patient, indicating that their profile does not strongly align with a particular risk factor pattern."

    return risk_category, explanation, diabetes_probability

# --- 4. User Interface (UI) Module with Streamlit ---

def main():
    st.set_page_config(page_title="Diabetes Risk Predictor (Interpretable AI)", layout="wide")

    st.title("🩺 Interpretable Diabetes Risk Predictor")
    st.markdown("""
    This tool helps general practitioners assess early diabetes risk using an interpretable AI model.
    Enter patient's anonymized health data to get a risk prediction and a clear explanation of the contributing factors.
    """)

    model_filename = "diabetes_dt_model.joblib"
    scaler_filename = "diabetes_scaler.joblib"
    feature_names_filename = "diabetes_feature_names.joblib"

    model = None
    scaler = None
    feature_names = None

    if os.path.exists(model_filename) and os.path.exists(scaler_filename) and os.path.exists(feature_names_filename):
        try:
            model = joblib.load(model_filename)
            scaler = joblib.load(scaler_filename)
            feature_names = joblib.load(feature_names_filename)
            st.sidebar.success("Pre-trained model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}. Training a new model.")
            model = None
    if model is None:
        st.sidebar.warning("Model not found or failed to load. Training a new model with synthetic data...")
        np.random.seed(42)
        n_samples = 1000
        synthetic_data = {
            "Pregnancies": np.random.randint(0, 17, n_samples),
            "Glucose": np.random.normal(120, 30, n_samples),
            "BloodPressure": np.random.normal(70, 15, n_samples),
            "SkinThickness": np.random.normal(25, 10, n_samples),
            "Insulin": np.random.normal(120, 100, n_samples),
            "BMI": np.random.normal(30, 8, n_samples),
            "DiabetesPedigreeFunction": np.random.normal(0.5, 0.3, n_samples),
            "Age": np.random.randint(21, 80, n_samples),
            "Outcome": np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        synthetic_df = pd.DataFrame(synthetic_data)
        synthetic_df.loc[synthetic_df["Glucose"] > 140, "Outcome"] = 1
        synthetic_df.loc[synthetic_df["BMI"] > 30, "Outcome"] = 1
        synthetic_df.loc[synthetic_df["Age"] > 45, "Outcome"] = 1
        synthetic_df["Outcome"] = synthetic_df.apply(
            lambda row: 1 if (row["Glucose"] > 140 or row["BMI"] > 30 or row["Age"] > 45 or row["Outcome"] == 1) else 0, axis=1
        )
        synthetic_df["Outcome"] = synthetic_df["Outcome"].clip(0, 1).astype(int)

        X_scaled, y, scaler, feature_names = load_and_preprocess_data(synthetic_df)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        model = train_model(X_train, y_train)

        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)
        joblib.dump(feature_names, feature_names_filename)
        st.sidebar.success("Model trained and saved successfully with synthetic data!")


    st.sidebar.header("Patient Data Input")

    with st.sidebar.form("patient_input_form"):
        pregnancies = st.slider("Pregnancies", 0, 17, 1)
        glucose = st.number_input("Glucose (mg/dL)", 0, 200, 100, 1)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 122, 70, 1)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 99, 20, 1)
        insulin = st.number_input("Insulin (muU/ml)", 0, 846, 50, 1)
        bmi = st.number_input("BMI (kg/m^2)", 0.0, 67.1, 22.0, 0.1)
        diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", 0.078, 2.42, 0.2, 0.001)
        age = st.slider("Age", 21, 81, 25)

        submitted = st.form_submit_button("Assess Risk")

    if submitted:
        patient_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree_function,
            "Age": age
        }

        risk_category, explanation, probability = predict_and_explain(model, scaler, patient_data, feature_names)

        st.subheader("📊 Risk Assessment Result")

        if risk_category == "High Risk":
            st.error(f"**Patient is classified as: {risk_category}** (Probability of Diabetes: {probability:.2f})")
        elif risk_category == "Medium Risk":
            st.warning(f"**Patient is classified as: {risk_category}** (Probability of Diabetes: {probability:.2f})")
        else:
            st.success(f"**Patient is classified as: {risk_category}** (Probability of Diabetes: {probability:.2f})")

        st.subheader("💡 Explanation of Decision")
        st.markdown(explanation)

        st.markdown("---")
        st.info("This tool is for diagnostic support only and should not replace professional medical advice.")

if __name__ == "__main__":
    main()