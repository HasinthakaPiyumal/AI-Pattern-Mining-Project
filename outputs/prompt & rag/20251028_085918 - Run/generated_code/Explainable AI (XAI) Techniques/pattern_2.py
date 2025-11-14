
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import shap
import os

# --- Configuration --- #
MODEL_PATH = "random_forest_model.joblib"
FEATURES_PATH = "feature_names.joblib"
DATA_PATH = "simulated_patient_data.csv"
TARGET_COLUMN = "diagnosis"

# --- 1. Data Layer: Simulate Patient Data --- #
def simulate_patient_data(n_samples=1000):
    np.random.seed(42)
    data = {
        "age": np.random.randint(20, 80, n_samples),
        "blood_pressure": np.random.randint(90, 180, n_samples),
        "cholesterol": np.random.randint(150, 300, n_samples),
        "heart_rate": np.random.randint(60, 100, n_samples),
        "sugar_level": np.random.randint(70, 200, n_samples),
        "bmi": np.random.uniform(18.0, 35.0, n_samples),
        "family_history": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "smoker": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "exercise_freq": np.random.randint(0, 7, n_samples), # days per week
    }
    df = pd.DataFrame(data)

    # Simulate a target diagnosis (simplified)
    conditions = [
        (df["age"] > 60) & (df["cholesterol"] > 250) & (df["blood_pressure"] > 140),
        (df["sugar_level"] > 150) & (df["bmi"] > 30),
        (df["heart_rate"] > 90) & (df["smoker"] == 1),
    ]
    choices = ["Cardiovascular Disease", "Diabetes", "Respiratory Issue"]
    df[TARGET_COLUMN] = np.select(conditions, choices, default="Healthy")

    # Add some noise and make it more complex
    df.loc[df["age"] < 30, TARGET_COLUMN] = np.random.choice(["Healthy", "Minor Ailment"], sum(df["age"] < 30), p=[0.9, 0.1])
    df.loc[df["cholesterol"] < 180, TARGET_COLUMN] = np.random.choice(["Healthy", "Minor Ailment"], sum(df["cholesterol"] < 180), p=[0.95, 0.05])

    # Map target diagnoses to numerical labels for the model
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("category")
    df["diagnosis_code"] = df[TARGET_COLUMN].cat.codes

    return df.drop(columns=TARGET_COLUMN), df[TARGET_COLUMN]

# --- 2. Model Layer: Train and Save Model --- #
def train_and_save_model(X, y):
    st.sidebar.write("Training AI model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y.cat.codes, test_size=0.2, random_state=42, stratify=y.cat.codes)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.success(f"Model trained with accuracy: {accuracy:.2f}")

    # Save model, feature names, and target names
    joblib.dump(model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), FEATURES_PATH)
    joblib.dump(y.cat.categories.tolist(), "target_names.joblib")
    st.sidebar.success("Model and metadata saved.")

    return model, X.columns.tolist(), y.cat.categories.tolist()

# --- Streamlit Application --- #
st.set_page_config(layout="wide", page_title="Medical Diagnosis Explainer AI")
st.title("🩺 Medical Diagnosis Explainer AI")
st.markdown("This application provides transparent explanations for AI-driven medical diagnoses.")

# --- Load/Generate Data and Model --- #
@st.cache_data
def load_data_and_model():
    df_features, df_target = None, None
    model, feature_names, target_names = None, None, None

    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        st.sidebar.info("Simulating data and training a new model...")
        df_features, df_target = simulate_patient_data()
        df_features.to_csv(DATA_PATH, index=False) # Save simulated data
        model, feature_names, target_names = train_and_save_model(df_features, df_target)
    else:
        st.sidebar.info("Loading existing data and model...")
        df_features = pd.read_csv(DATA_PATH)
        # Re-create target series for categories if needed
        temp_df, df_target_series_cat = simulate_patient_data(n_samples=len(df_features)) # just to get category info
        df_target = df_target_series_cat

        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        target_names = joblib.load("target_names.joblib")

    return df_features, df_target, model, feature_names, target_names

X_data, y_data, model, feature_names, target_names = load_data_and_model()

# --- Sidebar for Patient Selection --- #
st.sidebar.header("Patient Selection")
patient_ids = list(X_data.index)
selected_patient_id = st.sidebar.selectbox("Select a Patient ID", patient_ids)

selected_patient_data = X_data.loc[[selected_patient_id]]
predicted_diagnosis_code = model.predict(selected_patient_data)[0]
predicted_diagnosis_proba = model.predict_proba(selected_patient_data)[0]

predicted_label = target_names[predicted_diagnosis_code]
predicted_prob = predicted_diagnosis_proba[predicted_diagnosis_code]

st.sidebar.subheader("Selected Patient Details")
st.sidebar.write(selected_patient_data.T)

st.subheader("AI Diagnosis for Selected Patient")
st.metric(label="Predicted Diagnosis", value=predicted_label, delta=f"Confidence: {predicted_prob*100:.2f}%")

st.write("\n")

# --- Explainability Layer --- #
st.header("AI Model Explanations")

# Create tabs for different explanations
tabs = st.tabs(["Local Explanation (SHAP)", "Global Explanations (PDP & ICE)", "Feature Importance"])

with tabs[0]:
    st.subheader("Local Explanation: How individual factors contributed to this diagnosis")
    st.markdown("SHAP (SHapley Additive exPlanations) values show how much each feature contributes to the prediction for the selected patient.")

    # SHAP Explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(selected_patient_data)

        # If multiclass, shap_values is a list of arrays, one for each class.
        # We want the shap values for the predicted class.
        if isinstance(shap_values, list):
            shap_values_for_predicted_class = shap_values[predicted_diagnosis_code]
        else:
            shap_values_for_predicted_class = shap_values

        st.write("\n")
        st.info(f"Displaying SHAP explanation for predicted diagnosis: **{predicted_label}**")

        # SHAP Waterfall Plot for the predicted class
        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap.Explanation(values=shap_values_for_predicted_class[0],
                                              base_values=explainer.expected_value[predicted_diagnosis_code] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                                              data=selected_patient_data.iloc[0],
                                              feature_names=feature_names),
                              max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

    except Exception as e:
        st.error(f"Error generating SHAP explanation: {e}")


with tabs[1]:
    st.subheader("Global Explanations: Partial Dependence & Individual Conditional Expectation Plots")
    st.markdown("Partial Dependence Plots (PDP) show the average marginal effect of a feature on the predicted outcome. Individual Conditional Expectation (ICE) plots show this for each individual patient.")

    # Select a few features for PDP/ICE demonstration
    pdp_features = feature_names[:3] # Taking first 3 for example

    for feature in pdp_features:
        st.write(f"#### Feature: `{feature}`")
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(
            estimator=model,
            X=X_data,
            features=[feature],
            kind='both', # Shows both PDP and ICE
            feature_names=feature_names,
            target=predicted_diagnosis_code, # Explaining for the predicted class
            line_kw={"color": "red", "label": "Partial Dependence"},
            ax=ax
        )
        ax.set_title(f"PDP and ICE for {feature} on '{predicted_label}' diagnosis")
        ax.legend()
        st.pyplot(fig)
        plt.clf()

with tabs[2]:
    st.subheader("Global Explanation: Permutation Feature Importance")
    st.markdown("Permutation Feature Importance quantifies how much the model's prediction error increases when a feature's values are randomly shuffled, indicating its importance.")

    # Re-split data for permutation importance (can use all data for demonstration)
    # For accurate permutation importance, it's usually done on a held-out test set.
    # For simplicity, using X_data and a temporary y_encoded from original data for demonstration.
    X_display = X_data
    y_display = y_data.cat.codes

    result = permutation_importance(model, X_display, y_display, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()

    fig_perm, ax_perm = plt.subplots(figsize=(12, 7))
    ax_perm.boxplot(result.importances[sorted_idx].T,
                   vert=False, labels=np.array(feature_names)[sorted_idx])
    ax_perm.set_title("Permutation Feature Importance")
    ax_perm.set_xlabel("Decrease in accuracy score")
    plt.tight_layout()
    st.pyplot(fig_perm)
    plt.clf()
