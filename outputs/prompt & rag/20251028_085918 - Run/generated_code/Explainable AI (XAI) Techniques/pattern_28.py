import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance, plot_partial_dependence
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random

# --- 1. Data Ingestion & Preprocessing Module ---

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "Age": np.random.randint(20, 80, num_samples),
        "Gender": np.random.choice(["Male", "Female"], num_samples),
        "Symptom_Fever": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        "Symptom_Cough": np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
        "Symptom_Fatigue": np.random.choice([0, 1], num_samples, p=[0.5, 0.5]),
        "Lab_WBC": np.random.normal(7, 2, num_samples).round(1), # White Blood Cell Count
        "Lab_CRP": np.random.normal(5, 3, num_samples).round(1), # C-Reactive Protein
        "Medical_History_Diabetes": np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        "Medical_History_Hypertension": np.random.choice([0, 1], num_samples, p=[0.75, 0.25]),
    }
    df = pd.DataFrame(data)

    # Simulate a binary diagnosis (e.g., "Disease Present" / "Disease Absent")
    # More complex logic for diagnosis based on features
    df["Diagnosis"] = 0 # Default to absent
    df.loc[
        ((df["Symptom_Fever"] == 1) & (df["Symptom_Cough"] == 1)) |
        (df["Lab_WBC"] > 9) |
        ((df["Age"] > 60) & (df["Symptom_Fatigue"] == 1) & (df["Medical_History_Diabetes"] == 1)),
        "Diagnosis"
    ] = 1
    return df

# Preprocessing pipeline setup
def create_preprocessing_pipeline(df):
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    numerical_features.remove("Diagnosis")
    categorical_features = df.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ])
    return preprocessor, numerical_features, categorical_features

# --- 2. Core AI Diagnosis Models Module ---

def train_models(X_train, y_train, preprocessor):
    # Black-box Predictive Model (XGBoost)
    xgb_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
    ])
    xgb_pipeline.fit(X_train, y_train)

    # Inherently Interpretable Model (Logistic Regression)
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, solver="liblinear"))
    ])
    lr_pipeline.fit(X_train, y_train)

    return xgb_pipeline, lr_pipeline

# --- 3. Interpretability & Diagnostics Engine Module ---

class InterpretabilityEngine:
    def __init__(self, xgb_model, lr_model, X_train_raw, feature_names):
        self.xgb_model = xgb_model
        self.lr_model = lr_model
        self.X_train_raw = X_train_raw
        self.feature_names = feature_names

        # SHAP explainer for XGBoost (use the trained XGBoost model directly for SHAP values on preprocessed data)
        self.shap_explainer = shap.TreeExplainer(self.xgb_model.named_steps["classifier"])

    def get_xgb_shap_values(self, instance_raw):
        # Preprocess the instance using the pipeline's preprocessor
        preprocessed_instance = self.xgb_model.named_steps["preprocessor"].transform(pd.DataFrame([instance_raw]))
        # Get original feature names from preprocessor
        feature_names_out = self.xgb_model.named_steps["preprocessor"].get_feature_names_out()
        shap_values = self.shap_explainer.shap_values(preprocessed_instance)[1] # Assuming binary classification, get values for positive class
        # Create an explanation object for force plot
        expected_value = self.shap_explainer.expected_value[1]
        return shap_values, expected_value, feature_names_out

    def plot_ice(self, model_pipeline, instance_raw, feature_to_vary, original_df):
        fig, ax = plt.subplots(figsize=(8, 6))
        instance_df = pd.DataFrame([instance_raw])
        original_prediction = model_pipeline.predict_proba(instance_df)[:, 1][0]

        # Get the range for the feature to vary
        if pd.api.types.is_numeric_dtype(original_df[feature_to_vary]):
            feature_min = original_df[feature_to_vary].min()
            feature_max = original_df[feature_to_vary].max()
            feature_values = np.linspace(feature_min, feature_max, 50)
        else:
            feature_values = original_df[feature_to_vary].unique()

        predictions = []
        for val in feature_values:
            temp_instance = instance_df.copy()
            temp_instance[feature_to_vary] = val
            predictions.append(model_pipeline.predict_proba(temp_instance)[:, 1][0])

        ax.plot(feature_values, predictions, label=f"ICE for {feature_to_vary}")
        ax.axhline(original_prediction, color='r', linestyle='--', label='Original Prediction')
        ax.set_xlabel(feature_to_vary)
        ax.set_ylabel("Predicted Probability (Disease Present)")
        ax.set_title(f"Individual Conditional Expectation Plot for {feature_to_vary}")
        ax.legend()
        return fig

    def get_counterfactual_explanation(self, model_pipeline, instance_raw, target_label=0):
        # Simplified counterfactual: Find a minimal change to flip the prediction
        # A full DiCE integration would be more robust but complex for a single file.
        # This is a basic illustrative example.
        original_prediction = model_pipeline.predict(pd.DataFrame([instance_raw]))[0]
        original_proba = model_pipeline.predict_proba(pd.DataFrame([instance_raw]))[:, 1][0]

        if original_prediction == target_label:
            return f"Original prediction is already {target_label}. No counterfactual needed to flip to {target_label}."

        counterfactual_found = False
        counterfactual_instance = instance_raw.copy()
        changes = {}

        # Try perturbing features one by one (simplistic)
        for feature in instance_raw.keys():
            temp_instance = instance_raw.copy()
            original_val = temp_instance[feature]

            if pd.api.types.is_numeric_dtype(self.X_train_raw[feature]):
                perturbations = [original_val * 0.9, original_val * 1.1, original_val + 1, original_val - 1]
                perturbations = [p for p in perturbations if p >= self.X_train_raw[feature].min() and p <= self.X_train_raw[feature].max()]
            elif pd.api.types.is_object_dtype(self.X_train_raw[feature]): # Categorical
                other_values = [v for v in self.X_train_raw[feature].unique() if v != original_val]
                perturbations = other_values
            else:
                continue # Skip other types

            for p_val in perturbations:
                if p_val == original_val: continue
                temp_instance[feature] = p_val
                new_prediction = model_pipeline.predict(pd.DataFrame([temp_instance]))[0]

                if new_prediction == target_label:
                    counterfactual_found = True
                    changes[feature] = (original_val, p_val)
                    counterfactual_instance = temp_instance # Keep the first one found
                    return {"original": instance_raw, "counterfactual": counterfactual_instance, "changes": changes, "original_proba": original_proba, "cf_proba": model_pipeline.predict_proba(pd.DataFrame([counterfactual_instance]))[:, 1][0]}

        if not counterfactual_found:
            return "Could not find a simple counterfactual by perturbing a single feature to flip the prediction."
        return "Error in counterfactual generation."

    def plot_global_pdp(self, model_pipeline, feature_name, preprocessor_transformer):
        fig, ax = plt.subplots(figsize=(10, 6))
        # `plot_partial_dependence` expects the preprocessed data and the original feature name
        # We need to map the feature name to its column index after preprocessing

        # This part is tricky because plot_partial_dependence expects an estimator and X
        # If the preprocessor changes feature names/order, direct indexing is hard.
        # A simpler way for demo is to pass the trained model and the original dataframe
        # and let plot_partial_dependence handle it IF the model is a scikit-learn pipeline
        # that includes the preprocessor.

        # Get feature names after preprocessing
        original_feature_names = self.X_train_raw.columns.tolist()
        # Identify the index of the feature_name in the original dataset
        feature_idx = original_feature_names.index(feature_name)

        # Use the pipeline directly, plot_partial_dependence can handle it.
        # Ensure feature_name is a list for plot_partial_dependence
        display = plot_partial_dependence(model_pipeline, self.X_train_raw,
                                          features=[feature_name], target=1, 
                                          feature_names=original_feature_names, 
                                          grid_resolution=20, ax=ax)
        fig = display.figure_
        fig.suptitle(f"Partial Dependence Plot for {feature_name}")
        return fig

    def plot_permutation_importance(self, model_pipeline, X_test, y_test):
        result = permutation_importance(model_pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.boxplot(result.importances[sorted_idx].T,
                   vert=False, labels=np.array(self.feature_names)[sorted_idx])
        ax.set_title("Permutation Feature Importance")
        fig.tight_layout()
        return fig

    def subgroup_analysis(self, model_pipeline, df_test, y_test, group_feature="Gender"):
        results = {}
        if group_feature not in df_test.columns:
            return {"error": f"Group feature '{group_feature}' not found in data."}

        for group_val in df_test[group_feature].unique():
            group_df = df_test[df_test[group_feature] == group_val]
            group_y_test = y_test[df_test[group_feature] == group_val]

            if len(group_df) == 0:
                results[group_val] = {"count": 0, "accuracy": "N/A", "roc_auc": "N/A"}
                continue

            group_predictions = model_pipeline.predict(group_df)
            group_proba = model_pipeline.predict_proba(group_df)[:, 1]

            acc = accuracy_score(group_y_test, group_predictions)
            roc_auc = roc_auc_score(group_y_test, group_proba)

            results[group_val] = {
                "count": len(group_df),
                "accuracy": f"{acc:.2f}",
                "roc_auc": f"{roc_auc:.2f}"
            }
        return results

# --- 4. Interactive User Interface (UI) Module ---

st.set_page_config(layout="wide", page_title="Interpretable Medical Diagnosis Assistant")
st.title("🩺 Interpretable Medical Diagnosis Assistant")

@st.cache_data
def load_and_preprocess_data():
    df = generate_synthetic_data()
    preprocessor, num_features, cat_features = create_preprocessing_pipeline(df)
    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return df, X_train, X_test, y_train, y_test, preprocessor, num_features, cat_features

@st.cache_resource # Use cache_resource for models to avoid re-training
def load_and_train_models(X_train, y_train, preprocessor):
    xgb_model, lr_model = train_models(X_train, y_train, preprocessor)
    return xgb_model, lr_model

# Load data and models
df_full, X_train, X_test, y_train, y_test, preprocessor_pipeline, num_features, cat_features = load_and_preprocess_data()
xgb_pipeline, lr_pipeline = load_and_train_models(X_train, y_train, preprocessor_pipeline)

feature_names = X_train.columns.tolist()
interpret_engine = InterpretabilityEngine(xgb_pipeline, lr_pipeline, X_train, feature_names)

# Sidebar for model selection and global explanations
st.sidebar.header("Global Explanations & Model Settings")
selected_model_name = st.sidebar.selectbox(
    "Select Model for Global Explanations",
    ("XGBoost (Black-box)", "Logistic Regression (Interpretable)")
)

model_for_global_exp = xgb_pipeline if selected_model_name == "XGBoost (Black-box)" else lr_pipeline

st.sidebar.subheader("Permutation Feature Importance")
pfi_plot = interpret_engine.plot_permutation_importance(model_for_global_exp, X_test, y_test)
st.sidebar.pyplot(pfi_plot)

st.sidebar.subheader("Partial Dependence Plots")
pdp_feature = st.sidebar.selectbox(
    "Select Feature for PDP",
    feature_names, key="pdp_feature_select"
)
if pdp_feature:
    pdp_plot = interpret_engine.plot_global_pdp(model_for_global_exp, pdp_feature, preprocessor_pipeline)
    st.sidebar.pyplot(pdp_plot)

st.sidebar.subheader("Subgroup Diagnostics (DivExplorer-like)")
diagnostic_group_feature = st.sidebar.selectbox(
    "Group by Feature",
    [f for f in feature_names if f in cat_features or df_full[f].nunique() < 10], key="subgroup_feature_select"
)
if diagnostic_group_feature:
    subgroup_res = interpret_engine.subgroup_analysis(model_for_global_exp, X_test, y_test, diagnostic_group_feature)
    if "error" in subgroup_res:
        st.sidebar.error(subgroup_res["error"])
    else:
        st.sidebar.write("**Performance by Subgroup:**")
        for group, metrics in subgroup_res.items():
            st.sidebar.write(f"**{group}:** Count={metrics['count']}, Acc={metrics['accuracy']}, AUC={metrics['roc_auc']}")

# Main content for patient input and local explanations
st.header("Patient Data Input")

with st.form("patient_input_form"):
    cols = st.columns(len(feature_names) // 2 + len(feature_names) % 2)
    patient_data = {}
    for i, feature in enumerate(feature_names):
        with cols[i % (len(feature_names) // 2 + len(feature_names) % 2)]:
            if feature == "Age":
                patient_data[feature] = st.slider("Age", 20, 80, 45)
            elif feature == "Gender":
                patient_data[feature] = st.selectbox("Gender", ["Male", "Female"])
            elif "Symptom" in feature:
                patient_data[feature] = st.checkbox(f"Has {feature.replace('Symptom_', '')}", value=False)
            elif "Lab" in feature:
                min_val, max_val = df_full[feature].min(), df_full[feature].max()
                patient_data[feature] = st.number_input(f"{feature.replace('Lab_', '')} Value", min_value=float(min_val), max_value=float(max_val), value=float(df_full[feature].mean()), step=0.1)
            elif "Medical_History" in feature:
                patient_data[feature] = st.checkbox(f"Has {feature.replace('Medical_History_', '')} History", value=False)

    submitted = st.form_submit_button("Get Diagnosis and Explanations")

if submitted:
    st.subheader("Diagnosis Results")
    patient_df = pd.DataFrame([patient_data])

    st.write(f"#### Input Patient Data:")
    st.dataframe(patient_df)

    col1, col2 = st.columns(2)

    with col1:
        st.write("##### XGBoost Model Prediction")
        xgb_pred = xgb_pipeline.predict(patient_df)[0]
        xgb_proba = xgb_pipeline.predict_proba(patient_df)[:, 1][0]
        st.info(f"**Diagnosis (XGBoost):** {'Disease Present' if xgb_pred == 1 else 'Disease Absent'} (Probability: {xgb_proba:.2f})")

        st.write("##### Local Explanation (SHAP Values for XGBoost)")
        shap_values, expected_value, feature_names_out = interpret_engine.get_xgb_shap_values(patient_data)
        
        # Create a dummy shap.Explanation object for the force plot
        explanation = shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=xgb_pipeline.named_steps["preprocessor"].transform(patient_df).flatten(),
            feature_names=feature_names_out
        )
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st_shap(shap.force_plot(explanation.base_values, explanation.values, explanation.data, feature_names=explanation.feature_names))

    with col2:
        st.write("##### Logistic Regression Model Prediction")
        lr_pred = lr_pipeline.predict(patient_df)[0]
        lr_proba = lr_pipeline.predict_proba(patient_df)[:, 1][0]
        st.info(f"**Diagnosis (Logistic Regression):** {'Disease Present' if lr_pred == 1 else 'Disease Absent'} (Probability: {lr_proba:.2f})")

        st.write("##### Local Explanation (Coefficients for Logistic Regression)")
        # Coefficients for Logistic Regression are directly interpretable after preprocessing
        # Need to map coefficients back to original feature names for better understanding
        lr_coefficients = lr_pipeline.named_steps["classifier"].coef_[0]
        # Get feature names from preprocessor
        preprocessed_feature_names = lr_pipeline.named_steps["preprocessor"].get_feature_names_out()
        lr_coef_df = pd.DataFrame({"Feature": preprocessed_feature_names, "Coefficient": lr_coefficients})
        st.dataframe(lr_coef_df.sort_values(by="Coefficient", ascending=False))

    st.subheader("Individual Conditional Expectation (ICE) Plot")
    ice_feature = st.selectbox("Select Feature for ICE Plot", feature_names, key="ice_feature_select")
    if ice_feature:
        ice_plot = interpret_engine.plot_ice(xgb_pipeline, patient_data, ice_feature, df_full.drop("Diagnosis", axis=1))
        st.pyplot(ice_plot)

    st.subheader("Counterfactual Explanation")
    target_cf_label = 1 - xgb_pred # Try to flip the prediction
    counterfactual_result = interpret_engine.get_counterfactual_explanation(xgb_pipeline, patient_data, target_label=target_cf_label)
    if isinstance(counterfactual_result, str):
        st.write(counterfactual_result)
    else:
        st.write(f"To change the prediction from '{'Disease Present' if xgb_pred == 1 else 'Disease Absent'}' (P={counterfactual_result['original_proba']:.2f}) to '{'Disease Present' if target_cf_label == 1 else 'Disease Absent'}' (P={counterfactual_result['cf_proba']:.2f}), consider the following minimal changes:")
        changes_df = pd.DataFrame([counterfactual_result['changes']]).T.rename(columns={0: "(Original Value, Counterfactual Value)"})
        st.dataframe(changes_df)
        st.write("**Counterfactual Instance:**")
        st.dataframe(pd.DataFrame([counterfactual_result['counterfactual']]))

# Helper for SHAP streamlit visualization (from shap documentation/examples)
def st_shap(plot, height=None):
    shap_html = f"<head>{plot.html}</head><body>{plot.js}</body>"
    st.components.v1.html(shap_html, height=height)
