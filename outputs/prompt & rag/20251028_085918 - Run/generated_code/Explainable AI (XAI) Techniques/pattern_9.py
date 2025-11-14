"""
This file contains the combined code for the 'AI-powered Medical Diagnosis Interpretability Platform'.
It includes the model and data generation script, the Streamlit interpretability platform application,
and the README instructions.
"""

# --- FILE: model_and_data.py ---
# This script generates a synthetic medical dataset, trains a RandomForestClassifier,
# and saves the trained model, feature names, and the full dataset to disk.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

print("Running model_and_data.py to generate data and train model...")

# --- 1. Generate Synthetic Medical Dataset ---
np.random.seed(42)

num_samples = 1000

data = {
    "Pregnancies": np.random.randint(0, 10, num_samples),
    "Glucose": np.random.randint(70, 200, num_samples),
    "BloodPressure": np.random.randint(60, 120, num_samples),
    "SkinThickness": np.random.randint(10, 60, num_samples),
    "Insulin": np.random.randint(0, 300, num_samples),
    "BMI": np.random.uniform(18.0, 50.0, num_samples),
    "DiabetesPedigreeFunction": np.random.uniform(0.08, 2.5, num_samples),
    "Age": np.random.randint(20, 80, num_samples),
    "Outcome": np.random.randint(0, 2, num_samples) # 0 for No Diabetes, 1 for Diabetes
}

df = pd.DataFrame(data)

# Introduce some correlation to make the outcome somewhat predictable
df["Outcome"] = df.apply(lambda row:
    1 if (
        (row["Glucose"] > 140 and row["BMI"] > 30) or 
        (row["Age"] > 45 and row["Pregnancies"] > 3 and row["Glucose"] > 120)
    ) else (0 if np.random.rand() < 0.7 else 1), axis=1)

# Ensure a reasonable balance
num_diabetes = df["Outcome"].sum()
num_no_diabetes = num_samples - num_diabetes
if num_diabetes < num_samples * 0.3:
    # Artificially increase diabetes cases if too few
    diabetes_indices = df[df["Outcome"] == 0].sample(n=int(num_samples * 0.3 - num_diabetes), random_state=42).index
    df.loc[diabetes_indices, "Outcome"] = 1

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

feature_names = X.columns.tolist()

# --- 2. Train a RandomForestClassifier Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# --- 3. Save Model, Feature Names, and Dataset ---
model_filename = "random_forest_model.joblib"
features_filename = "feature_names.joblib"
dataset_X_filename = "dataset_X.csv"
dataset_y_filename = "dataset_y.csv"

joblib.dump(model, model_filename)
joblib.dump(feature_names, features_filename)
X.to_csv(dataset_X_filename, index=False)
y.to_csv(dataset_y_filename, index=False)

print(f"Model saved to {model_filename}")
print(f"Feature names saved to {features_filename}")
print(f"Dataset features (X) saved to {dataset_X_filename}")
print(f"Dataset target (y) saved to {dataset_y_filename}")
print("model_and_data.py execution complete.\n")


# --- FILE: interpretability_platform.py ---
# This Streamlit application provides an interactive platform for interpreting and debugging
# a medical diagnosis AI model.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence, permutation_importance
import matplotlib.pyplot as plt
import matplotlib
import io # For capturing matplotlib figures

matplotlib.use("Agg") # Use non-interactive backend for matplotlib

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="AI Medical Diagnosis Interpretability")

# --- 1. Model and Data Loading ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("random_forest_model.joblib")
        feature_names = joblib.load("feature_names.joblib")
        X = pd.read_csv("dataset_X.csv")
        y = pd.read_csv("dataset_y.csv").squeeze() # .squeeze() to get a Series
        return model, feature_names, X, y
    except FileNotFoundError:
        st.error("Required model and data files not found. Please run `model_and_data.py` first.")
        st.stop()

# Load model and data
model, feature_names, X_full, y_full = load_artifacts()

# Add 'Outcome_Label' for display
X_full['Outcome_Label'] = y_full.apply(lambda x: 'Diabetes' if x == 1 else 'No Diabetes')

# --- Streamlit Application Layout ---
st.title("🩺 AI-powered Medical Diagnosis Interpretability Platform")
st.write("Explore and understand black-box AI model predictions for medical diagnoses.")

# --- Sidebar for User Input ---
st.sidebar.header("Patient and Subgroup Selection")

# Patient Selection
patient_indices = list(X_full.index)
selected_patient_idx = st.sidebar.selectbox(
    "Select a Patient Instance for Local Explanation:",
    options=patient_indices,
    format_func=lambda x: f"Patient ID: {x}"
)
selected_patient_data = X_full.loc[selected_patient_idx].drop(columns=['Outcome_Label']) # Exclude Outcome_Label from features
selected_patient_true_outcome = X_full.loc[selected_patient_idx, 'Outcome_Label']

# Subgroup Filtering (simplified to Age ranges)
# Create age groups for filtering
age_bins = [0, 30, 45, 60, 100]
age_labels = ["<30", "30-44", "45-59", "60+"]
X_full["Age_Group"] = pd.cut(X_full["Age"], bins=age_bins, labels=age_labels, right=False)

subgroup_feature = "Age_Group"
all_subgroup_values = ["All Patients"] + list(X_full[subgroup_feature].unique().astype(str))
selected_subgroup_value = st.sidebar.selectbox(
    f"Filter Global Explanations by {subgroup_feature}:",
    options=all_subgroup_values
)

# Apply subgroup filter to the dataset for global explanations
if selected_subgroup_value != "All Patients":
    X_filtered_for_global = X_full[X_full[subgroup_feature] == selected_subgroup_value].drop(columns=[subgroup_feature, 'Outcome_Label'])
    y_filtered_for_global = y_full[X_full[subgroup_feature] == selected_subgroup_value]
    st.sidebar.info(f"Displaying global explanations for subgroup: **{selected_subgroup_value}**")
else:
    X_filtered_for_global = X_full.drop(columns=[subgroup_feature, 'Outcome_Label'])
    y_filtered_for_global = y_full


# --- Main Content Area ---
st.header("1. Selected Patient Diagnosis")
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Patient ID: {selected_patient_idx}")
    st.markdown("**Patient Features:**")
    st.dataframe(selected_patient_data.to_frame().T.astype(str))
    st.markdown(f"**True Outcome:** <span style='color: {'green' if selected_patient_true_outcome == 'No Diabetes' else 'red'}'>{selected_patient_true_outcome}</span>", unsafe_allow_html=True)

with col2:
    st.subheader("Model Prediction")
    patient_prediction_proba = model.predict_proba(selected_patient_data.to_frame().T)[0]
    patient_prediction_label = "Diabetes" if model.predict(selected_patient_data.to_frame().T)[0] == 1 else "No Diabetes"
    
    st.markdown(f"**Predicted Outcome:** <span style='color: {'green' if patient_prediction_label == 'No Diabetes' else 'red'}'>{patient_prediction_label}</span>", unsafe_allow_html=True)
    st.write(f"Probability of No Diabetes: {patient_prediction_proba[0]:.2f}")
    st.write(f"Probability of Diabetes: {patient_prediction_proba[1]:.2f}")

st.markdown("---<br>", unsafe_allow_html=True)

st.header("2. Local Interpretability: Understanding Individual Predictions")
st.write(
    "Local interpretability techniques help us understand *why* the model made a specific prediction for an individual patient. "
    "Methods like LACE, SHAP, and LIME provide instance-specific insights into feature contributions."
)

st.subheader("Conceptual Local Feature Contributions (LACE/SHAP-like)")
st.info(
    "*In a full implementation, a library like SHAP or LIME would generate exact feature contribution scores. "
    "Here, we provide a conceptual overview and use general feature importance as a proxy for the overall model.*"
)

# Simple conceptual local explanation (using model's global feature importances as a hint)
st.write("**Top features influencing this patient's prediction (conceptual):**")
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

st.dataframe(feature_importance_df.head())
st.caption("Note: These are global feature importances, not specific to this patient's prediction in a LACE/SHAP sense.")


st.subheader("Individual Conditional Expectation (ICE) Plots")
st.write(
    "ICE plots show how the prediction for a single instance changes as a specific feature varies, while "
    "all other features remain constant."
)

selected_ice_feature = st.selectbox(
    "Select a feature for ICE plot:",
    options=feature_names, key="ice_feature"
)

if st.button("Generate ICE Plot", key="generate_ice"):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_partial_dependence(
        estimator=model,
        X=X_full.drop(columns=[subgroup_feature, 'Outcome_Label']),
        features=[selected_ice_feature],
        kind='individual',
        feature_names=feature_names,
        ax=ax,
        target=1, # Plotting for the 'Diabetes' class
        # Optional: Specify a single instance for the individual plot
        # The plot_partial_dependence for individual kind actually plots for all instances by default
        # To get a single ICE plot for a selected patient, a custom plotting approach or specific library (e.g., alibi) might be needed
        # For this example, we'll show the effect across the dataset, but highlight the selected patient's value.
    )
    ax.set_title(f"ICE Plot for {selected_ice_feature} (Selected Patient value: {selected_patient_data[selected_ice_feature]:.2f})")
    ax.axvline(x=selected_patient_data[selected_ice_feature], color='red', linestyle='--', label='Selected Patient Value')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Conceptual Counterfactual Explanations")
st.write(
    "Counterfactual explanations answer 'what if' questions by showing the smallest changes to a patient's features "
    "that would change the model's prediction (e.g., from 'Diabetes' to 'No Diabetes'). This provides actionable insights."
)
st.info(
    "*A dedicated library like `dice_ml` or `alibi` would be used to generate robust counterfactuals. "
    "Here, we describe the concept: Imagine a patient predicted with Diabetes. A counterfactual might say: "
    "'If their Glucose was 110 (instead of 150) and BMI was 25 (instead of 32), the model would predict No Diabetes.'*"
)

st.markdown("---<br>", unsafe_allow_html=True)

st.header("3. Global Interpretability: Understanding Overall Model Behavior")
st.write(
    "Global interpretability techniques provide a high-level understanding of how the model works across the entire dataset "
    "or specific subgroups, helping to identify general trends and potential biases."
)

st.subheader("Partial Dependence Plots (PDP)")
st.write(
    "PDPs show the average marginal effect of one or two features on the predicted outcome of a machine learning model."
)

selected_pdp_feature = st.selectbox(
    "Select a feature for PDP:",
    options=feature_names, key="pdp_feature"
)

if st.button("Generate PDP", key="generate_pdp"):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_partial_dependence(
        estimator=model,
        X=X_filtered_for_global, # Use filtered data if subgroup selected
        features=[selected_pdp_feature],
        kind='average',
        feature_names=feature_names,
        ax=ax,
        target=1, # Plotting for the 'Diabetes' class
    )
    ax.set_title(f"Partial Dependence Plot for {selected_pdp_feature}")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Permutation Feature Importance")
st.write(
    "Permutation Feature Importance measures the increase in the prediction error of the model after permuting (shuffling) "
    "the values of a single feature, thus breaking the relationship between the feature and the true outcome. "
    "Features that cause a large increase in error are considered more important."
)

if st.button("Calculate and Display Permutation Importance", key="generate_pfi"):
    # Recalculate PFI for the filtered dataset
    result = permutation_importance(model, X_filtered_for_global, y_filtered_for_global, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=np.array(feature_names)[sorted_idx])
    ax.set_title("Permutation Feature Importance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---<br>", unsafe_allow_html=True)

st.header("4. Subgroup Comparison (DivExplorer-like)")
st.write(
    "This section enables comparison of model behavior and explanations across different patient subgroups. "
    "By selecting a subgroup in the sidebar, the global interpretability plots (PDP and Permutation Importance) "
    "above will automatically reflect the data for that subgroup, allowing medical professionals to identify "
    "divergent model behaviors and potential biases (e.g., if a feature is more important for one age group than another)."
)
st.info(
    f"The global plots above are currently showing explanations for: **{selected_subgroup_value}**."
    "Change the 'Filter Global Explanations by Age_Group' in the sidebar to compare different subgroups."
)



# --- FILE: README.md ---
"""
# AI-powered Medical Diagnosis Interpretability Platform

This project implements an AI Interpretability & Debugging Framework, focusing on a synthetic medical diagnosis scenario. It provides tools to understand and debug a black-box AI model's predictions, fostering trust and enabling responsible deployment.

## Project Structure and How to Use This Bundle

This file (`project_bundle.py`) combines the content of three logical files:

1.  `model_and_data.py`: Handles synthetic data generation and model training.
2.  `interpretability_platform.py`: The Streamlit web application for interactive model interpretability.
3.  `README.md`: This instruction set.

To run this project, you will first need to separate the contents into their respective files. The boundaries for each file are clearly marked within this bundle.

## Features

- **Synthetic Data Generation:** Creates a mock medical dataset for diabetes prediction.
- **Black-box Model Training:** Trains a `RandomForestClassifier` on the synthetic data.
- **Interactive Patient Selection:** Select an individual patient to inspect their diagnosis.
- **Local Interpretability (Conceptual):** Explains how local explanations like LACE/SHAP/LIME would provide instance-specific insights.
- **Individual Conditional Expectation (ICE) Plots:** Visualize how a single feature affects a patient's prediction.
- **Counterfactual Explanations (Conceptual):** Describes the concept of finding minimal changes to alter a prediction.
- **Global Interpretability:**
    - **Partial Dependence Plots (PDP):** Show the average effect of one or two features on the model's prediction across the dataset.
    - **Permutation Feature Importance:** Ranks features by their overall importance to the model's performance.
- **Subgroup Comparison:** Filter the dataset by demographic features (e.g., age group) to compare global explanations and identify potential biases.

## Setup and Installation

1.  **Separate the files:** Create the following three files from this `project_bundle.py` content:
    - `model_and_data.py`: Copy the content under `# --- FILE: model_and_data.py ---` to a new file named `model_and_data.py`.
    - `interpretability_platform.py`: Copy the content under `# --- FILE: interpretability_platform.py ---` to a new file named `interpretability_platform.py`.
    - `README.md`: Copy the content under `# --- FILE: README.md ---` to a new file named `README.md`.

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install pandas numpy scikit-learn matplotlib streamlit joblib
    ```

## Usage

### Step 1: Prepare the Model and Data

First, run the `model_and_data.py` script to generate the synthetic dataset and train the machine learning model. This script will save the model and data files needed by the Streamlit application.

```bash
python model_and_data.py
```

You should see output indicating that `random_forest_model.joblib`, `feature_names.joblib`, `dataset_X.csv`, and `dataset_y.csv` have been saved.

### Step 2: Run the Interpretability Platform

Once the model and data are prepared, launch the Streamlit application:

```bash
streamlit run interpretability_platform.py
```

This command will open the application in your web browser. You can then interact with the platform to explore model explanations.

## Conceptual Explanations

Some interpretability techniques (like advanced LACE, SHAP, LIME, and robust Counterfactual Explanations) require dedicated libraries (e.g., `shap`, `lime`, `dice_ml`) which are not directly included in this minimal example to keep it self-contained and focused on the `sklearn` and `streamlit` integration for core interpretability patterns. The application provides textual explanations and conceptual placeholders for these more advanced methods.
"""
