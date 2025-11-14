import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import plot_partial_dependence, permutation_importance
from sklearn.datasets import make_classification
import shap
import matplotlib.pyplot as plt

# --- 1. Data Layer: Synthetic Medical Data Generation ---
def generate_medical_data(n_samples=1000, n_features=10, random_state=42):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=2, n_classes=2, random_state=random_state)
    feature_names = [f"Symptom_{i+1}" for i in range(n_features - 5)] + \
                    [f"Lab_Result_{i+1}" for i in range(5)] # Example feature names
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y

X, y = generate_medical_data()
feature_names = X.columns.tolist()

# --- 2. Model Layer: Black-Box AI Model Training ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_model(X_train_data, y_train_data):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_data, y_train_data)
    return model

model = train_model(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- 3. Interpretability Layer Setup ---
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# --- 4. User Interface Layer (Streamlit) ---
st.set_page_config(layout="wide", page_title="Medical Diagnosis AI Explainer")

st.title("🩺 Medical Diagnosis AI Explainer")
st.markdown("An interactive tool to understand black-box AI medical diagnostic models.")

st.sidebar.header("Application Settings")

st.sidebar.subheader("Model Performance")
st.sidebar.write(f"**Overall Model Accuracy:** {accuracy:.2f}")

# --- Tabs for different explanation views ---
tabs = st.tabs(["Patient-Specific Explanation", "Global Model Explanations", "Counterfactuals (Conceptual)"])

with tabs[0]:
    st.header("Patient-Specific Explanation (Local Interpretability)")
    st.markdown("Understand the contribution of each patient's attributes to their specific diagnosis.")

    patient_idx = st.slider("Select a Patient Instance", 0, len(X) - 1, 0)
    selected_patient = X.iloc[patient_idx]
    predicted_diagnosis = model.predict(selected_patient.to_frame().T)[0]
    predicted_proba = model.predict_proba(selected_patient.to_frame().T)[0, predicted_diagnosis]

    st.subheader(f"Predicted Diagnosis for Patient {patient_idx}:")
    st.info(f"**Diagnosis:** {'Condition Present' if predicted_diagnosis == 1 else 'Condition Absent'} (Probability: {predicted_proba:.2f})")

    st.subheader("Feature Contributions (SHAP Force Plot)")
    st.write("The SHAP force plot shows how each feature pushes the model output from the base value (average prediction) to the final prediction for this patient.")
    
    # SHAP force plot for the selected patient
    shap_object = explainer(X)
    
    # Ensure correct output_names are set for binary classification
    # shap_object.output_names = ['Condition Absent', 'Condition Present'] if len(shap_object.output_names) != 2 else shap_object.output_names

    # For binary classification, SHAP returns an array of shape (N, F, 2) where the last dimension is for each class
    # We typically explain the positive class (class 1)
    try:
        fig, ax = plt.subplots(figsize=(10, 3))
        shap.force_plot(explainer.expected_value[1], shap_values[1][patient_idx,:], selected_patient, matplotlib=True, show=False)
        st.pyplot(fig, bbox_inches='tight')
    except Exception as e:
        st.error(f"Could not generate SHAP force plot: {e}. \nThis often happens with older `shap` versions or specific model types. Try `shap.waterfall_plot` instead.")
        # Fallback to waterfall plot if force plot fails
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(shap.Explanation(values=shap_values[1][patient_idx,:], base_values=explainer.expected_value[1], data=selected_patient.values, feature_names=feature_names), max_display=15, show=False)
            st.pyplot(fig, bbox_inches='tight')
            st.write("Using Waterfall Plot as a fallback for SHAP explanation.")
        except Exception as wf_e:
            st.error(f"Could not generate SHAP waterfall plot either: {wf_e}")

    st.subheader("Overall SHAP Feature Importance (Summary Plot)")
    st.write("A global view of which features are most important across all predictions.")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
    st.pyplot(fig, bbox_inches='tight')

with tabs[1]:
    st.header("Global Model Explanations")
    st.markdown("Understand the overall behavior and feature importance of the diagnostic model.")

    st.subheader("Partial Dependence Plots (PDP)")
    st.write("Shows the marginal effect of one or two features on the predicted outcome.")

    col1, col2 = st.columns(2)
    with col1:
        pdp_feature1 = st.selectbox("Select Feature 1 for PDP", feature_names, index=0)
    with col2:
        pdp_feature2 = st.selectbox("Select Feature 2 for PDP (optional)", [None] + feature_names, index=0 if len(feature_names) < 2 else 1)

    features_to_plot = [pdp_feature1]
    if pdp_feature2 and pdp_feature2 != pdp_feature1:
        features_to_plot.append(pdp_feature2)

    if features_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_partial_dependence(model, X_train, features=features_to_plot, 
                               feature_names=feature_names, ax=ax, grid_resolution=20)
        fig.suptitle(f"Partial Dependence Plot for {', '.join(features_to_plot)}", y=1.02)
        plt.tight_layout()
        st.pyplot(fig, bbox_inches='tight')
    else:
        st.info("Please select at least one feature for the Partial Dependence Plot.")

    st.subheader("Permutation Feature Importance (PFI)")
    st.write("Measures the overall importance of each feature by shuffling its values and observing the drop in model performance.")
    
    # Calculate PFI (can be time-consuming for large datasets)
    if st.button("Calculate Permutation Importance"): 
        with st.spinner('Calculating Permutation Importance...'):
            result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
            sorted_idx = result.importances_mean.argsort()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(result.importances[sorted_idx].T,
                       vert=False, labels=np.array(feature_names)[sorted_idx])
            ax.set_title("Permutation Feature Importance")
            ax.set_ylabel("Feature")
            ax.set_xlabel("Mean accuracy decrease")
            plt.tight_layout()
            st.pyplot(fig, bbox_inches='tight')

    st.subheader("Individual Conditional Expectation (ICE) Plots (Conceptual)")
    st.write("ICE plots show the prediction for each instance as a function of a feature of interest, revealing heterogeneity. \n*Full interactive implementation is complex for a prototype, but conceptually, you'd see individual lines for each patient here.*")
    st.info("For a full ICE plot, you would typically use libraries like `PyALE` or `Alibi Explainer`. This section is a conceptual placeholder.")

with tabs[2]:
    st.header("Counterfactual Explanations (Conceptual)")
    st.markdown("Explore 'what-if' scenarios to understand how minimal changes to input features could alter the model's prediction.")
    st.write("Counterfactual explanations answer questions like: 'What is the smallest change to this patient's symptoms that would lead to a different diagnosis?'")
    st.info("Generating robust counterfactual explanations is a research area and complex to implement directly in a simple prototype. Libraries like `DiCE` or `Alibi Explainer` offer frameworks for this. This section serves as a conceptual demonstration.")
    
    st.subheader("Example Scenario:")
    st.write("Imagine Patient X was diagnosed with 'Condition Present'. A counterfactual explanation might suggest: 'If Symptom_3 was reduced by 2 units and Lab_Result_1 increased by 0.5 units, the diagnosis would likely change to 'Condition Absent'.'")
    st.markdown("--- This section would contain an interactive tool to modify features and see potential outcome shifts. ---")

