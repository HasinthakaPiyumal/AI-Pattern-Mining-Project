
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# For LIME
import lime
import lime.lime_tabular

# For SHAP
import shap

# --- Configuration --- #
MODEL_PATH = 'diagnosis_model.joblib'
PREPROCESSOR_PATH = 'preprocessor.joblib'

# --- 1. Data Layer: Data Generation & Preprocessing --- #
def generate_and_preprocess_data(n_samples=1000):
    np.random.seed(42)

    # Generate synthetic data
    data = {
        'Age': np.random.randint(20, 80, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Symptom1': np.random.choice(['Fever', 'Cough', 'Fatigue', 'None'], n_samples, p=[0.3, 0.3, 0.2, 0.2]),
        'Symptom2': np.random.choice(['Headache', 'Rash', 'Nausea', 'None'], n_samples, p=[0.25, 0.25, 0.25, 0.25]),
        'BloodPressure': np.random.normal(120, 15, n_samples),
        'Cholesterol': np.random.normal(200, 30, n_samples),
    }
    df = pd.DataFrame(data)

    # Simulate a diagnosis based on some rules (simplified for demonstration)
    conditions = [
        (df['Symptom1'] == 'Fever') & (df['Symptom2'] == 'Headache') & (df['Age'] > 40),
        (df['Symptom1'] == 'Cough') & (df['BloodPressure'] > 130),
        (df['Symptom1'] == 'Fatigue') & (df['Cholesterol'] > 220),
        (df['Symptom2'] == 'Rash') & (df['Age'] < 30),
    ]
    choices = ['Flu', 'Hypertension', 'Fatigue Syndrome', 'Allergy']
    df['Diagnosis'] = np.select(conditions, choices, default='Healthy')

    # Define features and target
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    # Define categorical and numerical features
    categorical_features = ['Gender', 'Symptom1', 'Symptom2']
    numerical_features = ['Age', 'BloodPressure', 'Cholesterol']

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    return X, y, preprocessor, categorical_features, numerical_features

# --- 2. Model Layer: Model Training --- #
def train_model(X, y, preprocessor):
    # Create a pipeline that first preprocesses and then trains the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train the model
    model_pipeline.fit(X, y)
    return model_pipeline

# --- 3. Model Persistence --- #
def save_artifacts(model, preprocessor, X_cols, y_classes):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(X_cols, 'feature_names.joblib')
    joblib.dump(y_classes, 'target_names.joblib')

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_cols = joblib.load('feature_names.joblib')
    y_classes = joblib.load('target_names.joblib')
    return model, preprocessor, X_cols, y_classes

# --- Helper function for getting feature names after preprocessing ---
def get_feature_names(column_transformer):
    output_features = []
    for name, preprocessor, features in column_transformer.transformers_:
        if name == 'num':
            output_features.extend(features)
        elif name == 'cat':
            output_features.extend(preprocessor.get_feature_names_out(features))
    return output_features

# --- Streamlit Application --- #
st.set_page_config(layout="wide", page_title="Interpretable Medical Diagnosis System")
st.title("🩺 Interpretable Medical Diagnosis Prediction System")

# --- Sidebar for Data & Model Management --- #
st.sidebar.header("System Configuration")

if st.sidebar.button("Generate New Data & Re-train Model"):
    with st.spinner("Generating data and training model..."):
        X_raw, y_raw, preprocessor_obj, cat_feats, num_feats = generate_and_preprocess_data()
        trained_model = train_model(X_raw, y_raw, preprocessor_obj)
        
        # Get feature names after preprocessing
        full_feature_names = get_feature_names(preprocessor_obj)
        target_class_names = trained_model.classes_

        save_artifacts(trained_model, preprocessor_obj, full_feature_names, target_class_names)
        st.session_state['model_trained'] = True
        st.session_state['X_raw'] = X_raw
        st.session_state['y_raw'] = y_raw
        st.session_state['cat_feats'] = cat_feats
        st.session_state['num_feats'] = num_feats
    st.sidebar.success("Data generated, model trained, and artifacts saved!")

# Load model and preprocessor if available
if 'model_trained' not in st.session_state or not st.session_state['model_trained']:
    try:
        model, preprocessor, feature_names_post_prep, target_names = load_artifacts()
        X_raw, y_raw, preprocessor_obj_dummy, cat_feats, num_feats = generate_and_preprocess_data() # Regenerate X_raw and y_raw for consistency
        st.session_state['model_trained'] = True
        st.session_state['model'] = model
        st.session_state['preprocessor'] = preprocessor
        st.session_state['feature_names_post_prep'] = feature_names_post_prep
        st.session_state['target_names'] = target_names
        st.session_state['X_raw'] = X_raw
        st.session_state['y_raw'] = y_raw
        st.session_state['cat_feats'] = cat_feats
        st.session_state['num_feats'] = num_feats
    except FileNotFoundError:
        st.warning("Model and preprocessor not found. Please click 'Generate New Data & Re-train Model' in the sidebar.")
        st.stop()
else:
    model = st.session_state['model']
    preprocessor = st.session_state['preprocessor']
    feature_names_post_prep = st.session_state['feature_names_post_prep']
    target_names = st.session_state['target_names']
    X_raw = st.session_state['X_raw']
    y_raw = st.session_state['y_raw']
    cat_feats = st.session_state['cat_feats']
    num_feats = st.session_state['num_feats']


# --- Patient Input/Selection --- #
st.header("Patient Data")

patient_option = st.radio(
    "Select patient data source:",
    ("Select existing patient", "Enter new patient data")
)

patient_data = None
patient_idx = None

if patient_option == "Select existing patient":
    patient_idx = st.selectbox("Select a patient ID:", X_raw.index)
    patient_data = X_raw.loc[patient_idx]
    st.write(f"Original Diagnosis: {y_raw.loc[patient_idx]}")
else:
    st.subheader("Enter New Patient Data")
    new_age = st.slider("Age", 20, 80, 45)
    new_gender = st.selectbox("Gender", ['Male', 'Female'])
    new_symptom1 = st.selectbox("Symptom 1", ['Fever', 'Cough', 'Fatigue', 'None'])
    new_symptom2 = st.selectbox("Symptom 2", ['Headache', 'Rash', 'Nausea', 'None'])
    new_blood_pressure = st.slider("Blood Pressure", 90, 180, 120)
    new_cholesterol = st.slider("Cholesterol", 100, 300, 200)

    patient_data = pd.DataFrame({
        'Age': [new_age],
        'Gender': [new_gender],
        'Symptom1': [new_symptom1],
        'Symptom2': [new_symptom2],
        'BloodPressure': [new_blood_pressure],
        'Cholesterol': [new_cholesterol],
    })
    # Reset index to avoid issues with .loc later if needed
    patient_data.reset_index(drop=True, inplace=True)

# Ensure patient_data is a DataFrame for consistent processing
if isinstance(patient_data, pd.Series):
    patient_data_df = pd.DataFrame([patient_data])
else:
    patient_data_df = patient_data.copy()

# --- Prediction --- #
if patient_data is not None:
    st.subheader("Prediction")
    predicted_diagnosis_proba = model.predict_proba(patient_data_df)[0]
    predicted_diagnosis_idx = np.argmax(predicted_diagnosis_proba)
    predicted_diagnosis = target_names[predicted_diagnosis_idx]
    
    st.success(f"Predicted Diagnosis: **{predicted_diagnosis}** (Confidence: {predicted_diagnosis_proba[predicted_diagnosis_idx]:.2f})")

    st.write("--- ")

    # --- 4. Interpretability Layer --- #
    st.header("Model Interpretability")

    # Preprocess the single patient data for interpretability tools that require it
    patient_data_processed = preprocessor.transform(patient_data_df)

    tab1, tab2, tab3 = st.tabs(["Local Explanations", "Global Explanations", "Subgroup Analysis"])

    with tab1:
        st.subheader("Local Interpretability (Instance-Specific)")

        st.markdown("### LIME (Local Interpretable Model-agnostic Explanations)")
        st.info("LIME explains individual predictions by creating a local surrogate model around the prediction.")

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=preprocessor.transform(X_raw.sample(500, random_state=42)), # Use a sample for speed
            feature_names=feature_names_post_prep,
            class_names=target_names,
            mode='classification'
        )
        
        # Explain the prediction for the target class
        explanation = explainer.explain_instance(
            data_row=patient_data_processed[0],
            predict_fn=model.predict_proba,
            num_features=5,
            top_labels=1 # Explain only the top predicted label
        )
        fig = explanation.as_pyplot_figure(label=predicted_diagnosis_idx)
        st.pyplot(fig)
        st.markdown("**LIME Explanation Details:**")
        for feature, weight in explanation.as_list(label=predicted_diagnosis_idx):
            st.write(f"- {feature}: {weight:.4f}")

        st.markdown("### SHAP (SHapley Additive exPlanations)")
        st.info("SHAP values assign each feature an importance value for a particular prediction.")

        # SHAP explainer requires raw input data and model's predict_proba
        # For tree models, TreeExplainer is more efficient
        if isinstance(model.named_steps['classifier'], RandomForestClassifier):
            explainer_shap = shap.TreeExplainer(model.named_steps['classifier'])
            # SHAP expects processed features, so we need to transform X_raw for background data
            X_processed_sample = preprocessor.transform(X_raw.sample(100, random_state=42))
            shap_values = explainer_shap.shap_values(patient_data_processed)
            
            # For multi-class, shap_values is a list of arrays, one for each class
            # We want explanation for the predicted class
            shap_values_for_predicted_class = shap_values[predicted_diagnosis_idx]

            # Plot the SHAP explanation (force plot is great for single instance)
            st.write(f"SHAP explanation for class: **{predicted_diagnosis}**")
            # Streamlit doesn't natively render JS-heavy plots like force plot. Use summary_plot for simplicity here.
            # For an interactive force plot, one would need to save to HTML or use a workaround.
            # As a workaround, we'll plot a bar chart of SHAP values.
            shap_df = pd.DataFrame({
                'Feature': feature_names_post_prep,
                'SHAP Value': shap_values_for_predicted_class[0]
            }).sort_values(by='SHAP Value', ascending=False)
            
            fig_shap = px.bar(shap_df, x='SHAP Value', y='Feature', orientation='h',
                              title=f'SHAP Values for {predicted_diagnosis}',
                              color='SHAP Value', color_continuous_scale=px.colors.sequential.Bluered)
            st.plotly_chart(fig_shap, use_container_width=True)
            
        else:
            st.warning("SHAP TreeExplainer is used. If using a different model, KernelExplainer might be needed (more computationally intensive).")
            # Example for KernelExplainer (more general but slower)
            # explainer_shap = shap.KernelExplainer(model.predict_proba, X_processed_sample)
            # shap_values = explainer_shap.explain_instance(patient_data_processed[0], top_labels=1)


        st.markdown("### Individual Conditional Expectation (ICE) Plots")
        st.info("ICE plots show how the prediction for an individual instance changes as a single feature varies.")
        
        selected_ice_feature = st.selectbox("Select feature for ICE plot:", num_feats, key='ice_feature_select')

        if selected_ice_feature:
            fig_ice, ax_ice = plt.subplots()
            
            # To get a single ICE curve, we need to pass a single row to PartialDependenceDisplay
            # However, sklearn's PartialDependenceDisplay with kind='individual' usually plots multiple individuals.
            # A workaround for a single individual is to create a small dataset just for that individual.
            
            # Create a dataframe for the single patient by repeating it for each value of the selected feature
            feature_values = np.linspace(X_raw[selected_ice_feature].min(), X_raw[selected_ice_feature].max(), 100)
            ice_df_raw = pd.concat([patient_data_df.copy()]*len(feature_values), ignore_index=True)
            ice_df_raw[selected_ice_feature] = feature_values

            # Predict probabilities for each diagnosis class
            predictions_proba = model.predict_proba(ice_df_raw)
            
            # Assuming we want the probability of the predicted diagnosis class
            predicted_class_idx = model.classes_.tolist().index(predicted_diagnosis)
            ice_curve = predictions_proba[:, predicted_class_idx]

            ax_ice.plot(feature_values, ice_curve, label=f'Probability of {predicted_diagnosis}')
            ax_ice.set_xlabel(selected_ice_feature)
            ax_ice.set_ylabel(f'Predicted Probability for {predicted_diagnosis}')
            ax_ice.set_title(f'ICE Plot for {selected_ice_feature} for Patient ID {patient_idx if patient_idx is not None else "New"}')
            ax_ice.grid(True)
            st.pyplot(fig_ice)

        st.markdown("### Counterfactual Explanations (Conceptual)")
        st.info("Counterfactual explanations answer 'What if?' questions, e.g., 'What is the smallest change to my symptoms that would have resulted in a different diagnosis?'")
        st.write("**Current Prediction:** Your predicted diagnosis is **" + predicted_diagnosis + "**.")
        st.write("**Conceptual Counterfactual:** To potentially change your diagnosis from **" + predicted_diagnosis + "** to a 'Healthy' outcome, you might need to:")
        st.markdown("- *If 'Fever'*: Reduce your fever severity or have 'None' as Symptom1.")
        st.markdown("- *If 'Hypertension'*: Lower your Blood Pressure significantly.")
        st.markdown("*(Note: A full implementation would use libraries like `dice-ml` for rigorous counterfactual generation.)*\n")

    with tab2:
        st.subheader("Global Interpretability (Overall Model Behavior)")

        st.markdown("### Partial Dependence Plots (PDP)")
        st.info("PDPs show the marginal effect of one or two features on the predicted outcome of a model, averaging over all other features.")

        selected_pdp_feature = st.selectbox("Select feature for PDP:", num_feats + cat_feats, key='pdp_feature_select')

        if selected_pdp_feature:
            fig_pdp, ax_pdp = plt.subplots(figsize=(8, 6))
            
            # PartialDependenceDisplay expects the model and preprocessed data, or a pipeline
            # We pass the full pipeline to ensure preprocessing is handled.
            try:
                display = PartialDependenceDisplay.from_estimator(
                    model,
                    X_raw,
                    features=[selected_pdp_feature],
                    feature_names=X_raw.columns.tolist(), # Use original feature names here
                    target=model.classes_.tolist().index(predicted_diagnosis), # Focus on the predicted diagnosis class
                    kind='average',
                    ax=ax_pdp
                )
                ax_pdp.set_title(f"Partial Dependence of {selected_pdp_feature} on {predicted_diagnosis} Probability")
                st.pyplot(fig_pdp)
            except Exception as e:
                st.error(f"Could not generate PDP for {selected_pdp_feature}. Error: {e}")
                st.warning("PDP for categorical features might require careful handling of `feature_names` and `features` parameters within PartialDependenceDisplay.")

        st.markdown("### Permutation Feature Importance (PFI)")
        st.info("PFI quantifies the importance of features by measuring the decrease in model performance when a feature's values are randomly shuffled.")

        if st.button("Calculate Permutation Feature Importance"):
            with st.spinner("Calculating PFI..."):
                # Use a smaller sample of X_raw for faster PFI calculation in Streamlit
                X_test_sample, y_test_sample = X_raw.sample(frac=0.3, random_state=42), y_raw.sample(frac=0.3, random_state=42)
                
                result = permutation_importance(model, X_test_sample, y_test_sample, n_repeats=10, random_state=42, n_jobs=-1)
                sorted_idx = result.importances_mean.argsort()

                fig_pfi, ax_pfi = plt.subplots()
                ax_pfi.boxplot(result.importances[sorted_idx].T,
                               vert=False, labels=X_raw.columns[sorted_idx])
                ax_pfi.set_title("Permutation Feature Importance (Overall Model)")
                ax_pfi.set_xlabel("Importance Score (Mean decrease in accuracy)")
                ax_pfi.set_ylabel("Feature")
                st.pyplot(fig_pfi)

    with tab3:
        st.subheader("Subgroup Analysis (Identifying Biases & Discrepancies)")
        st.info("Explore model predictions and explanations across different patient subgroups to identify potential biases or divergent behaviors.")

        st.markdown("### Filter Data for Subgroup Analysis")
        col1, col2 = st.columns(2)
        with col1:
            selected_gender = st.multiselect("Filter by Gender", ['Male', 'Female'], default=['Male', 'Female'])
        with col2:
            age_range = st.slider("Filter by Age Range", 20, 80, (20, 80))

        filtered_X_raw = X_raw[
            (X_raw['Gender'].isin(selected_gender)) &
            (X_raw['Age'] >= age_range[0]) &
            (X_raw['Age'] <= age_range[1])
        ]
        filtered_y_raw = y_raw.loc[filtered_X_raw.index]

        st.write(f"**Number of patients in selected subgroup:** {len(filtered_X_raw)}")
        st.write(f"**Diagnoses distribution in subgroup:**")
        st.write(filtered_y_raw.value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

        if len(filtered_X_raw) > 5:
            st.markdown("### Global Interpretability for Subgroup")
            st.write("Re-running Permutation Feature Importance for this subgroup to see if feature importance shifts.")

            if st.button("Calculate PFI for Subgroup"):
                with st.spinner("Calculating Subgroup PFI..."):
                    result_subgroup = permutation_importance(model, filtered_X_raw, filtered_y_raw, n_repeats=5, random_state=42, n_jobs=-1)
                    sorted_idx_subgroup = result_subgroup.importances_mean.argsort()

                    fig_pfi_subgroup, ax_pfi_subgroup = plt.subplots()
                    ax_pfi_subgroup.boxplot(result_subgroup.importances[sorted_idx_subgroup].T,
                                           vert=False, labels=filtered_X_raw.columns[sorted_idx_subgroup])
                    ax_pfi_subgroup.set_title(f"Permutation Feature Importance (Subgroup: {', '.join(selected_gender)}, Age {age_range[0]}-{age_range[1]})")
                    ax_pfi_subgroup.set_xlabel("Importance Score (Mean decrease in accuracy)")
                    ax_pfi_subgroup.set_ylabel("Feature")
                    st.pyplot(fig_pfi_subgroup)
        elif len(filtered_X_raw) > 0:
            st.warning("Not enough data points in the selected subgroup to reliably calculate global explanations like PFI.")
        else:
            st.warning("No patients found in the selected subgroup.")

else:
    st.warning("Please select or enter patient data to proceed with prediction and interpretability.")

