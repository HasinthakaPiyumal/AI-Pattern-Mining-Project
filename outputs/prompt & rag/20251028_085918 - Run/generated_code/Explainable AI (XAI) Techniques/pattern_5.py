import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

# --- 1. Simulate Data and Train Model ---
@st.cache_resource
def load_data_and_train_model():
    # Simulate patient data
    np.random.seed(42)
    data_size = 500
    
    ages = np.random.randint(20, 80, data_size)
    genders = np.random.choice(['Male', 'Female'], data_size)
    symptoms_fever = np.random.randint(0, 2, data_size) # 0: No, 1: Yes
    symptoms_cough = np.random.randint(0, 2, data_size)
    symptoms_fatigue = np.random.randint(0, 2, data_size)
    lab_wbc = np.random.normal(7000, 2000, data_size).astype(int) # White Blood Cell count
    lab_crp = np.random.normal(5, 3, data_size).astype(int) # C-reactive protein
    
    # Simulate a target variable: Diagnosis (e.g., 'Flu', 'Common Cold', 'Bacterial Infection', 'Healthy')
    # This is a highly simplified simulation for demonstration purposes
    diagnoses = []
    for i in range(data_size):
        if symptoms_fever[i] == 1 and lab_wbc[i] > 8000 and lab_crp[i] > 7:
            diagnoses.append('Bacterial Infection')
        elif symptoms_fever[i] == 1 and symptoms_cough[i] == 1 and lab_crp[i] < 5:
            diagnoses.append('Flu')
        elif symptoms_cough[i] == 1 and symptoms_fatigue[i] == 1:
            diagnoses.append('Common Cold')
        else:
            diagnoses.append('Healthy')
            
    df = pd.DataFrame({
        'Age': ages,
        'Gender': genders,
        'Fever': symptoms_fever,
        'Cough': symptoms_cough,
        'Fatigue': symptoms_fatigue,
        'WBC': lab_wbc,
        'CRP': lab_crp,
        'Diagnosis': diagnoses
    })
    
    # Preprocessing
    le_gender = LabelEncoder()
    df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
    
    le_diagnosis = LabelEncoder()
    df['Diagnosis_encoded'] = le_diagnosis.fit_transform(df['Diagnosis'])
    
    features = ['Age', 'Gender_encoded', 'Fever', 'Cough', 'Fatigue', 'WBC', 'CRP']
    X = df[features]
    y = df['Diagnosis_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, y_train, features, le_gender, le_diagnosis, df

model, X_train, y_train, features, le_gender, le_diagnosis, original_df = load_data_and_train_model()
class_names = le_diagnosis.classes_

# --- 2. Explanation Functions ---
def get_local_explanation(model, instance, feature_names, class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    explanation = explainer.explain_instance(
        data_row=instance.values,
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    return explanation

def plot_permutation_importance(model, X_test, y_test, feature_names):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=np.array(feature_names)[sorted_idx])
    ax.set_title("Permutation Feature Importance")
    ax.set_ylabel("Features")
    ax.set_xlabel("Importance Score (mean decrease in accuracy)")
    fig.tight_layout()
    return fig

def plot_partial_dependence(model, X_train, features):
    fig, ax = plt.subplots(figsize=(15, 8))
    display = PartialDependenceDisplay.from_estimator(
        model, X_train, features, kind="average", ax=ax, 
        feature_names=features, 
        line_kw={'color': 'blue'}
    )
    fig.suptitle("Partial Dependence Plots")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    return fig

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Healthcare AI Diagnosis Explainer")

st.title("🩺 Healthcare AI Diagnosis Explainer")
st.markdown("Understand the reasoning behind AI-powered medical diagnoses.")

st.sidebar.header("Patient Data Input")

with st.sidebar.form("patient_form"):
    age = st.slider("Age", 1, 100, 45)
    gender_input = st.selectbox("Gender", ['Male', 'Female'])
    fever = st.checkbox("Fever")
    cough = st.checkbox("Cough")
    fatigue = st.checkbox("Fatigue")
    wbc = st.slider("WBC Count", 1000, 20000, 7500)
    crp = st.slider("CRP Level", 1, 50, 6)
    
    submitted = st.form_submit_button("Get Diagnosis and Explanation")

if submitted:
    # Prepare input for prediction
    gender_encoded_input = le_gender.transform([gender_input])[0]
    input_data = pd.DataFrame([[age, gender_encoded_input, int(fever), int(cough), int(fatigue), wbc, crp]],
                                columns=features)
    
    # Make prediction
    prediction_proba = model.predict_proba(input_data)[0]
    predicted_class_idx = np.argmax(prediction_proba)
    predicted_diagnosis = le_diagnosis.inverse_transform([predicted_class_idx])[0]
    
    st.header(f"Predicted Diagnosis: :blue[{predicted_diagnosis}] (Confidence: {prediction_proba[predicted_class_idx]:.2f})")
    
    st.subheader("Local Explanation for this Diagnosis")
    explanation = get_local_explanation(model, input_data.iloc[0], features, class_names)
    
    # Display local explanation
    exp_df = pd.DataFrame(explanation.as_list(), columns=['Feature', 'Contribution'])
    st.dataframe(exp_df)
    
    st.markdown("The table above shows how each feature contributed positively or negatively to the predicted diagnosis. Positive contributions support the predicted diagnosis, while negative contributions suggest other diagnoses.")
    
    st.subheader("Counterfactual Explanation (What-If Scenario)")
    st.write("Adjust a feature below to see how the diagnosis or its explanation changes.")
    
    col1, col2 = st.columns(2)
    with col1:
        cf_feature = st.selectbox("Feature to change", features, index=0)
    with col2:
        # Determine appropriate input widget based on feature type
        current_value = input_data[cf_feature].iloc[0]
        if cf_feature == 'Age':
            cf_value = st.slider(f"New {cf_feature} value", 1, 100, current_value)
        elif cf_feature == 'Gender_encoded':
            cf_value_display = st.selectbox(f"New {le_gender.inverse_transform([current_value])[0]} value", le_gender.classes_, index=current_value)
            cf_value = le_gender.transform([cf_value_display])[0]
        elif cf_feature in ['Fever', 'Cough', 'Fatigue']:
            cf_value = st.checkbox(f"New {cf_feature} value (Yes)", value=bool(current_value))
            cf_value = int(cf_value)
        elif cf_feature == 'WBC':
            cf_value = st.slider(f"New {cf_feature} value", 1000, 20000, current_value)
        elif cf_feature == 'CRP':
            cf_value = st.slider(f"New {cf_feature} value", 1, 50, current_value)
        else:
            cf_value = st.text_input(f"New {cf_feature} value", value=current_value)

    if st.button("Apply Counterfactual and Re-Explain"):
        cf_input_data = input_data.copy()
        cf_input_data[cf_feature] = cf_value
        
        cf_prediction_proba = model.predict_proba(cf_input_data)[0]
        cf_predicted_class_idx = np.argmax(cf_prediction_proba)
        cf_predicted_diagnosis = le_diagnosis.inverse_transform([cf_predicted_class_idx])[0]
        
        st.subheader(f"Counterfactual Diagnosis: :orange[{cf_predicted_diagnosis}] (Confidence: {cf_prediction_proba[cf_predicted_class_idx]:.2f})")
        st.write(f"If {cf_feature} was changed to {cf_value} (or {le_gender.inverse_transform([cf_value])[0] if cf_feature == 'Gender_encoded' else ''}), the diagnosis would be {cf_predicted_diagnosis}.")
        
        st.subheader("Local Explanation for Counterfactual Diagnosis")
        cf_explanation = get_local_explanation(model, cf_input_data.iloc[0], features, class_names)
        cf_exp_df = pd.DataFrame(cf_explanation.as_list(), columns=['Feature', 'Contribution'])
        st.dataframe(cf_exp_df)

st.subheader("Global Model Explanations")
tabs = st.tabs(["Permutation Feature Importance", "Partial Dependence Plots"])

with tabs[0]:
    st.markdown("### Permutation Feature Importance (PFI)")
    st.write("PFI measures how much the model's prediction accuracy decreases when a single feature's values are randomly shuffled. A larger decrease indicates higher importance.")
    if st.button("Show Permutation Feature Importance"):
        pfi_fig = plot_permutation_importance(model, X_train, y_train, features) # Using X_train for simplicity
        st.pyplot(pfi_fig)
        st.caption("Note: PFI calculated on training data for demonstration. In a real scenario, it's often run on a validation/test set.")

with tabs[1]:
    st.markdown("### Partial Dependence Plots (PDPs)")
    st.write("PDPs show the marginal effect of one or two features on the predicted outcome of a machine learning model, averaging over the values of all other features.")
    if st.button("Show Partial Dependence Plots"):
        pdp_fig = plot_partial_dependence(model, X_train, features)
        st.pyplot(pdp_fig)
        st.caption("Note: PDPs calculated on training data for demonstration.")
