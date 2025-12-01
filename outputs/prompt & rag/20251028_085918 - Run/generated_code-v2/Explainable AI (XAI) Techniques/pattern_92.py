import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# --- 1. Data Simulation (data_generator.py logic) ---
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'Symptom_A': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'Symptom_B': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        'Lab_Result_X': np.random.normal(100, 15, num_samples),
        'Lab_Result_Y': np.random.normal(5, 1, num_samples),
    }
    df = pd.DataFrame(data)

    # Simple rule for diagnosis (simulated, not medically accurate)
    df['Diagnosis'] = ((df['Age'] > 50) * 0.4 + 
                       (df['Symptom_A'] == 1) * 0.3 + 
                       (df['Lab_Result_X'] < 80) * 0.2 + 
                       (df['Lab_Result_Y'] > 6) * 0.1 + 
                       np.random.rand(num_samples) * 0.5 > 0.6).astype(int)

    return df

# --- 2. Black-box Diagnostic Model (model.py logic) ---
MODEL_PATH = 'random_forest_model.joblib'

def train_and_save_model(df):
    df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    X = df_encoded.drop('Diagnosis', axis=1)
    y = df_encoded['Diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model, X.columns

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# --- 3. LACE-inspired Explanation Module (explainer.py logic) ---
class LACEExplainer:
    def __init__(self, model, feature_names, original_df):
        self.model = model
        self.feature_names = feature_names
        self.original_df = original_df

    def _perturb_instance(self, instance, feature_name, perturbation_value=None, perturbation_type='numeric'):
        perturbed_instance = instance.copy()
        if perturbation_type == 'numeric':
            perturbed_instance[feature_name] = perturbation_value if perturbation_value is not None else instance[feature_name] * 0.9
        elif perturbation_type == 'binary':
            perturbed_instance[feature_name] = 1 - instance[feature_name]
        elif perturbation_type == 'categorical':
            # For gender, switch between Male_Female (1) and (0) if the feature is one-hot encoded
            if feature_name == 'Gender_Male': # Assuming Male_Female is the encoded column
                perturbed_instance['Gender_Male'] = 1 - instance['Gender_Male']
        return perturbed_instance

    def explain_instance(self, instance_data, original_instance_df):
        explanation = {}
        original_prediction_proba = self.model.predict_proba(pd.DataFrame([instance_data]))[0]
        original_predicted_class = np.argmax(original_prediction_proba)

        for feature in self.feature_names:
            temp_instance = instance_data.copy()
            
            # Determine perturbation type based on feature name and original_df columns
            if feature in ['Age', 'Lab_Result_X', 'Lab_Result_Y']:
                # Numeric features: perturb by a fixed percentage (e.g., 10% decrease)
                if original_instance_df[feature].iloc[0] != 0:
                    perturbed_val = original_instance_df[feature].iloc[0] * 0.9
                else:
                    perturbed_val = 0.1 # Small constant if original is 0
                perturbed_instance_df = self._perturb_instance(original_instance_df.iloc[0], feature, perturbed_val, 'numeric')
                
            elif feature in ['Symptom_A', 'Symptom_B']:
                # Binary features: flip value
                perturbed_instance_df = self._perturb_instance(original_instance_df.iloc[0], feature, perturbation_type='binary')
            
            elif feature == 'Gender_Male': # Assuming one-hot encoded gender
                 perturbed_instance_df = self._perturb_instance(original_instance_df.iloc[0], feature, perturbation_type='categorical')
            else:
                continue # Skip other features if any
            
            # Convert perturbed_instance_df to a format suitable for the model
            # Ensure it has all feature columns, even if not perturbed
            perturbed_instance_for_model = perturbed_instance_df.reindex(columns=self.feature_names, fill_value=0).to_frame().T
            perturbed_prediction_proba = self.model.predict_proba(perturbed_instance_for_model)[0]
            
            # Impact is the absolute change in probability for the original predicted class
            impact = abs(original_prediction_proba[original_predicted_class] - perturbed_prediction_proba[original_predicted_class])
            explanation[feature] = impact
            
        # Sort explanations by impact
        sorted_explanation = sorted(explanation.items(), key=lambda item: item[1], reverse=True)
        return sorted_explanation

# --- Streamlit UI (app.py logic) ---
st.set_page_config(layout="wide", page_title="MedExplain")
st.title("MedExplain: Interactive Diagnosis Explanation")

@st.cache_resource
def setup_application():
    df = generate_synthetic_data()
    model, feature_cols = train_and_save_model(df)
    explainer = LACEExplainer(model, feature_cols, df.drop('Diagnosis', axis=1))
    return df, model, explainer, feature_cols

data, model, explainer, feature_names = setup_application()

if model is None:
    st.error("Model not found. Please ensure the model is trained and saved.")
else:
    st.sidebar.header("Patient Selection")
    patient_ids = data.index.tolist()
    selected_patient_id = st.sidebar.selectbox("Select Patient ID", patient_ids)

    st.sidebar.header("What-if Analysis")
    enable_what_if = st.sidebar.checkbox("Enable What-if Analysis")

    original_instance_data = data.loc[selected_patient_id].drop('Diagnosis')
    original_diagnosis = data.loc[selected_patient_id]['Diagnosis']

    # Prepare instance for model prediction (one-hot encode Gender)
    instance_for_prediction = original_instance_data.to_frame().T
    instance_for_prediction_encoded = pd.get_dummies(instance_for_prediction, columns=['Gender'], drop_first=True)
    
    # Ensure all model features are present, fill missing with 0
    instance_for_prediction_encoded = instance_for_prediction_encoded.reindex(columns=feature_names, fill_value=0)

    current_instance_data = instance_for_prediction_encoded.iloc[0].to_dict()
    current_instance_df = instance_for_prediction_encoded

    if enable_what_if:
        st.sidebar.subheader("Adjust Patient Attributes")
        what_if_changes = {}
        for feature in original_instance_data.index:
            if feature == 'Gender':
                current_val = original_instance_data[feature]
                new_val = st.sidebar.radio(f"Gender (Original: {current_val})", ['Male', 'Female'], index=0 if current_val == 'Male' else 1, key=f"what_if_{feature}")
                if new_val != current_val:
                    what_if_changes[feature] = new_val
            elif data[feature].dtype == 'int64' and feature not in ['Diagnosis']:
                current_val = int(original_instance_data[feature])
                new_val = st.sidebar.slider(f"{feature} (Original: {current_val})", int(data[feature].min()), int(data[feature].max()), current_val, key=f"what_if_{feature}")
                if new_val != current_val:
                    what_if_changes[feature] = new_val
            elif data[feature].dtype == 'float64':
                current_val = float(original_instance_data[feature])
                min_val, max_val = data[feature].min(), data[feature].max()
                new_val = st.sidebar.slider(f"{feature} (Original: {current_val:.2f})", float(min_val), float(max_val), current_val, format="%.2f", key=f"what_if_{feature}")
                if new_val != current_val:
                    what_if_changes[feature] = new_val
        
        # Apply what-if changes to the instance data for prediction and explanation
        if what_if_changes:
            temp_instance_df = original_instance_data.to_frame().T.copy()
            for feat, val in what_if_changes.items():
                temp_instance_df[feat] = val
            
            current_instance_df = pd.get_dummies(temp_instance_df, columns=['Gender'], drop_first=True).reindex(columns=feature_names, fill_value=0)
            current_instance_data = current_instance_df.iloc[0].to_dict()


    st.header(f"Patient ID: {selected_patient_id}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Data")
        display_data = original_instance_data.to_dict()
        if enable_what_if and what_if_changes:
            st.write("**Original vs. What-if Attributes:**")
            for feat, original_val in original_instance_data.items():
                if feat in what_if_changes:
                    st.write(f"- **{feat}:** {original_val} -> **{what_if_changes[feat]}**")
                else:
                    st.write(f"- {feat}: {original_val}")
        else:
            st.dataframe(original_instance_data)


        st.subheader("Model Prediction")
        prediction_proba = model.predict_proba(current_instance_df)[0]
        predicted_class = np.argmax(prediction_proba)
        diagnosis_label = "Disease" if predicted_class == 1 else "Healthy"
        
        st.markdown(f"**Predicted Diagnosis:** <span style='font-size:24px; color:{'red' if predicted_class == 1 else 'green'};'>{diagnosis_label}</span>", unsafe_allow_html=True)
        st.write(f"Probability of Disease: {prediction_proba[1]:.2f}")
        st.write(f"Probability of Healthy: {prediction_proba[0]:.2f}")
        st.write(f"True Diagnosis (for reference): {'Disease' if original_diagnosis == 1 else 'Healthy'}")

    with col2:
        st.subheader("Explanation (LACE-inspired)")
        explanation = explainer.explain_instance(current_instance_data, current_instance_df)
        if explanation:
            st.write("Features ranked by their impact on the current prediction:")
            for feature, impact in explanation:
                st.write(f"- **{feature}**: {impact:.4f} (Impact on prediction)")
        else:
            st.write("No explanation available for this instance.")


