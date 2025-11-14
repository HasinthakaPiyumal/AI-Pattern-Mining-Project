import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

# --- 1. Data Ingestion and Preprocessing Layer (Simulated Data) ---

@st.cache_resource
def load_and_preprocess_data():
    # Simulate patient data
    np.random.seed(42)
    data_size = 1000
    
    # Features
    age = np.random.randint(20, 80, data_size)
    bmi = np.random.uniform(18, 35, data_size)
    blood_pressure_sys = np.random.randint(90, 180, data_size)
    blood_pressure_dia = np.random.randint(60, 110, data_size)
    cholesterol = np.random.randint(150, 300, data_size)
    glucose = np.random.randint(70, 200, data_size)
    smoking = np.random.choice([0, 1], data_size, p=[0.7, 0.3]) # 0: No, 1: Yes
    family_history = np.random.choice([0, 1], data_size, p=[0.6, 0.4]) # 0: No, 1: Yes

    # Simulate a target variable (disease diagnosis: 0=Healthy, 1=Disease A, 2=Disease B)
    # Simplified logic: higher age, BMI, BP, cholesterol, glucose, smoking, family history increase disease risk
    disease_prob_A = (age * 0.005 + bmi * 0.03 + blood_pressure_sys * 0.005 + cholesterol * 0.002 + glucose * 0.003 + smoking * 0.2 + family_history * 0.1)
    disease_prob_B = (age * 0.006 + bmi * 0.02 + blood_pressure_dia * 0.004 + cholesterol * 0.003 + glucose * 0.004 + smoking * 0.15 + family_history * 0.15)
    
    # Introduce some randomness and thresholding for classification
    random_noise = np.random.uniform(-0.5, 0.5, data_size)
    
    diagnosis = np.zeros(data_size, dtype=int)
    diagnosis[(disease_prob_A + random_noise > 0.8)] = 1  # Disease A
    diagnosis[(disease_prob_B + random_noise > 0.9) & (diagnosis == 0)] = 2 # Disease B (if not already Disease A)

    df = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'blood_pressure_sys': blood_pressure_sys,
        'blood_pressure_dia': blood_pressure_dia,
        'cholesterol': cholesterol,
        'glucose': glucose,
        'smoking': smoking,
        'family_history': family_history,
        'diagnosis': diagnosis
    })

    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    feature_names = X.columns.tolist()
    class_names = ['Healthy', 'Disease A', 'Disease B']

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, scaler, feature_names, class_names

# --- 2. Core AI Prediction Model (XGBoost) ---

@st.cache_resource
def train_model(X_train, y_train):
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    model.fit(X_train, y_train)
    return model

# --- 3. Interpretability & Explanation Layer ---

# Local Interpretability
def explain_lime(model, X_train_df, instance, feature_names, class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_df.values,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    explanation = explainer.explain_instance(
        data_row=instance.values[0],
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    return explanation

def explain_shap(model, X_train_df, instance):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(instance)
    return shap_values, explainer

# Global Interpretability
def plot_pdp(model, X_data, feature_names):
    fig, ax = plt.subplots(figsize=(10, 7))
    # Select a few features for PDP, or all if feasible
    features_to_plot = feature_names[:min(5, len(feature_names))] # Plot first 5 features
    PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=X_data,
        features=features_to_plot,
        feature_names=feature_names,
        target=0, # Assuming we want PDP for 'Healthy' class
        ax=ax
    )
    ax.set_title("Partial Dependence Plots (Target: Healthy)")
    plt.tight_layout()
    return fig

def plot_permutation_importance(model, X_test_df, y_test, feature_names):
    result = permutation_importance(model, X_test_df, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=np.array(feature_names)[sorted_idx])
    ax.set_title("Permutation Feature Importance")
    ax.set_ylabel("Features")
    ax.set_xlabel("Importance Score (Decrease in Accuracy)")
    plt.tight_layout()
    return fig

# --- 4. Interactive Debugging and Validation User Interface (Streamlit) ---

st.set_page_config(layout="wide", page_title="Interpretable AI for Disease Diagnosis")

st.title("🩺 Interpretable AI System for Disease Diagnosis")
st.markdown("This application demonstrates an AI system for disease diagnosis with integrated interpretability tools.")

# Load and preprocess data, train model
X_train, X_test, y_train, y_test, scaler, feature_names, class_names = load_and_preprocess_data()
model = train_model(X_train, y_train)

st.sidebar.header("Patient Input Features")

# Collect user input for a single patient
input_data = {}
for feature in feature_names:
    if feature == 'smoking' or feature == 'family_history':
        input_data[feature] = st.sidebar.selectbox(f"Select {feature.replace('_', ' ').title()}", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    else:
        min_val = X_train[feature].min() * scaler.scale_[feature_names.index(feature)] + scaler.mean_[feature_names.index(feature)]
        max_val = X_train[feature].max() * scaler.scale_[feature_names.index(feature)] + scaler.mean_[feature_names.index(feature)]
        default_val = float(X_train[feature].median() * scaler.scale_[feature_names.index(feature)] + scaler.mean_[feature_names.index(feature)])
        input_data[feature] = st.sidebar.slider(f"Enter {feature.replace('_', ' ').title()}", min_value=float(min_val), max_value=float(max_val), value=default_val, step=0.1)

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)
input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

if st.sidebar.button("Get Diagnosis and Explanations"):
    st.subheader("AI Diagnosis")
    prediction_proba = model.predict_proba(input_scaled_df)[0]
    predicted_class_idx = np.argmax(prediction_proba)
    predicted_diagnosis = class_names[predicted_class_idx]
    
    st.success(f"Predicted Diagnosis: **{predicted_diagnosis}** (Probability: {prediction_proba[predicted_class_idx]:.2f})")
    
    st.markdown("--- ")
    st.subheader("Local Interpretability (Why this specific diagnosis?)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### LIME Explanation")
        with st.spinner("Generating LIME explanation..."):
            lime_exp = explain_lime(model, X_train, input_scaled_df, feature_names, class_names)
            st.write("**Features contributing to the prediction:**")
            for feature, weight in lime_exp.as_list():
                st.write(f"- {feature}: {weight:.4f}")
            fig_lime = lime_exp.as_pyplot_figure()
            st.pyplot(fig_lime)

    with col2:
        st.write("#### SHAP Explanation")
        with st.spinner("Generating SHAP explanation..."):
            shap_values, explainer = explain_shap(model, input_scaled_df)
            # Assuming multi-output, get values for the predicted class
            if isinstance(shap_values, list):
                shap_values_predicted_class = shap_values[predicted_class_idx][0]
            else:
                shap_values_predicted_class = shap_values[0]
            
            fig_shap, ax = plt.subplots(figsize=(8, 6))
            shap.plot_waterfall(shap.Explanation(values=shap_values_predicted_class, 
                                                  base_values=explainer.expected_value[predicted_class_idx] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                                                  data=input_scaled_df.values[0],
                                                  feature_names=feature_names),
                                  show=False)
            ax.set_title(f"SHAP Values for {predicted_diagnosis}")
            st.pyplot(fig_shap)

    st.markdown("--- ")
    st.subheader("Global Interpretability (How does the model generally behave?)")
    
    st.write("#### Partial Dependence Plots")
    with st.spinner("Generating Partial Dependence Plots..."):
        pdp_fig = plot_pdp(model, X_train, feature_names)
        st.pyplot(pdp_fig)
        st.caption("Shows the marginal effect of selected features on the predicted probability of 'Healthy' class.")

    st.write("#### Permutation Feature Importance")
    with st.spinner("Calculating Permutation Feature Importance..."):
        perm_imp_fig = plot_permutation_importance(model, X_test, y_test, feature_names)
        st.pyplot(perm_imp_fig)
        st.caption("Quantifies the importance of features by measuring the decrease in model performance when feature values are randomly shuffled.")

else:
    st.info("Enter patient details in the sidebar and click 'Get Diagnosis and Explanations' to see the AI's prediction and its interpretability.")
