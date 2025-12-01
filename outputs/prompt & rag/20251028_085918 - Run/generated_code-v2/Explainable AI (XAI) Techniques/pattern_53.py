import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

# --- 1. Data Handling and Preprocessing (Synthetic Data) ---
def generate_synthetic_data(num_samples=100):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, num_samples),
        'BMI': np.random.uniform(18, 40, num_samples),
        'Cholesterol': np.random.uniform(150, 250, num_samples),
        'Blood_Pressure': np.random.randint(90, 180, num_samples),
        'Smoker': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'Family_History': np.random.choice([0, 1], num_samples, p=[0.6, 0.4])
    }
    df = pd.DataFrame(data)

    # Generate a target variable (Disease_Risk) based on some features
    df['Disease_Risk'] = (df['Age'] * 0.05 + df['BMI'] * 0.1 + df['Cholesterol'] * 0.02 +
                            df['Blood_Pressure'] * 0.03 + df['Smoker'] * 10 + df['Family_History'] * 8 +
                            np.random.normal(0, 5, num_samples)).apply(lambda x: 1 if x > 40 else 0)
    return df

# --- 2. Black-box Disease Prediction Model ---
def train_dummy_model(df):
    X = df.drop('Disease_Risk', axis=1)
    y = df['Disease_Risk']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns

# Generate data and train model
synthetic_data = generate_synthetic_data()
model, feature_names = train_dummy_model(synthetic_data)

# --- 3. ICE Plot Calculation Logic ---
def calculate_ice(model, patient_instance, feature_to_vary, feature_range_min, feature_range_max, num_steps=50):
    ice_data = []
    feature_values = np.linspace(feature_range_min, feature_range_max, num_steps)

    for val in feature_values:
        temp_instance = patient_instance.copy()
        temp_instance[feature_to_vary] = val
        # Ensure the instance is a DataFrame row for prediction
        prediction = model.predict_proba(pd.DataFrame([temp_instance]))[0][1] # Probability of disease
        ice_data.append({'Feature_Value': val, 'Predicted_Risk': prediction})

    return pd.DataFrame(ice_data)

# --- 4. Visualization Component ---
def plot_ice(ice_df, feature_to_vary, patient_risk):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=ice_df, x='Feature_Value', y='Predicted_Risk', ax=ax, color='blue', linewidth=2)
    ax.set_title(f'Individual Conditional Expectation (ICE) Plot for {feature_to_vary}')
    ax.set_xlabel(f'{feature_to_vary} Value')
    ax.set_ylabel('Predicted Disease Risk (Probability)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1)
    ax.axhline(patient_risk, color='red', linestyle=':', label='Current Patient Risk')
    ax.legend()
    st.pyplot(fig)

# --- 5. User Interface (Streamlit Application) ---
st.title('🩺 Personalized Patient Risk Explainer (ICE Plots)')
st.write('Explore how changes in a single feature affect a patient\'s predicted disease risk.')

st.sidebar.header('Patient Features Input')

# Input fields for patient features (using sliders for numerical, selectbox for categorical)
patient_data = {}

for feature in feature_names:
    if feature == 'Age':
        patient_data[feature] = st.sidebar.slider(f'Select {feature}', min_value=20, max_value=80, value=45)
    elif feature == 'BMI':
        patient_data[feature] = st.sidebar.slider(f'Select {feature}', min_value=18.0, max_value=40.0, value=25.0, step=0.1)
    elif feature == 'Cholesterol':
        patient_data[feature] = st.sidebar.slider(f'Select {feature}', min_value=150, max_value=250, value=200)
    elif feature == 'Blood_Pressure':
        patient_data[feature] = st.sidebar.slider(f'Select {feature}', min_value=90, max_value=180, value=120)
    elif feature == 'Smoker':
        patient_data[feature] = st.sidebar.selectbox(f'Is {feature}?', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', index=0)
    elif feature == 'Family_History':
        patient_data[feature] = st.sidebar.selectbox(f'Has {feature}?', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', index=0)

patient_instance_df = pd.DataFrame([patient_data], columns=feature_names)

st.header('Current Patient Information')
st.write(patient_instance_df)

# Predict current risk for the patient
current_risk = model.predict_proba(patient_instance_df)[0][1]
st.metric(label="Predicted Disease Risk", value=f"{current_risk:.2f}")

st.header('ICE Plot Explanation')
feature_to_explain = st.selectbox('Select a feature to explain with ICE Plot:', options=list(feature_names))

if st.button('Generate ICE Plot'):
    # Determine feature range dynamically or pre-define
    if feature_to_explain == 'Age':
        min_val, max_val = 20, 80
    elif feature_to_explain == 'BMI':
        min_val, max_val = 18.0, 40.0
    elif feature_to_explain == 'Cholesterol':
        min_val, max_val = 150, 250
    elif feature_to_explain == 'Blood_Pressure':
        min_val, max_val = 90, 180
    else: # For binary features, we can just show the two points
        min_val, max_val = 0, 1

    ice_df = calculate_ice(model, patient_data, feature_to_explain, min_val, max_val)
    plot_ice(ice_df, feature_to_explain, current_risk)