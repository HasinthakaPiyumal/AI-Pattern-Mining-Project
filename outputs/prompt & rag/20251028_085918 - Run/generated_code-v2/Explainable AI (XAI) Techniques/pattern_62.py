import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Data Simulation Module
def generate_synthetic_data(num_patients=100):
    np.random.seed(42)
    data = {
        "Age": np.random.randint(20, 80, num_patients),
        "BMI": np.random.uniform(18.0, 40.0, num_patients),
        "Blood Sugar Level": np.random.uniform(70, 200, num_patients),
        "Family History": np.random.choice([0, 1], num_patients, p=[0.7, 0.3]),
        "Activity Level": np.random.uniform(1, 10, num_patients), # 1 (low) to 10 (high)
        "Smoking Status": np.random.choice([0, 1], num_patients, p=[0.8, 0.2]),
        "Cholesterol": np.random.uniform(150, 250, num_patients),
        "Blood Pressure": np.random.uniform(90, 180, num_patients),
    }
    df = pd.DataFrame(data)

    # Simulate 'Disease Risk' (binary target)
    # A simple non-linear relationship for demonstration
    df["Disease Risk"] = (
        (df["Age"] * 0.02)
        + (df["BMI"] * 0.1)
        + (df["Blood Sugar Level"] * 0.015)
        + (df["Family History"] * 0.5)
        - (df["Activity Level"] * 0.05)
        + (df["Smoking Status"] * 0.4)
        + (df["Cholesterol"] * 0.005)
        + (df["Blood Pressure"] * 0.005)
        + np.random.normal(0, 0.5, num_patients)
    )
    df["Disease Risk"] = (df["Disease Risk"] > np.percentile(df["Disease Risk"], 70)).astype(int) # Top 30% are high risk
    return df

# 2. Black-Box Model Module
@st.cache_resource
def train_risk_model(data_df):
    X = data_df.drop("Disease Risk", axis=1)
    y = data_df["Disease Risk"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

# 3. ICE Explainer Module
def generate_ice_data(model, patient_instance, feature_to_vary, feature_range_min, feature_range_max, num_steps=50):
    varied_feature_values = np.linspace(feature_range_min, feature_range_max, num_steps)
    ice_predictions = []

    # Create a DataFrame for the patient instance, ensuring it has the same columns as the training data
    base_instance_df = pd.DataFrame([patient_instance.values], columns=patient_instance.index)

    for value in varied_feature_values:
        modified_instance = base_instance_df.copy()
        modified_instance[feature_to_vary] = value
        # Predict probability for the positive class (risk = 1)
        pred_prob = model.predict_proba(modified_instance)[:, 1][0]
        ice_predictions.append(pred_prob)

    return np.array(ice_predictions), varied_feature_values

def plot_ice_data(ice_data, feature_values, feature_name, original_feature_value, patient_id):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=feature_values, y=ice_data, ax=ax, label=f"Patient {patient_id} ICE")
    ax.axvline(x=original_feature_value, color='red', linestyle='--', label=f"Original {feature_name}: {original_feature_value:.2f}")
    ax.set_title(f"Individual Conditional Expectation (ICE) Plot for Patient {patient_id} - {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Predicted Disease Risk Probability")
    ax.set_ylim(0, 1)
    ax.legend()
    st.pyplot(fig)

# 4. Streamlit Application
st.set_page_config(layout="wide")
st.title("🏥 Patient Risk Explainer using ICE Plots")
st.markdown("Understand individual patient risk predictions by varying one feature at a time.")

# Generate data and train model (cached)
data_df = generate_synthetic_data(num_patients=500)
model, feature_names = train_risk_model(data_df)

st.sidebar.header("Patient and Feature Selection")

# Patient selection
patient_ids = data_df.index.tolist()
selected_patient_id = st.sidebar.selectbox("Select a Patient ID", patient_ids)
selected_patient_instance = data_df.loc[selected_patient_id].drop("Disease Risk")

# Feature selection
selected_feature = st.sidebar.selectbox("Select Feature to Vary", feature_names)

st.subheader(f"Details for Patient ID: {selected_patient_id}")
col1, col2 = st.columns(2)
with col1:
    st.write("**Original Patient Characteristics:**")
    st.dataframe(selected_patient_instance.to_frame(), use_container_width=True)

with col2:
    initial_prediction = model.predict_proba(selected_patient_instance.to_frame().T)[:, 1][0]
    st.write(f"**Initial Predicted Disease Risk:** {initial_prediction:.4f}")
    st.info("This is the model's prediction for the patient's actual characteristics.")

st.subheader(f"ICE Plot for '{selected_feature}'")

# Get feature min/max for range sliders
feature_min = data_df[selected_feature].min()
feature_max = data_df[selected_feature].max()
original_feature_value = selected_patient_instance[selected_feature]

# Sliders for feature range and steps
st.sidebar.markdown("--- ")
st.sidebar.subheader(f"Range for '{selected_feature}'")
range_min = st.sidebar.slider(f"Minimum value for {selected_feature}", float(feature_min), float(feature_max), float(original_feature_value * 0.8), 0.1)
range_max = st.sidebar.slider(f"Maximum value for {selected_feature}", float(feature_min), float(feature_max), float(original_feature_value * 1.2), 0.1)
num_steps = st.sidebar.slider("Number of steps for variation", 10, 200, 50)

if st.button("Generate ICE Plot"):
    if range_min >= range_max:
        st.error("Minimum range value must be less than maximum range value.")
    else:
        ice_predictions, varied_feature_values = generate_ice_data(
            model,
            selected_patient_instance,
            selected_feature,
            range_min,
            range_max,
            num_steps
        )
        plot_ice_data(ice_predictions, varied_feature_values, selected_feature, original_feature_value, selected_patient_id)
else:
    st.info("Click 'Generate ICE Plot' to see how changing the selected feature affects the patient's risk.")
