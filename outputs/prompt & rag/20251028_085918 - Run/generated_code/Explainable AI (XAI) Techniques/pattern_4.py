import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Ingestion and Preprocessing (Simulated) ---
@st.cache_resource
def load_and_preprocess_data():
    # Simulate a medical dataset
    np.random.seed(42)
    n_samples = 1000
    data = {
        'Age': np.random.randint(20, 80, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Blood_Pressure_Systolic': np.random.randint(100, 180, n_samples),
        'Blood_Pressure_Diastolic': np.random.randint(60, 110, n_samples),
        'Cholesterol': np.random.randint(150, 300, n_samples),
        'HDL': np.random.randint(30, 80, n_samples),
        'BMI': np.random.uniform(18.0, 35.0, n_samples),
        'Smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Family_History': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'Diagnosis': np.random.choice(['Healthy', 'Condition A', 'Condition B'], n_samples, p=[0.6, 0.25, 0.15])
    }
    df = pd.DataFrame(data)

    # Define target and features
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Create preprocessor pipeline
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return X, y, preprocessor, numerical_features, categorical_features, df

# --- 2. Predictive AI Model --- 
@st.cache_resource
def train_model(X, y, preprocessor):
    # Create a pipeline with preprocessor and classifier
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train the model
    model_pipeline.fit(X, y)
    return model_pipeline

# --- Main Application Logic --- 

st.set_page_config(layout="wide", page_title="Explainable AI for Medical Diagnosis")
st.title("🩺 Explainable AI for Medical Diagnosis")
st.write("Understand the factors driving AI diagnostic predictions.")

# Load data and train model
X, y, preprocessor, numerical_features, categorical_features, df_raw = load_and_preprocess_data()
model = train_model(X, y, preprocessor)

# Get feature names after one-hot encoding for SHAP
# This requires fitting the preprocessor separately to get feature names
preprocessor.fit(X)
encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
all_feature_names = numerical_features.tolist() + encoded_feature_names

# Create a SHAP explainer for the trained model
# We need a function that takes raw features and returns predictions
def model_predict(data_point):
    data_df = pd.DataFrame([data_point], columns=X.columns)
    return model.predict_proba(data_df)

# explainer = shap.KernelExplainer(model_predict, shap.sample(X_processed, 100)) # KernelExplainer can be slow for large datasets
# For tree-based models, TreeExplainer is much faster and more accurate

# Get preprocessed data for global explanations
X_processed = preprocessor.transform(X)

# Using TreeExplainer for Random Forest
# The explainer needs to be fitted on the preprocessed training data
explainer = shap.TreeExplainer(model.named_steps['classifier'], feature_perturbation="tree_path_dependent")
# shap_values_global = explainer.shap_values(X_processed)

# Sidebar for user input
st.sidebar.header("Patient Data Input")

user_input = {}
for feature in numerical_features:
    min_val = df_raw[feature].min()
    max_val = df_raw[feature].max()
    mean_val = df_raw[feature].mean()
    user_input[feature] = st.sidebar.slider(f"Enter {feature}", float(min_val), float(max_val), float(mean_val))

for feature in categorical_features:
    options = df_raw[feature].unique().tolist()
    user_input[feature] = st.sidebar.selectbox(f"Select {feature}", options)

user_input['Smoking'] = st.sidebar.selectbox("Smoking History", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
user_input['Family_History'] = st.sidebar.selectbox("Family Medical History", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

st.sidebar.markdown("---")
st.sidebar.markdown("**Global Explanations**")
show_global_shap_summary = st.sidebar.button("Show Global Feature Importance")


# --- Main Content Area --- 
st.subheader("AI Diagnosis and Explanations")

if st.button("Get Diagnosis and Explanation"):
    st.markdown("### Prediction")
    prediction_proba = model.predict_proba(input_df)[0]
    prediction_class = model.predict(input_df)[0]

    st.write(f"The AI predicts: **{prediction_class}**")
    st.write(f"Prediction Confidence:")
    for i, prob in enumerate(prediction_proba):
        st.write(f"- {model.classes_[i]}: {prob:.2f}")

    st.markdown("### Local Explanation (Why this specific diagnosis?)")
    # Preprocess the single user input for SHAP
    input_processed = preprocessor.transform(input_df)

    # Get SHAP values for the predicted class
    # We need to map the output index to the predicted class index
    predicted_class_idx = np.where(model.classes_ == prediction_class)[0][0]

    shap_values_local = explainer.shap_values(input_processed)

    # If shap_values_local is a list (for multi-output models), select the relevant array
    if isinstance(shap_values_local, list):
        shap_values_local_for_class = shap_values_local[predicted_class_idx][0] # [0] because it's a single instance
    else:
        shap_values_local_for_class = shap_values_local[0] # Single output, single instance

    # Create a SHAP explanation object for the force plot
    # base_value is the expected value of the model's output for the given class
    base_value = explainer.expected_value[predicted_class_idx] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value

    # Ensure feature names match the SHAP values
    # For the force plot, we need original feature names mapped to processed values
    # This part is a bit tricky with `ColumnTransformer` and `OneHotEncoder`
    # The input_processed values correspond to the `all_feature_names` list

    # To make the force plot work nicely, we will create a temporary Explanation object
    # that uses the correct feature names and values.

    # Get the values for the input instance after preprocessing
    instance_processed_values = input_processed[0]

    # Create an explainer for the single instance for visualization
    shap.initjs()
    fig_force = shap.force_plot(
        base_value,
        shap_values_local_for_class,
        instance_processed_values,
        feature_names=all_feature_names,
        matplotlib=True, # Render with matplotlib for Streamlit
        show=False # Prevent immediate display
    )
    st.pyplot(fig_force, bbox_inches='tight')
    st.write("The force plot shows how each feature contributes to pushing the prediction from the base value (average prediction) to the final predicted value for the selected class. Red indicates features increasing the risk of the predicted condition, blue indicates features decreasing it.")


if show_global_shap_summary:
    st.markdown("### Global Feature Importance (Overall Model)")
    st.write("This plot summarizes the impact of each feature across all predictions.")

    # Calculate SHAP values for the entire dataset (or a sample for faster computation)
    # For global explanations, it's common to compute shap values once or on a sample
    with st.spinner('Calculating global SHAP values... This may take a moment.'):
        # Use a sample of the processed data for global explanation to keep it fast
        sample_size_global = min(500, X_processed.shape[0])
        X_processed_sample = shap.sample(pd.DataFrame(X_processed, columns=all_feature_names), sample_size_global)
        shap_values_global = explainer.shap_values(X_processed_sample)

    # SHAP summary plot for all classes (if multiclass)
    if isinstance(shap_values_global, list):
        # For multi-output, shap_values_global is a list of arrays, one for each class.
        # We can plot for a specific class or combine them (more complex).
        # Let's plot for the first class as an example.
        # Or better, use shap.summary_plot which handles multiclass automatically when passed a list.
        fig_summary, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_global, X_processed_sample, feature_names=all_feature_names, plot_type="bar", show=False)
        st.pyplot(fig_summary, bbox_inches='tight')

        fig_beeswarm, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values_global, X_processed_sample, feature_names=all_feature_names, show=False)
        st.pyplot(fig_beeswarm, bbox_inches='tight')
    else:
        # Binary classification
        fig_summary, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_global, X_processed_sample, feature_names=all_feature_names, plot_type="bar", show=False)
        st.pyplot(fig_summary, bbox_inches='tight')

        fig_beeswarm, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values_global, X_processed_sample, feature_names=all_feature_names, show=False)
        st.pyplot(fig_beeswarm, bbox_inches='tight')
    
    st.write("The bar plot shows the average magnitude of SHAP values for each feature, indicating overall importance. The beeswarm plot shows the distribution of SHAP values for each feature, revealing how different feature values impact the output and the direction of that impact.")

    st.markdown("### Global Feature Dependence (How a feature affects the outcome)")
    st.write("Select a feature to see how its value influences the prediction for a specific class.")

    # Ensure shap_values_global is available from the previous calculation
    if 'shap_values_global' in locals():
        selected_feature_for_dependence = st.selectbox(
            "Select a feature for Dependence Plot:",
            options=all_feature_names
        )
        selected_class_for_dependence = st.selectbox(
            "Select a class for Dependence Plot:",
            options=model.classes_
        )
        selected_class_idx = np.where(model.classes_ == selected_class_for_dependence)[0][0]

        fig_dependence, ax = plt.subplots(figsize=(10, 6))
        if isinstance(shap_values_global, list):
            shap.dependence_plot(
                selected_feature_for_dependence, 
                shap_values_global[selected_class_idx], 
                X_processed_sample, 
                feature_names=all_feature_names, 
                interaction_index=None, # or another feature for interaction
                show=False
            )
        else:
            shap.dependence_plot(
                selected_feature_for_dependence, 
                shap_values_global, 
                X_processed_sample, 
                feature_names=all_feature_names, 
                interaction_index=None, # or another feature for interaction
                show=False
            )
        st.pyplot(fig_dependence, bbox_inches='tight')
        st.write("This plot shows the relationship between a feature's value and its impact on the model's output for a chosen class. Each dot is a patient, and the y-axis is the SHAP value for the selected feature.")

st.markdown("---")
st.info("Disclaimer: This is a simulated demonstration for educational purposes and should not be used for actual medical diagnosis.")