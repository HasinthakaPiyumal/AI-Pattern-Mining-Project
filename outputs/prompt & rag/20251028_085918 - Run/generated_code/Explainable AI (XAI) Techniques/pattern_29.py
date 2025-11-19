import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from lime import lime_tabular
import matplotlib.pyplot as plt

# --- 1. Data Generation ---
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 80, n_samples),
        'blood_pressure': np.random.randint(90, 180, n_samples),
        'cholesterol': np.random.randint(120, 280, n_samples),
        'glucose': np.random.randint(70, 200, n_samples),
        'bmi': np.random.uniform(18.0, 40.0, n_samples)
    }
    df = pd.DataFrame(data)

    # Create a synthetic 'diagnosis' (0 or 1) based on a simple rule
    # More complex relationships can be added for realism
    df['diagnosis'] = ((df['age'] > 50) * 0.3 +
                       (df['blood_pressure'] > 140) * 0.2 +
                       (df['cholesterol'] > 200) * 0.25 +
                       (df['glucose'] > 120) * 0.15 +
                       (df['bmi'] > 25) * 0.1).apply(lambda x: 1 if x > 0.6 else 0)

    return df

# --- 2. Model Training ---
@st.cache_resource
def train_model(df):
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_test, X.columns.tolist()

# --- Main Streamlit Application ---
st.set_page_config(layout="wide", page_title="MedLens: Interpretable AI for Diagnosis")

st.title("🩺 MedLens: Interpretable AI Assistant for Medical Diagnosis")
st.markdown("Understand your AI's medical predictions with local and global explanations.")

# Generate and train data/model
df_synthetic = generate_synthetic_data()
model, X_train, X_test, y_test, feature_names = train_model(df_synthetic)

# LIME Explainer
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=['No Disease', 'Disease'],
    mode='classification'
)

st.sidebar.header("Patient Input Features")

with st.sidebar: # Using a sidebar for input for better layout
    st.subheader("Enter patient's details:")
    input_age = st.slider("Age", 20, 80, 55)
    input_blood_pressure = st.slider("Blood Pressure (mmHg)", 90, 180, 130)
    input_cholesterol = st.slider("Cholesterol (mg/dL)", 120, 280, 200)
    input_glucose = st.slider("Glucose (mg/dL)", 70, 200, 100)
    input_bmi = st.slider("BMI (kg/m²)", 18.0, 40.0, 25.0, step=0.1)

    input_data = pd.DataFrame([[input_age, input_blood_pressure, input_cholesterol, input_glucose, input_bmi]],
                              columns=feature_names)

    if st.button("Get Diagnosis and Explanation"): 
        st.session_state['show_diagnosis'] = True
        st.session_state['input_data'] = input_data
    else:
        if 'show_diagnosis' not in st.session_state:
            st.session_state['show_diagnosis'] = False


col1, col2 = st.columns([1, 1])

if st.session_state['show_diagnosis']:
    with col1:
        st.subheader("Individual Patient Diagnosis")
        prediction_proba = model.predict_proba(st.session_state['input_data'])[0]
        prediction_class = np.argmax(prediction_proba)
        diagnosis_label = 'Disease' if prediction_class == 1 else 'No Disease'

        st.write(f"**Predicted Diagnosis:** **{diagnosis_label}**")
        st.write(f"Probability of Disease: {prediction_proba[1]:.2f}")
        st.write(f"Probability of No Disease: {prediction_proba[0]:.2f}")

        st.subheader("Local Explanation (LIME)")
        # Explain the instance
        exp = explainer.explain_instance(
            data_row=st.session_state['input_data'].values[0],
            predict_fn=model.predict_proba,
            num_features=len(feature_names)
        )

        fig_lime = exp.as_pyplot_figure()
        st.pyplot(fig_lime)
        plt.close(fig_lime) # Close the figure to prevent display issues

with col2:
    st.subheader("Global Model Explanations")

    # Permutation Feature Importance
    st.markdown("#### Permutation Feature Importance")
    perm_importance_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = perm_importance_result.importances_mean.argsort()

    fig_perm, ax_perm = plt.subplots()
    ax_perm.barh(np.array(feature_names)[sorted_idx], perm_importance_result.importances_mean[sorted_idx])
    ax_perm.set_xlabel("Permutation Importance")
    ax_perm.set_title("Global Feature Importance")
    st.pyplot(fig_perm)
    plt.close(fig_perm)

    # Partial Dependence Plots
    st.markdown("#### Partial Dependence Plots")
    st.write("Visualize how each feature affects the prediction on average.")

    # Select a few features for PDPs
    pdp_features = ['age', 'cholesterol', 'blood_pressure']

    fig_pdp, ax_pdp = plt.subplots(ncols=len(pdp_features), figsize=(5 * len(pdp_features), 5))
    if len(pdp_features) == 1:
        ax_pdp = [ax_pdp] # Ensure ax_pdp is iterable even for a single plot

    for i, feature in enumerate(pdp_features):
        PartialDependenceDisplay.from_estimator(model, X_train, features=[feature], ax=ax_pdp[i], feature_names=feature_names)
        ax_pdp[i].set_title(f"PDP for {feature}")
    st.pyplot(fig_pdp)
    plt.close(fig_pdp)