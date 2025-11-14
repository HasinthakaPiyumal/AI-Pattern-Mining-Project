import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Generate Synthetic Medical Data ---
@st.cache_data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, n_samples),
        'Symptom_A': np.random.randint(0, 2, n_samples),  # 0 or 1
        'Symptom_B': np.random.randint(0, 2, n_samples),
        'Test_Result_1': np.random.rand(n_samples) * 100, # Continuous
        'Test_Result_2': np.random.rand(n_samples) * 50,  # Continuous
    }
    df = pd.DataFrame(data)

    # Create a 'Diagnosis' target based on some rules (simulating a complex medical logic)
    def get_diagnosis(row):
        if row['Age'] > 60 and row['Symptom_A'] == 1 and row['Test_Result_1'] > 70:
            return 'Diagnosis_C'
        elif row['Age'] < 40 and row['Symptom_B'] == 1 and row['Test_Result_2'] < 20:
            return 'Diagnosis_A'
        elif row['Symptom_A'] == 1 and row['Test_Result_1'] > 50:
            return 'Diagnosis_B'
        else:
            return 'Diagnosis_D'

    df['Diagnosis'] = df.apply(get_diagnosis, axis=1)
    return df

# --- 2. Train Black-box Model ---
@st.cache_resource
def train_model(df):
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, X_train, X_test, y_test, accuracy, X.columns.tolist(), model.classes_.tolist()

# --- Streamlit Application ---
st.set_page_config(layout="wide", page_title="AI Medical Diagnosis Interpretability")
st.title("🩺 AI Medical Diagnosis Interpretability & Debugging Framework")
st.markdown("This application demonstrates how to interpret a black-box AI model for medical diagnosis using local and global interpretability techniques.")

# Generate data and train model
df = generate_synthetic_data()
model, X_train, X_test, y_test, accuracy, feature_names, class_names = train_model(df)

st.sidebar.header("Model Information")
st.sidebar.write(f"**Model Type:** RandomForestClassifier")
st.sidebar.write(f"**Accuracy on Test Set:** {accuracy:.2f}")
st.sidebar.write(f"**Features:** {', '.join(feature_names)}")
st.sidebar.write(f"**Diagnoses (Classes):** {', '.join(class_names)}")


# --- Tabs for different interpretability views ---
tabs = st.tabs(["Model Overview", "Local Explanations (LIME)", "Global Explanations (PDP & PFI)"])

with tabs[0]: # Model Overview
    st.header("Model Performance and Data Sample")
    st.subheader("Synthetic Medical Data Sample")
    st.dataframe(df.head())

    st.subheader("Diagnosis Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Diagnosis', data=df, ax=ax)
    ax.set_title('Distribution of Diagnoses in Data')
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Count')
    st.pyplot(fig)

with tabs[1]: # Local Explanations (LIME)
    st.header("Local Explanations (LIME)")
    st.markdown("LIME helps to understand individual predictions by explaining what features contributed most to a specific outcome for a single patient.")

    st.subheader("Select a Patient for Explanation")
    patient_index = st.slider("Select a patient from the test set (index)", 0, len(X_test) - 1, 0)

    instance_to_explain = X_test.iloc[patient_index]
    true_label = y_test.iloc[patient_index]
    model_prediction = model.predict(instance_to_explain.to_frame().T)[0]

    st.write(f"**Selected Patient (Index: {patient_index}):**")
    st.dataframe(instance_to_explain.to_frame().T)
    st.write(f"**True Diagnosis:** {true_label}")
    st.write(f"**Model Predicted Diagnosis:** {model_prediction}")

    st.subheader("LIME Explanation for Selected Patient")

    # LIME explainer setup
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

    # Generate explanation
    with st.spinner("Generating LIME explanation..."):
        explanation = explainer.explain_instance(
            data_row=instance_to_explain.values,
            predict_fn=model.predict_proba,
            num_features=5
        )

    fig_lime = explanation.as_pyplot_figure()
    st.pyplot(fig_lime)
    st.write("The plot above shows the features contributing most positively or negatively to the model's prediction for the selected patient.")

with tabs[2]: # Global Explanations (PDP & PFI)
    st.header("Global Explanations (Partial Dependence & Permutation Feature Importance)")
    st.markdown("Global explanations provide insights into the overall behavior of the model across the entire dataset.")

    st.subheader("Permutation Feature Importance (PFI)")
    st.markdown("PFI measures how much the model's accuracy decreases when a single feature's values are randomly shuffled, indicating its importance.")

    with st.spinner("Calculating Permutation Feature Importance..."):
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()

    fig_pfi, ax_pfi = plt.subplots(figsize=(10, 6))
    ax_pfi.boxplot(result.importances[sorted_idx].T,
                   vert=False, labels=np.array(feature_names)[sorted_idx])
    ax_pfi.set_title("Permutation Feature Importance")
    ax_pfi.set_xlabel("Decrease in Accuracy (Mean +/- Std Dev)")
    st.pyplot(fig_pfi)

    st.subheader("Partial Dependence Plots (PDPs)")
    st.markdown("PDPs show the marginal effect of one or two features on the predicted outcome of a machine learning model, averaging over the values of all other features.")

    # Select features for PDPs
    pdp_features = st.multiselect(
        "Select up to 2 features for Partial Dependence Plots",
        options=feature_names,
        default=['Age', 'Test_Result_1']
    )

    if len(pdp_features) > 2:
        st.warning("Please select at most 2 features for PDPs.")
    elif len(pdp_features) > 0:
        st.write(f"Generating PDP for: {', '.join(pdp_features)}")
        fig_pdp, ax_pdp = plt.subplots(figsize=(10, 6))
        with st.spinner("Generating Partial Dependence Plot..."):
            display = PartialDependenceDisplay.from_estimator(
                model, X_train, pdp_features, ax=ax_pdp, kind='average', grid_resolution=50
            )
            display.figure_.suptitle(f"Partial Dependence Plot for {', '.join(pdp_features)}")
            st.pyplot(fig_pdp)
    else:
        st.info("Select features to view Partial Dependence Plots.")
