import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

import shap
import dice_ml

# Set page config for Streamlit
st.set_page_config(layout="wide", page_title="Interpretable AI Clinical Decision Support")

# --- 1. Data Layer: Synthetic Data Generation ---
@st.cache_data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        "Age": np.random.randint(20, 80, n_samples),
        "Cholesterol": np.random.randint(150, 250, n_samples),
        "BloodPressure": np.random.randint(90, 180, n_samples),
        "BMI": np.random.uniform(18.0, 35.0, n_samples),
        "Glucose": np.random.randint(70, 200, n_samples),
        "Smoking": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]), # 0: No, 1: Yes
        "FamilyHistory": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]), # 0: No, 1: Yes
        "Gender": np.random.choice(["Male", "Female"], n_samples, p=[0.5, 0.5])
    }
    df = pd.DataFrame(data)

    # Simulate disease based on features (simplified for demonstration)
    df["Disease"] = 0
    df.loc[(df["Age"] > 50) & (df["Cholesterol"] > 200) & (df["BloodPressure"] > 140), "Disease"] = 1
    df.loc[(df["BMI"] > 30) & (df["Glucose"] > 120), "Disease"] = 1
    df.loc[(df["Smoking"] == 1) | (df["FamilyHistory"] == 1), "Disease"] = 1
    df.loc[df.sample(frac=0.1).index, "Disease"] = 1 # Add some random noise

    # Ensure at least some cases of both disease/no disease
    df["Disease"] = df["Disease"].astype(int)
    
    return df

# --- 2. Prediction Model Layer: Model Training ---
@st.cache_resource
def train_model(X_train, y_train, random_state=42):
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# --- Streamlit App Layout ---
st.title("🩺 Interpretable AI-Powered Clinical Decision Support System")
st.markdown("This framework provides transparent insights into a disease prediction model's decisions, fostering trust and enabling responsible deployment.")

# Sidebar for navigation and settings
st.sidebar.header("Configuration")
page_selection = st.sidebar.radio("Go to", ["Dashboard", "Local Interpretability", "Global Interpretability", "Bias Detection"])

# --- Data Generation and Model Training --- 
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train_encoded' not in st.session_state:
    st.session_state.X_train_encoded = None
if 'X_test_encoded' not in st.session_state:
    st.session_state.X_test_encoded = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

with st.sidebar.expander("Data & Model Setup", expanded=True):
    st.markdown("### Data Generation")
    num_samples = st.slider("Number of synthetic patient records", 100, 5000, 1000, step=100)
    if st.button("Generate Synthetic Data & Train Model"):
        st.session_state.data = generate_synthetic_data(num_samples)
        
        df_encoded = pd.get_dummies(st.session_state.data, columns=["Gender"], drop_first=True) # One-hot encode Gender
        X = df_encoded.drop("Disease", axis=1)
        y = df_encoded["Disease"]
        
        st.session_state.feature_names = X.columns.tolist()

        st.session_state.X_train_encoded, st.session_state.X_test_encoded, y_train, st.session_state.y_test = \
            train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        st.session_state.model = train_model(st.session_state.X_train_encoded, y_train)

        st.success("Synthetic data generated and model trained successfully!")


if st.session_state.data is None or st.session_state.model is None:
    st.warning("Please generate synthetic data and train the model in the sidebar to proceed.")
    st.stop()

# --- Dashboard Page ---
if page_selection == "Dashboard":
    st.header("📊 Model Performance Dashboard")
    st.write("Overview of the trained model's performance on the test set.")

    if st.session_state.model and st.session_state.X_test_encoded is not None:
        y_pred = st.session_state.model.predict(st.session_state.X_test_encoded)
        y_proba = st.session_state.model.predict_proba(st.session_state.X_test_encoded)[:, 1]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(st.session_state.y_test, y_pred):.2f}")
        col2.metric("Precision", f"{precision_score(st.session_state.y_test, y_pred):.2f}")
        col3.metric("Recall", f"{recall_score(st.session_state.y_test, y_pred):.2f}")
        col4.metric("F1 Score", f"{f1_score(st.session_state.y_test, y_pred):.2f}")
        st.metric("ROC-AUC Score", f"{roc_auc_score(st.session_state.y_test, y_proba):.2f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(st.session_state.y_test, y_pred))
    else:
        st.error("Model not trained or test data not available. Please generate data and train the model.")

# --- Local Interpretability Page ---
elif page_selection == "Local Interpretability":
    st.header("🔬 Local Interpretability: Instance-Specific Insights")
    st.write("Understand why the model made a specific prediction for an individual patient.")

    if st.session_state.model and st.session_state.X_test_encoded is not None:
        patient_idx = st.slider(
            "Select a patient from the test set (index)",
            0, len(st.session_state.X_test_encoded) - 1, 0
        )
        patient_instance = st.session_state.X_test_encoded.iloc[[patient_idx]]
        true_label = st.session_state.y_test.iloc[patient_idx]
        predicted_proba = st.session_state.model.predict_proba(patient_instance)[:, 1][0]
        predicted_label = st.session_state.model.predict(patient_instance)[0]

        st.markdown(f"### Patient Details (Index: {patient_idx})")
        st.write(patient_instance)
        st.info(f"**True Disease Status:** {true_label} | **Predicted Disease Probability:** {predicted_proba:.2f} | **Predicted Disease Status:** {predicted_label}")

        st.subheader("1. SHAP (SHapley Additive exPlanations)")
        st.write("SHAP values show how much each feature contributes to the prediction for this patient.")
        
        # Explainer for tree-based models
        explainer = shap.TreeExplainer(st.session_state.model)
        shap_values = explainer.shap_values(patient_instance)

        # Force plot (requires JS, so using waterfall as an alternative for static display)
        st.set_option('deprecation.showPyplotGlobalUse', False) # Suppress warning
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(values=shap_values[1][0], 
                                           base_values=explainer.expected_value[1], 
                                           data=patient_instance.values[0], 
                                           feature_names=st.session_state.feature_names),
                            show=False)
        st.pyplot(fig)


        st.subheader("2. Counterfactual Explanations (DiCE)")
        st.write("What minimal changes to the patient's features would change the prediction?")

        # Initialize DiCE explainer
        d = dice_ml.Data(dataframe=st.session_state.data.drop('Disease', axis=1), 
                         continuous_features=st.session_state.feature_names[:-1], # All except Gender_Male
                         outcome_name='Disease')
        m = dice_ml.Model(model=st.session_state.model, backend="sklearn")
        exp = dice_ml.Dice(d, m)

        # Generate counterfactuals (target_binary=0 for healthy, 1 for diseased)
        target_cf_label = 1 - predicted_label # Try to flip the prediction
        
        try:
            dice_exp = exp.generate_counterfactuals(
                patient_instance, 
                total_CFs=3, 
                desired_class=target_cf_label
            )
            st.write(f"Counterfactuals to change prediction to disease status: {target_cf_label}")
            st.dataframe(dice_exp.cf_examples_list[0].final_cfs_df)
            st.markdown("**Explanation:** The table above shows hypothetical scenarios (counterfactuals) where the patient's features are slightly altered to change the model's prediction. For example, if a patient's 'Cholesterol' was X instead of Y, their predicted disease status might change.")
        except Exception as e:
            st.warning(f"Could not generate counterfactuals: {e}. This often happens if no counterfactuals can be found with the given constraints.")
    else:
        st.error("Model not trained or test data not available. Please generate data and train the model.")

# --- Global Interpretability Page ---
elif page_selection == "Global Interpretability":
    st.header("🌍 Global Interpretability: Overall Model Understanding")
    st.write("Understand the overall behavior and key drivers of the model across the entire dataset.")

    if st.session_state.model and st.session_state.X_train_encoded is not None:
        st.subheader("1. Partial Dependence Plots (PDP)")
        st.write("How does a feature (or two) affect the predicted outcome on average?")
        
        features_to_plot = st.multiselect(
            "Select features for PDP (max 2)", 
            st.session_state.feature_names, 
            default=st.session_state.feature_names[:2]
        )
        if features_to_plot:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                PartialDependenceDisplay.from_estimator(
                    st.session_state.model, 
                    st.session_state.X_train_encoded, 
                    features=features_to_plot, 
                    target=1, # Probability of disease
                    feature_names=st.session_state.feature_names, 
                    ax=ax, 
                    grid_resolution=20
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating PDP: {e}")

        st.subheader("2. Permutation Feature Importance (PFI)")
        st.write("Which features are most important for the model's overall predictive performance?")
        
        if st.session_state.X_test_encoded is not None and st.session_state.y_test is not None:
            result = permutation_importance(
                st.session_state.model, 
                st.session_state.X_test_encoded, 
                st.session_state.y_test, 
                n_repeats=10, 
                random_state=42, 
                n_jobs=-1
            )
            sorted_idx = result.importances_mean.argsort()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(
                result.importances[sorted_idx].T,
                vert=False,
                labels=np.array(st.session_state.feature_names)[sorted_idx],
            )
            ax.set_title("Permutation Feature Importance")
            ax.set_xlabel("Mean decrease in accuracy")
            st.pyplot(fig)
        else:
            st.warning("Test data not available for Permutation Feature Importance.")
    else:
        st.error("Model not trained or training data not available. Please generate data and train the model.")

# --- Bias Detection Page ---
elif page_selection == "Bias Detection":
    st.header("🔍 Bias Detection & Subgroup Analysis")
    st.write("Identify potential biases by comparing model performance across different demographic subgroups.")

    if st.session_state.model and st.session_state.X_test_encoded is not None and st.session_state.y_test is not None:
        # Reconstruct full test set with original 'Gender' for filtering
        test_data_original_gender = st.session_state.data.loc[st.session_state.X_test_encoded.index].copy()
        test_data_original_gender['Disease_Predicted'] = st.session_state.model.predict(st.session_state.X_test_encoded)
        test_data_original_gender['Disease_True'] = st.session_state.y_test

        demographic_feature = st.selectbox("Select demographic feature for subgroup analysis", ["Gender"])

        if demographic_feature:
            st.subheader(f"Performance by {demographic_feature}")
            groups = test_data_original_gender[demographic_feature].unique()
            
            metrics_df = pd.DataFrame(columns=["Group", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])

            for group in groups:
                subset_data = test_data_original_gender[test_data_original_gender[demographic_feature] == group]
                if not subset_data.empty and len(subset_data['Disease_True'].unique()) > 1: # Ensure at least two classes in subset
                    y_true_subset = subset_data["Disease_True"]
                    y_pred_subset = subset_data["Disease_Predicted"]
                    # Need original encoded X for predict_proba
                    X_subset_encoded = st.session_state.X_test_encoded.loc[subset_data.index]
                    y_proba_subset = st.session_state.model.predict_proba(X_subset_encoded)[:, 1]

                    acc = accuracy_score(y_true_subset, y_pred_subset)
                    prec = precision_score(y_true_subset, y_pred_subset, zero_division=0)
                    rec = recall_score(y_true_subset, y_pred_subset, zero_division=0)
                    f1 = f1_score(y_true_subset, y_pred_subset, zero_division=0)
                    roc_auc = roc_auc_score(y_true_subset, y_proba_subset)

                    metrics_df = pd.concat([
                        metrics_df, 
                        pd.DataFrame([{
                            "Group": group, 
                            "Accuracy": f"{acc:.2f}", 
                            "Precision": f"{prec:.2f}", 
                            "Recall": f"{rec:.2f}", 
                            "F1-Score": f"{f1:.2f}", 
                            "ROC-AUC": f"{roc_auc:.2f}"
                        }])
                    ], ignore_index=True)
                else:
                    st.warning(f"Not enough data or unique classes for subgroup '{group}' to calculate all metrics.")
            
            if not metrics_df.empty:
                st.dataframe(metrics_df.set_index("Group"))
            else:
                st.info("No sufficient subgroups found to display performance metrics.")

            st.markdown("**Interpretation:** Compare metrics across groups. Significant differences might indicate bias.")

            st.subheader("Feedback Mechanism (Conceptual)")
            st.info("In a real-world system, this section would allow clinicians to provide feedback on specific predictions or subgroup biases, which could then be used for model retraining or data collection efforts.")
            feedback_text = st.text_area("Enter your feedback here:", "e.g., 'Model performs poorly for older female patients.'")
            if st.button("Submit Feedback"):
                st.success("Feedback submitted! (This is a conceptual placeholder)")

    else:
        st.error("Model not trained or test data not available. Please generate data and train the model.")
