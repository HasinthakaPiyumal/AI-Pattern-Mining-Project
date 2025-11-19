import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(29, 77, n_samples),
        'sex': np.random.randint(0, 2, n_samples),  # 0: female, 1: male
        'cp': np.random.randint(0, 4, n_samples),  # chest pain type
        'trestbps': np.random.randint(94, 200, n_samples),  # resting blood pressure
        'chol': np.random.randint(126, 564, n_samples),  # serum cholestoral in mg/dl
        'fbs': np.random.randint(0, 2, n_samples),  # fasting blood sugar > 120 mg/dl
        'restecg': np.random.randint(0, 3, n_samples),  # resting electrocardiographic results
        'thalach': np.random.randint(71, 202, n_samples),  # maximum heart rate achieved
        'exang': np.random.randint(0, 2, n_samples),  # exercise induced angina
        'oldpeak': np.round(np.random.uniform(0.0, 6.2, n_samples), 1),  # ST depression induced by exercise relative to rest
        'slope': np.random.randint(0, 3, n_samples),  # the slope of the peak exercise ST segment
        'ca': np.random.randint(0, 4, n_samples),  # number of major vessels (0-3) colored by flourosopy
        'thal': np.random.randint(0, 3, n_samples),  # 0: normal; 1: fixed defect; 2: reversible defect
        'target': np.random.randint(0, 2, n_samples)  # 0: no disease, 1: disease
    }
    df = pd.DataFrame(data)
    
    # Introduce some correlation for target
    df['target'] = (df['age'] * 0.02 + df['sex'] * 0.5 + df['chol'] * 0.005 + df['oldpeak'] * 0.3 + np.random.randn(n_samples) * 0.8 > 3.5).astype(int)
    df.loc[df['age'] < 40, 'target'] = np.random.randint(0, 2, sum(df['age'] < 40), ) # less likely for younger
    df.loc[df['age'] > 60, 'target'] = np.random.randint(0, 2, sum(df['age'] > 60), ) # more likely for older
    df['target'] = (df['target'] + np.random.randint(0,2,n_samples)) % 2 # Shuffle a bit more

    return df


def train_and_save_model(df):
    X = df.drop('target', axis=1)
    y = df['target']

    numerical_features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_features)
        ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    model_pipeline.fit(X, y)
    joblib.dump(model_pipeline, 'cardiac_model_pipeline.pkl')
    st.success("Model trained and saved!")
    return model_pipeline, X.columns


def load_model():
    try:
        model_pipeline = joblib.load('cardiac_model_pipeline.pkl')
        return model_pipeline
    except FileNotFoundError:
        st.error("No trained model found. Please train the model first.")
        return None


st.set_page_config(layout="wide", page_title="Interpretable Medical Diagnostic Assistant")
st.title("🩺 Interpretable Medical Diagnostic Assistant for Cardiovascular Disease")
st.markdown("This application leverages an AI model to predict cardiovascular disease and provides comprehensive explanations for its predictions, both for individual patients and overall model behavior.")

# --- Sidebar for Model Management and Global Explanations ---
st.sidebar.header("Model Management & Global Insights")

df_full = generate_synthetic_data()

if st.sidebar.button("Train/Retrain Model"):
    model_pipeline, feature_names = train_and_save_model(df_full)
    st.session_state['model_pipeline'] = model_pipeline
    st.session_state['feature_names'] = feature_names

if 'model_pipeline' not in st.session_state:
    st.session_state['model_pipeline'] = load_model()
    if st.session_state['model_pipeline'] is not None:
        # Infer feature names from the preprocessor's fit data
        # This is a bit tricky with ColumnTransformer if not explicitly saved
        # For synthetic data, we can just use the original columns
        st.session_state['feature_names'] = df_full.drop('target', axis=1).columns.tolist()

model = st.session_state.get('model_pipeline')
feature_names = st.session_state.get('feature_names')

if model is None:
    st.warning("Please train the model first using the 'Train/Retrain Model' button in the sidebar.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(df_full.drop('target', axis=1), df_full['target'], test_size=0.2, random_state=42)

st.sidebar.subheader("Global Explanations")
if st.sidebar.checkbox("Show Permutation Feature Importance"):
    st.sidebar.markdown("Permutation Importance quantifies how much a model's performance decreases when a feature's values are randomly shuffled. Higher drop indicates higher importance.")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = perm_importance.importances_mean.argsort()
    
    fig_pi, ax_pi = plt.subplots(figsize=(10, 6))
    ax_pi.barh(np.array(feature_names)[sorted_idx], perm_importance.importances_mean[sorted_idx])
    ax_pi.set_xlabel("Permutation Importance")
    ax_pi.set_title("Global Permutation Feature Importance")
    st.sidebar.pyplot(fig_pi)

if st.sidebar.checkbox("Show Partial Dependence Plots (PDPs)"):
    st.sidebar.markdown("Partial Dependence Plots show the marginal effect of one or two features on the predicted outcome of a machine learning model.")
    
    # Select a few key features for PDPs
    pdp_features = ['age', 'chol', 'trestbps', 'oldpeak'] 
    
    fig_pdp, ax_pdp = plt.subplots(figsize=(12, 8))
    display = PartialDependenceDisplay.from_estimator(model, X_test, pdp_features, 
                                                      kind='average', ax=ax_pdp, 
                                                      feature_names=feature_names)
    fig_pdp.suptitle("Partial Dependence Plots")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    st.sidebar.pyplot(fig_pdp)

st.sidebar.subheader("Bias and Subgroup Analysis")
if st.sidebar.checkbox("Analyze Model Performance Across Subgroups"):
    st.sidebar.markdown("Evaluate how the model performs on different patient subgroups to identify potential biases or areas of divergent performance.")
    
    subgroups = {
        "All Patients": X_test,
        "Age < 50": X_test[X_test['age'] < 50],
        "Age >= 50": X_test[X_test['age'] >= 50],
        "Female": X_test[X_test['sex'] == 0],
        "Male": X_test[X_test['sex'] == 1]
    }

    metrics_data = []

    for name, subgroup_X in subgroups.items():
        if not subgroup_X.empty:
            subgroup_y_true = y_test.loc[subgroup_X.index]
            subgroup_y_pred = model.predict(subgroup_X)
            subgroup_y_proba = model.predict_proba(subgroup_X)[:, 1]
            
            accuracy = accuracy_score(subgroup_y_true, subgroup_y_pred)
            precision = precision_score(subgroup_y_true, subgroup_y_pred, zero_division=0)
            recall = recall_score(subgroup_y_true, subgroup_y_pred, zero_division=0)
            f1 = f1_score(subgroup_y_true, subgroup_y_pred, zero_division=0)
            try:
                roc_auc = roc_auc_score(subgroup_y_true, subgroup_y_proba)
            except ValueError:
                roc_auc = np.nan # AUC not defined for single class

            metrics_data.append({
                "Subgroup": name,
                "Count": len(subgroup_X),
                "Accuracy": f"{accuracy:.3f}",
                "Precision": f"{precision:.3f}",
                "Recall": f"{recall:.3f}",
                "F1-Score": f"{f1:.3f}",
                "ROC AUC": f"{roc_auc:.3f}"
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.sidebar.table(metrics_df)


# --- Main Content for Individual Patient Prediction and Local Explanations ---
st.header("Individual Patient Diagnosis and Explanation")
st.markdown("Enter patient details below to get a cardiovascular disease prediction and understand the factors contributing to it.")

input_data = {}
cols = st.columns(3) # Use columns for better layout

for i, feature in enumerate(feature_names):
    col = cols[i % 3]
    min_val = df_full[feature].min()
    max_val = df_full[feature].max()
    mean_val = df_full[feature].mean()

    if feature in ['sex', 'fbs', 'exang', 'cp', 'restecg', 'slope', 'ca', 'thal']:
        if feature == 'sex':
            input_data[feature] = col.selectbox(f"Patient {feature.capitalize()}", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        elif feature == 'cp':
            input_data[feature] = col.selectbox(f"Chest Pain Type", options=[0, 1, 2, 3])
        elif feature == 'fbs':
            input_data[feature] = col.selectbox(f"Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        elif feature == 'exang':
            input_data[feature] = col.selectbox(f"Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        elif feature == 'restecg':
            input_data[feature] = col.selectbox(f"Resting ECG", options=[0, 1, 2])
        elif feature == 'slope':
            input_data[feature] = col.selectbox(f"Slope of Peak Exercise ST Segment", options=[0, 1, 2])
        elif feature == 'ca':
            input_data[feature] = col.selectbox(f"Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
        elif feature == 'thal':
            input_data[feature] = col.selectbox(f"Thalassemia", options=[0, 1, 2], format_func=lambda x: {
                0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}.get(x, "Unknown"))
        else:
            input_data[feature] = col.number_input(f"Patient {feature.capitalize()}", min_value=int(min_val), max_value=int(max_val), value=int(mean_val))
    elif feature == 'oldpeak':
         input_data[feature] = col.number_input(f"Patient {feature.capitalize()} (ST Depression)", min_value=float(min_val), max_value=float(max_val), value=float(mean_val), step=0.1, format="%.1f")
    else:
        input_data[feature] = col.number_input(f"Patient {feature.capitalize()}", min_value=int(min_val), max_value=int(max_val), value=int(mean_val))


if st.button("Get Prediction and Explanation"):
    if model is not None:
        patient_df = pd.DataFrame([input_data])
        
        # Ensure the order of columns matches training
        patient_df = patient_df[feature_names]

        prediction = model.predict(patient_df)[0]
        prediction_proba = model.predict_proba(patient_df)[:, 1][0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"The model predicts a HIGH likelihood of Cardiovascular Disease (Probability: {prediction_proba:.2f})")
        else:
            st.success(f"The model predicts a LOW likelihood of Cardiovascular Disease (Probability: {prediction_proba:.2f})")

        st.subheader("Local Explanation (SHAP Values)")
        st.markdown("SHAP (SHapley Additive exPlanations) values show how much each feature contributes to the prediction for this specific patient.")

        # SHAP requires the preprocessed data to explain the classifier
        # We need to get the preprocessed version of the patient_df
        preprocessed_patient_df = model.named_steps['preprocessor'].transform(patient_df)
        preprocessed_df_full = model.named_steps['preprocessor'].transform(X_train) # Use training data for background

        # Access the RandomForestClassifier directly
        explainer = shap.TreeExplainer(model.named_steps['classifier'], data=preprocessed_df_full)
        shap_values = explainer.shap_values(preprocessed_patient_df)

        # If the model is a classifier, shap_values will be a list of arrays (one for each class)
        # For binary classification, we usually look at the SHAP values for the positive class (index 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # For the positive class (disease)

        # Get the feature names after preprocessing (scaler doesn't change names, but order matters)
        # This is a simplification; for complex preprocessors, getting exact post-preprocessing names is harder
        # We'll use the original feature_names for the plot for interpretability
        
        # Force plot (interactive but requires JS)
        st.write("#### SHAP Force Plot (Interactive)")
        # st_shap(shap.force_plot(explainer.expected_value[1], shap_values[0,:], patient_df.iloc[0,:]))
        # Using waterfall plot as it's easier to render statically in Streamlit
        
        st.write("#### SHAP Waterfall Plot")
        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap.Explanation(values=shap_values[0], 
                                              base_values=explainer.expected_value[1], 
                                              data=patient_df.iloc[0].values, 
                                              feature_names=feature_names.tolist()), 
                             show=False)
        st.pyplot(fig_waterfall)

        st.write("The waterfall plot shows how each feature value moves the output from the base value (average prediction) to the model's output for this patient. Features pushing the prediction towards \'disease\' are in red, and features pushing towards \'no disease\' are in blue.")

    else:
        st.warning("Model not trained. Please train the model first.")
