import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
import lime
import lime.lime_tabular

# --- 1. Data Generation (Synthetic) ---
@st.cache_data
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, num_samples),
        'BloodPressure': np.random.randint(90, 180, num_samples),
        'Cholesterol': np.random.randint(150, 300, num_samples),
        'BMI': np.random.uniform(18, 40, num_samples),
        'Glucose': np.random.randint(70, 200, num_samples),
        'Smoking': np.random.randint(0, 2, num_samples), # 0: No, 1: Yes
        'ExerciseHours': np.random.uniform(1, 10, num_samples),
        'FamilyHistory': np.random.randint(0, 2, num_samples), # 0: No, 1: Yes
    }
    df = pd.DataFrame(data)

    # Create a synthetic 'diagnosis' target variable
    # Simplified logic: higher BP, cholesterol, BMI, glucose, smoking, family history increase risk
    df['Diagnosis'] = ((df['BloodPressure'] > 140) * 0.3 + 
                       (df['Cholesterol'] > 240) * 0.2 + 
                       (df['BMI'] > 30) * 0.15 + 
                       (df['Glucose'] > 120) * 0.2 + 
                       df['Smoking'] * 0.1 + 
                       df['FamilyHistory'] * 0.05 + 
                       (10 - df['ExerciseHours']) * 0.05 + # Less exercise increases risk
                       np.random.rand(num_samples) * 0.2 > 0.7).astype(int)
    return df

# --- 2. Machine Learning Model Training ---
@st.cache_resource
def train_model(df):
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return model, X_train, X_test, y_test, accuracy, report, X.columns.tolist()

# --- Streamlit Application --- 
st.set_page_config(layout="wide", page_title="Explainable Medical Diagnosis AI")

st.title("🩺 Explainable Medical Diagnosis AI")
st.markdown("This application demonstrates local (LIME) and global (Permutation Importance) interpretability for a medical diagnosis AI model.")
st.markdown("---")

with st.sidebar:
    st.header("Configuration")
    num_samples = st.slider("Number of synthetic patient samples", 100, 5000, 1000, key="num_samples_slider")
    st.write("---")
    st.info("Generate data and train the model, then explore its explanations.")

# Section for Data Generation and Model Training
st.header("1. Data Generation & Model Training")
if st.button("Generate Data and Train Model", key="train_model_button"):
    with st.spinner("Generating data and training model..."):
        df = generate_synthetic_data(num_samples)
        st.session_state['df'] = df
        st.success("Synthetic patient data generated successfully!")
        st.dataframe(df.head())

        model, X_train, X_test, y_test, accuracy, report, feature_names = train_model(df)
        st.session_state['model'] = model
        st.session_state['X_train'] = X_train
        st.session_state['feature_names'] = feature_names

        st.subheader("Model Performance on Test Set")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{accuracy:.2f}")
        with col2:
            st.json(report)
        st.success("Random Forest Classifier trained successfully!")
else:
    if 'model' not in st.session_state:
        st.warning("Click 'Generate Data and Train Model' to start the application.")

if 'model' in st.session_state:
    model = st.session_state['model']
    X_train = st.session_state['X_train']
    feature_names = st.session_state['feature_names']

    st.markdown("---")

    # Section for Global Feature Importance (Permutation Importance)
    st.header("2. Global Feature Importance (Permutation Importance)")
    st.write("This shows the overall importance of each feature to the model's predictions. High importance indicates that shuffling the feature significantly increases the model's error.")
    
    if st.button("Calculate Global Feature Importance", key="calculate_global_importance_button"):
        with st.spinner("Calculating permutation importance..."):
            # Using sklearn's permutation_importance
            perm_importance_result = permutation_importance(
                model, X_train, model.predict(X_train), n_repeats=10, random_state=42, n_jobs=-1
            )
            sorted_idx = perm_importance_result.importances_mean.argsort()[::-1]
            
            perm_df = pd.DataFrame({
                "Feature": [feature_names[i] for i in sorted_idx],
                "Importance Mean (Decrease in Accuracy)": perm_importance_result.importances_mean[sorted_idx],
                "Importance Std": perm_importance_result.importances_std[sorted_idx],
            })
            st.dataframe(perm_df)
            st.bar_chart(perm_df.set_index("Feature")["Importance Mean (Decrease in Accuracy)"])

    st.markdown("---")

    # Section for Local Interpretability (LIME)
    st.header("3. Local Interpretability (LIME)")
    st.write("Understand how individual features contribute to a specific patient's diagnosis prediction.")

    st.subheader("Enter New Patient Data for Explanation:")
    input_data = {}
    col_inputs = st.columns(len(feature_names) // 2 + len(feature_names) % 2)
    col_idx = 0
    for feature in feature_names:
        with col_inputs[col_idx % len(col_inputs)]:
            if feature in ['Smoking', 'FamilyHistory']:
                input_data[feature] = st.selectbox(f"{feature}:", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key=f"input_{feature}")
            elif feature == 'Age':
                input_data[feature] = st.slider(f"{feature}:", 20, 80, 50, key=f"input_{feature}")
            elif feature == 'BloodPressure':
                input_data[feature] = st.slider(f"{feature}:", 90, 180, 120, key=f"input_{feature}")
            elif feature == 'Cholesterol':
                input_data[feature] = st.slider(f"{feature}:", 150, 300, 200, key=f"input_{feature}")
            elif feature == 'BMI':
                input_data[feature] = st.slider(f"{feature}:", 18.0, 40.0, 25.0, step=0.1, key=f"input_{feature}")
            elif feature == 'Glucose':
                input_data[feature] = st.slider(f"{feature}:", 70, 200, 100, key=f"input_{feature}")
            elif feature == 'ExerciseHours':
                input_data[feature] = st.slider(f"{feature}:", 1.0, 10.0, 5.0, step=0.1, key=f"input_{feature}")
        col_idx += 1

    new_patient_df = pd.DataFrame([input_data])
    
    if st.button("Get Local Explanation for Patient", key="get_local_explanation_button"):
        with st.spinner("Generating LIME explanation..."):
            prediction = model.predict(new_patient_df)[0]
            prediction_proba = model.predict_proba(new_patient_df)[0]

            st.subheader(f"Model Prediction for this Patient: {'Diagnosis Present' if prediction == 1 else 'No Diagnosis'}")
            st.write(f"Probability of No Diagnosis: {prediction_proba[0]:.2f}")
            st.write(f"Probability of Diagnosis Present: {prediction_proba[1]:.2f}")

            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=feature_names,
                class_names=['No Diagnosis', 'Diagnosis Present'],
                mode='classification'
            )

            explanation = explainer.explain_instance(
                data_row=new_patient_df.iloc[0].values,
                predict_fn=model.predict_proba,
                num_features=len(feature_names)
            )

            st.write("#### LIME Explanation (How features contributed to this specific prediction):")
            st.info("Green bars indicate features pushing towards the predicted class, red bars push against it.")
            
            # LIME explanation as list and dataframe for easier viewing
            exp_list = explanation.as_list()
            exp_df = pd.DataFrame(exp_list, columns=["Feature Condition", "Weight (Contribution)"])
            st.dataframe(exp_df)
            
            # Displaying LIME explanation as HTML for full visualization
            st.components.v1.html(explanation.as_html(), height=500, scrolling=True)


st.markdown("---")
st.caption("Developed as part of an AI Interpretability & Debugging Framework demonstration.")