import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        "age": np.random.randint(20, 80, n_samples),
        "bmi": np.random.uniform(18.0, 40.0, n_samples),
        "blood_pressure_sys": np.random.randint(90, 180, n_samples),
        "blood_pressure_dia": np.random.randint(60, 120, n_samples),
        "cholesterol": np.random.randint(150, 300, n_samples),
        "glucose": np.random.randint(70, 200, n_samples),
        "smoker": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "exercise_freq": np.random.randint(0, 7, n_samples),
    }
    df = pd.DataFrame(data)

    # Simulate a 'disease' outcome based on features
    df["disease"] = ((df["age"] > 55) * 0.3 + 
                     (df["bmi"] > 28) * 0.2 + 
                     (df["blood_pressure_sys"] > 140) * 0.25 + 
                     (df["cholesterol"] > 220) * 0.15 + 
                     (df["glucose"] > 120) * 0.1 + 
                     (df["smoker"] == 1) * 0.2 - 
                     (df["exercise_freq"] > 3) * 0.1).apply(lambda x: 1 if x > 0.5 + np.random.normal(0, 0.1) else 0)
    return df

@st.cache_resource
def train_model():
    df = generate_synthetic_data()
    X = df.drop("disease", axis=1)
    y = df["disease"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.sidebar.write(f"Model Accuracy (on synthetic data): {accuracy_score(y_test, y_pred):.2f}")
    return model, X.columns

def get_explanation(model, instance_df, feature_names):
    feature_importances = model.feature_importances_
    explanation = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
    return explanation.sort_values(by="Importance", ascending=False)


st.set_page_config(layout="wide")
st.title("MediExplain: Interactive Clinical Decision Support")

model, feature_names = train_model()

st.sidebar.header("Patient Data Input")

input_data = {}
input_data["age"] = st.sidebar.slider("Age", 20, 80, 50)
input_data["bmi"] = st.sidebar.slider("BMI", 18.0, 40.0, 25.0, 0.1)
input_data["blood_pressure_sys"] = st.sidebar.slider("Systolic Blood Pressure", 90, 180, 120)
input_data["blood_pressure_dia"] = st.sidebar.slider("Diastolic Blood Pressure", 60, 120, 80)
input_data["cholesterol"] = st.sidebar.slider("Cholesterol (mg/dL)", 150, 300, 200)
input_data["glucose"] = st.sidebar.slider("Glucose (mg/dL)", 70, 200, 100)
input_data["smoker"] = st.sidebar.selectbox("Smoker", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
input_data["exercise_freq"] = st.sidebar.slider("Exercise Frequency (days/week)", 0, 7, 3)

patient_df = pd.DataFrame([input_data])

st.subheader("Current Patient Profile")
st.write(patient_df)

if st.button("Get Prediction & Explanation"):
    prediction_proba = model.predict_proba(patient_df)[0]
    disease_risk = prediction_proba[1]
    
    st.subheader("Model Prediction")
    st.write(f"**Predicted Disease Risk:** {disease_risk:.2f} (Probability of having the disease)")
    
    st.subheader("Explanation (Feature Importance)")
    explanation_df = get_explanation(model, patient_df, feature_names)
    st.bar_chart(explanation_df.set_index("Feature"))

st.markdown("--- ")
st.info("This is a simplified demonstration. The model and explanations are based on synthetic data and a basic feature importance method.")