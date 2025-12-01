import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option("deprecation.showPyplotGlobalUse", False)

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "Age": np.random.randint(20, 90, num_samples),
        "Gender": np.random.choice(["Male", "Female"], num_samples),
        "Number of Comorbidities": np.random.randint(0, 5, num_samples),
        "Blood Pressure (Systolic)": np.random.randint(90, 180, num_samples),
        "Cholesterol Level": np.random.randint(120, 300, num_samples),
        "Readmission_Risk": np.random.randint(0, 2, num_samples) # 0: Low, 1: High
    }
    df = pd.DataFrame(data)

    # Introduce some correlation for a more realistic model
    df["Readmission_Risk"] = (
        (df["Age"] > 60).astype(int) * 0.4 +
        (df["Number of Comorbidities"] > 2).astype(int) * 0.3 +
        (df["Blood Pressure (Systolic)"] > 140).astype(int) * 0.2 +
        (df["Cholesterol Level"] > 200).astype(int) * 0.1 +
        np.random.rand(num_samples) > 0.7 # Add some randomness
    ).astype(int)

    df["Readmission_Risk"] = df["Readmission_Risk"].apply(lambda x: 1 if x > 0 else 0)

    return df

@st.cache_resource
def train_model(df):
    X = df.drop("Readmission_Risk", axis=1)
    y = df["Readmission_Risk"]

    categorical_features = ["Gender"]
    numerical_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ])

    X_processed = preprocessor.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_processed, y)

    return model, preprocessor, list(numerical_features) + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features))

# Streamlit App
st.title("🏥 Healthcare Predictive Analytics Dashboard")
st.write("Understand patient readmission risk predictions using Partial Dependence Plots.")

# Generate or load data
data = generate_synthetic_data()
st.subheader("Sample Patient Data")
st.dataframe(data.head())

# Train model
model, preprocessor, feature_names = train_model(data)

X_train_processed = preprocessor.transform(data.drop("Readmission_Risk", axis=1))
y_train = data["Readmission_Risk"]

# Display model performance (optional)
st.subheader("Model Performance (RandomForestClassifier)")
predictions = model.predict(X_train_processed)
accuracy = accuracy_score(y_train, predictions)
st.write(f"Training Accuracy: {accuracy:.2f}")

st.sidebar.header("Partial Dependence Plot Settings")

# Feature selection for PDP
all_features = [f for f in feature_names if f not in ["Readmission_Risk"]]
selected_pdp_features = st.sidebar.multiselect(
    "Select up to 2 features for PDP:",
    all_features,
    default=all_features[:2]
)

if len(selected_pdp_features) > 2:
    st.sidebar.warning("Please select at most 2 features for Partial Dependence Plots.")
    selected_pdp_features = selected_pdp_features[:2]

st.subheader("Partial Dependence Plots (PDPs)")

if selected_pdp_features:
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        display = PartialDependenceDisplay.from_estimator(
            estimator=model,
            X=X_train_processed,
            features=[feature_names.index(f) for f in selected_pdp_features],
            feature_names=feature_names,
            target=1, # For binary classification, typically plot probability of positive class
            kind="average",
            ax=ax
        )
        ax.set_title(f"Partial Dependence Plot for {', '.join(selected_pdp_features)}")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating PDP: {e}")
        st.info("Ensure selected features are appropriate for PDP generation.")
else:
    st.write("Please select one or two features from the sidebar to generate Partial Dependence Plots.")

st.subheader("Make a Prediction")
with st.form("prediction_form"):
    age = st.slider("Age", 20, 90, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    comorbidities = st.slider("Number of Comorbidities", 0, 5, 1)
    blood_pressure = st.slider("Blood Pressure (Systolic)", 90, 180, 120)
    cholesterol = st.slider("Cholesterol Level", 120, 300, 200)

    submit_button = st.form_submit_button("Predict Readmission Risk")

    if submit_button:
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Number of Comorbidities": comorbidities,
            "Blood Pressure (Systolic)": blood_pressure,
            "Cholesterol Level": cholesterol
        }])
        
        input_processed = preprocessor.transform(input_data)
        prediction_proba = model.predict_proba(input_processed)[:, 1]
        prediction_class = model.predict(input_processed)[0]

        st.write(f"\n\n#### Prediction for current patient:")
        st.write(f"Predicted Readmission Risk Probability: {prediction_proba[0]:.2f}")
        if prediction_class == 1:
            st.error("Predicted Readmission Risk: HIGH")
        else:
            st.success("Predicted Readmission Risk: LOW")

        st.write("\n\n")
