import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# --- 1. Data Generation & Dummy Model Training ---

def generate_synthetic_data(num_samples=100):
    np.random.seed(42)
    data = {
        "age": np.random.randint(20, 80, num_samples),
        "blood_pressure": np.random.randint(90, 180, num_samples),
        "cholesterol": np.random.randint(150, 300, num_samples),
        "hdl": np.random.randint(30, 80, num_samples),
        "bmi": np.random.uniform(18.0, 35.0, num_samples),
        "smoking": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        "exercise_freq": np.random.randint(0, 7, num_samples),
    }
    df = pd.DataFrame(data)

    # Create a dummy 'diagnosis' based on some rules
    df["diagnosis"] = "Healthy"
    df.loc[(df["blood_pressure"] > 140) | (df["cholesterol"] > 240) | (df["age"] > 60) | (df["smoking"] == 1), "diagnosis"] = "High Risk"
    df.loc[(df["blood_pressure"] > 160) & (df["cholesterol"] > 260) & (df["age"] > 70), "diagnosis"] = "Critical Condition"
    df.loc[(df["bmi"] > 30) & (df["blood_pressure"] > 130), "diagnosis"] = "Obesity Related"

    return df

def train_dummy_model(df):
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.sidebar.write(f"Dummy Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    return model, X.columns.tolist()

# --- 2. Backend Services ---

class DataHandler:
    def __init__(self, df):
        self.patient_data = df

    def get_patient_by_id(self, patient_id):
        if patient_id in self.patient_data.index:
            return self.patient_data.loc[patient_id]
        return None

    def preprocess_features(self, patient_features):
        # In a real scenario, this would handle scaling, encoding, etc.
        # For this demo, assuming features are ready for the model.
        return patient_features.to_frame().T # Ensure it's a 2D array-like input

class ModelInferenceService:
    def __init__(self, model):
        self.model = model

    def predict(self, features):
        prediction = self.model.predict(features)
        return prediction[0]

    def predict_proba(self, features):
        probabilities = self.model.predict_proba(features)
        return dict(zip(self.model.classes_, probabilities[0]))

class ExplanationGenerationService:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def generate_lace_explanation(self, patient_data_series, prediction):
        explanation = {
            "feature_importance": {},
            "local_rules": [],
            "counterfactuals": []
        }

        # Simplified Feature Importance (dummy implementation)
        # In a real LACE, this would come from the explanation algorithm
        feature_values = patient_data_series.to_dict()
        for feature in self.feature_names:
            # Assigning arbitrary importance based on deviation or specific values
            importance = 0.1 # default low importance
            if feature == "blood_pressure" and feature_values[feature] > 140:
                importance = 0.8
            elif feature == "cholesterol" and feature_values[feature] > 240:
                importance = 0.7
            elif feature == "age" and feature_values[feature] > 60:
                importance = 0.6
            elif feature == "smoking" and feature_values[feature] == 1:
                importance = 0.9
            elif feature == "bmi" and feature_values[feature] > 30:
                importance = 0.5
            explanation["feature_importance"][feature] = importance
        
        # Sort feature importance for better display
        explanation["feature_importance"] = dict(sorted(explanation["feature_importance"].items(), key=lambda item: item[1], reverse=True))

        # Simplified Local Rules (dummy implementation)
        if prediction == "High Risk":
            if patient_data_series["blood_pressure"] > 140:
                explanation["local_rules"].append(f"High blood pressure ({patient_data_series['blood_pressure']}) contributes significantly.")
            if patient_data_series["cholesterol"] > 240:
                explanation["local_rules"].append(f"Elevated cholesterol ({patient_data_series['cholesterol']}) is a key factor.")
            if patient_data_series["age"] > 60:
                explanation["local_rules"].append(f"Age ({patient_data_series['age']}) is a contributing risk factor.")
            if patient_data_series["smoking"] == 1:
                explanation["local_rules"].append("Patient is a smoker, increasing risk.")
        elif prediction == "Critical Condition":
            explanation["local_rules"].append("Multiple severe risk factors detected, indicating critical status.")
            if patient_data_series["blood_pressure"] > 160:
                explanation["local_rules"].append(f"Extremely high blood pressure ({patient_data_series['blood_pressure']}) is critical.")
        elif prediction == "Obesity Related":
            if patient_data_series["bmi"] > 30:
                explanation["local_rules"].append(f"High BMI ({patient_data_series['bmi']:.1f}) is a primary concern.")

        # Simplified Counterfactuals (dummy implementation)
        # Suggest changes that *might* flip the prediction to 'Healthy' or a less severe state
        original_features_df = patient_data_series.to_frame().T
        original_prediction = self.model.predict(original_features_df)[0]

        if original_prediction != "Healthy":
            # Attempt to find a simple counterfactual
            temp_features = patient_data_series.copy()
            if temp_features["blood_pressure"] > 120:
                temp_features["blood_pressure"] = 115 # Lower BP
                if self.model.predict(temp_features.to_frame().T)[0] == "Healthy":
                    explanation["counterfactuals"].append(
                        f"If blood pressure was {115} instead of {patient_data_series['blood_pressure']}, diagnosis might be 'Healthy'."
                    )
                    temp_features["blood_pressure"] = patient_data_series["blood_pressure"] # Reset

            if temp_features["cholesterol"] > 200:
                temp_features["cholesterol"] = 190 # Lower Cholesterol
                if self.model.predict(temp_features.to_frame().T)[0] == "Healthy":
                    explanation["counterfactuals"].append(
                        f"If cholesterol was {190} instead of {patient_data_series['cholesterol']}, diagnosis might be 'Healthy'."
                    )
                    temp_features["cholesterol"] = patient_data_series["cholesterol"] # Reset
            
            if temp_features["smoking"] == 1:
                temp_features["smoking"] = 0 # Stop smoking
                if self.model.predict(temp_features.to_frame().T)[0] == "Healthy":
                    explanation["counterfactuals"].append(
                        "If patient stopped smoking, diagnosis might be 'Healthy'."
                    )
                    temp_features["smoking"] = patient_data_series["smoking"] # Reset

        return explanation

class RuleEngine:
    def evaluate_user_rule(self, explanation, user_rule_text):
        feedback = []
        user_rule_text_lower = user_rule_text.lower()

        if not user_rule_text_lower:
            return []

        # Simple keyword matching for demonstration
        if "blood pressure" in user_rule_text_lower:
            if any("blood pressure" in rule.lower() for rule in explanation["local_rules"]):
                feedback.append("User rule regarding blood pressure aligns with explanation.")
            else:
                feedback.append("User rule regarding blood pressure does not explicitly appear in local rules.")
        
        if "cholesterol" in user_rule_text_lower:
            if any("cholesterol" in rule.lower() for rule in explanation["local_rules"]):
                feedback.append("User rule regarding cholesterol aligns with explanation.")
            else:
                feedback.append("User rule regarding cholesterol does not explicitly appear in local rules.")
        
        if "smoking" in user_rule_text_lower:
            if any("smoker" in rule.lower() for rule in explanation["local_rules"]):
                feedback.append("User rule regarding smoking aligns with explanation.")
            else:
                feedback.append("User rule regarding smoking does not explicitly appear in local rules.")

        if not feedback:
            feedback.append("No direct alignment found for the user rule with the generated explanation. This doesn't mean it's incorrect, just not explicitly covered.")
        
        return feedback

    def aggregate_explanations(self, all_explanations):
        # This is a very simplified aggregation for global insights
        # In a real system, this would involve sophisticated analysis over many explanations
        common_high_importance_features = {}
        for exp in all_explanations:
            for feature, importance in exp["feature_importance"].items():
                if importance > 0.5: # Arbitrary threshold for 'high' importance
                    common_high_importance_features[feature] = common_high_importance_features.get(feature, 0) + 1
        
        if common_high_importance_features:
            sorted_features = sorted(common_high_importance_features.items(), key=lambda item: item[1], reverse=True)
            return f"Top recurring important features across cases: {', '.join([f'{f} ({count} cases)' for f, count in sorted_features[:3]])}"
        return "No significant global patterns identified from aggregated explanations (yet)."


# --- Streamlit UI --- 
st.set_page_config(layout="wide")
st.title("MediExplain: Interactive Diagnostic Assistant Explanation Platform 🩺")

# --- Data & Model Initialization (cached) ---
@st.cache_resource
def load_data_and_model():
    df = generate_synthetic_data(num_samples=200)
    model, feature_names = train_dummy_model(df.copy())
    data_handler = DataHandler(df)
    model_inference_service = ModelInferenceService(model)
    explanation_service = ExplanationGenerationService(model, feature_names)
    rule_engine = RuleEngine()
    return data_handler, model_inference_service, explanation_service, rule_engine, df.index.tolist(), feature_names

data_handler, model_inference_service, explanation_service, rule_engine, patient_ids, feature_names = load_data_and_model()

# Store explanations for global insights
if "all_explanations" not in st.session_state:
    st.session_state.all_explanations = []

# Sidebar for Global Insights
st.sidebar.header("Global Explanation Insights")
st.sidebar.info(rule_engine.aggregate_explanations(st.session_state.all_explanations))

# Main content layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Patient Information & Diagnosis")
    selected_patient_id = st.selectbox("Select Patient ID", patient_ids)

    patient_data = data_handler.get_patient_by_id(selected_patient_id)
    if patient_data is not None:
        st.subheader("Selected Patient Attributes")
        st.dataframe(patient_data.to_frame().T)

        st.subheader("AI Diagnostic Prediction")
        features_for_prediction = data_handler.preprocess_features(patient_data)
        original_prediction = model_inference_service.predict(features_for_prediction)
        st.success(f"Predicted Diagnosis: **{original_prediction}**")

        st.subheader("Prediction Probabilities")
        probabilities = model_inference_service.predict_proba(features_for_prediction)
        st.json(probabilities)

        # Generate and store explanation
        current_explanation = explanation_service.generate_lace_explanation(patient_data, original_prediction)
        if current_explanation not in st.session_state.all_explanations:
            st.session_state.all_explanations.append(current_explanation)

        st.subheader("Explanation for Diagnosis (LACE-like)")
        
        st.write("#### Feature Importance")
        feature_importance_df = pd.DataFrame(current_explanation["feature_importance"].items(), columns=["Feature", "Importance"])
        st.bar_chart(feature_importance_df.set_index("Feature"))

        st.write("#### Local Rules")
        if current_explanation["local_rules"]:
            for rule in current_explanation["local_rules"]:
                st.write(f"- {rule}")
        else:
            st.info("No specific local rules identified for this prediction.")

        st.write("#### Counterfactual Examples")
        if current_explanation["counterfactuals"]:
            for cf in current_explanation["counterfactuals"]:
                st.info(f"- {cf}")
        else:
            st.info("No simple counterfactuals found to change diagnosis to 'Healthy' significantly with minor tweaks.")

with col2:
    st.header("Interactive Exploration & Validation")

    st.subheader("'What-if' Analysis")
    st.write("Adjust patient attributes to see how the diagnosis and explanation change.")

    modified_patient_data = patient_data.copy()
    for feature in feature_names:
        if patient_data[feature].dtype == "int64":
            modified_patient_data[feature] = st.slider(
                f"Adjust {feature}",
                int(df[feature].min()),
                int(df[feature].max()),
                int(patient_data[feature]),
                key=f"what_if_{feature}"
            )
        elif patient_data[feature].dtype == "float64":
             modified_patient_data[feature] = st.slider(
                f"Adjust {feature}",
                float(df[feature].min()),
                float(df[feature].max()),
                float(patient_data[feature]),
                key=f"what_if_{feature}",
                format="%.1f"
            )
        elif patient_data[feature].dtype == "object": # Assuming categorical like 'smoking' which is 0/1 for now
             modified_patient_data[feature] = st.radio(
                f"Adjust {feature}",
                options=df[feature].unique().tolist(),
                index=df[feature].unique().tolist().index(patient_data[feature]),
                key=f"what_if_{feature}"
            )

    if st.button("Run 'What-if' Analysis"): 
        st.subheader("\n'What-if' Scenario Results")
        what_if_features = data_handler.preprocess_features(modified_patient_data)
        what_if_prediction = model_inference_service.predict(what_if_features)
        st.warning(f"'What-if' Predicted Diagnosis: **{what_if_prediction}**")
        
        what_if_explanation = explanation_service.generate_lace_explanation(modified_patient_data, what_if_prediction)
        st.write("##### Feature Importance (What-if)")
        what_if_feature_importance_df = pd.DataFrame(what_if_explanation["feature_importance"].items(), columns=["Feature", "Importance"])
        st.bar_chart(what_if_feature_importance_df.set_index("Feature"))

        st.write("##### Local Rules (What-if)")
        if what_if_explanation["local_rules"]:
            for rule in what_if_explanation["local_rules"]:
                st.write(f"- {rule}")
        else:
            st.info("No specific local rules identified for this 'what-if' prediction.")

    st.subheader("User-Defined Clinical Rules")
    user_rule_input = st.text_area(
        "Enter a clinical rule (e.g., 'High blood pressure is a critical factor for heart disease'):",
        "If blood pressure is over 140, it's a concern."
    )

    if st.button("Evaluate User Rule"): # Needs to be outside if what-if button to prevent double triggering
        rule_feedback = rule_engine.evaluate_user_rule(current_explanation, user_rule_input)
        if rule_feedback:
            for fb in rule_feedback:
                st.info(fb)
        else:
            st.info("Could not evaluate the rule. Please try a different phrasing.")

    st.subheader("Diagnosis/Model Comparison (Conceptual)")
    st.info("This section would allow comparing explanations across different diagnoses for the same patient or between different AI models. For this demo, 'What-if' analysis serves as a form of comparison.")