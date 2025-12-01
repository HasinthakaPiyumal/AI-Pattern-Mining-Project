import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# --- 1. Model Training Module ---

def train_model(features_df, target_series):
    model = DecisionTreeClassifier(max_depth=4, random_state=42) # Limiting depth for interpretability
    model.fit(features_df, target_series)
    return model

# Simulate synthetic dataset
np.random.seed(42)
n_samples = 1000

data = {
    "age": np.random.randint(20, 80, n_samples),
    "bmi": np.random.uniform(18.0, 40.0, n_samples),
    "blood_pressure": np.random.randint(90, 180, n_samples),
    "cholesterol": np.random.uniform(150.0, 300.0, n_samples),
    "family_history": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]) # 0: No, 1: Yes
}

features_df = pd.DataFrame(data)

# Generate a synthetic target variable (chronic_disease)
# High age, high BMI, high blood pressure, high cholesterol, and family history increase likelihood
target_series = (
    (features_df["age"] > 55).astype(int) +
    (features_df["bmi"] > 28).astype(int) +
    (features_df["blood_pressure"] > 140).astype(int) +
    (features_df["cholesterol"] > 220).astype(int) +
    (features_df["family_history"] == 1).astype(int)
)

target_series = (target_series >= 3).astype(int) # If 3 or more risk factors, disease is likely

# Train the model
trained_dt_model = train_model(features_df, target_series)

# Save the model and feature names
model_filename = "decision_tree_model.joblib"
joblib.dump({"model": trained_dt_model, "features": features_df.columns.tolist()}, model_filename)
print(f"Model and features saved to {model_filename}")

# --- 2. Prediction Service Module ---

def load_model(filename):
    data = joblib.load(filename)
    return data["model"], data["features"]

def predict_and_explain(patient_data, model, feature_names):
    # Convert patient data to DataFrame matching model's expected input
    patient_df = pd.DataFrame([patient_data], columns=feature_names)

    # Get prediction and probabilities
    prediction = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0, 1] # Probability of chronic disease (class 1)

    explanation = []
    tree_ = model.tree_
    node_idx = 0

    explanation.append("Decision Path:")

    while tree_.feature[node_idx] != -2: # -2 indicates a leaf node
        feature_idx = tree_.feature[node_idx]
        threshold = tree_.threshold[node_idx]
        feature_name = feature_names[feature_idx]
        patient_value = patient_df[feature_name].iloc[0]

        if patient_value <= threshold:
            explanation.append(f" - If {feature_name} <= {threshold:.2f} (Patient's {feature_name}: {patient_value:.2f}), go left.")
            node_idx = tree_.children_left[node_idx]
        else:
            explanation.append(f" - If {feature_name} > {threshold:.2f} (Patient's {feature_name}: {patient_value:.2f}), go right.")
            node_idx = tree_.children_right[node_idx]

    # At a leaf node, get the class distribution
    class_counts = tree_.value[node_idx][0]
    total_samples_in_leaf = sum(class_counts)
    if total_samples_in_leaf > 0:
        leaf_probs = class_counts / total_samples_in_leaf
        explanation.append(f" - Reached leaf node with class distribution: {class_counts} (Class 0: {leaf_probs[0]:.2f}, Class 1: {leaf_probs[1]:.2f})")

    explanation.append(f"\nPrediction: {'Chronic Disease' if prediction == 1 else 'No Chronic Disease'}")
    explanation.append(f"Probability of Chronic Disease: {probability:.2%}")

    return prediction, probability, "\n".join(explanation)

# Example Usage of Prediction Service
if __name__ == "__main__":
    loaded_model, loaded_features = load_model(model_filename)

    # Example patient 1: High risk factors
    patient_1_data = {
        "age": 65,
        "bmi": 32.5,
        "blood_pressure": 160,
        "cholesterol": 250,
        "family_history": 1
    }
    pred_1, prob_1, explanation_1 = predict_and_explain(patient_1_data, loaded_model, loaded_features)
    print("\n--- Patient 1 Prediction and Explanation ---")
    print(f"Patient Data: {patient_1_data}")
    print(explanation_1)

    # Example patient 2: Low risk factors
    patient_2_data = {
        "age": 35,
        "bmi": 22.0,
        "blood_pressure": 110,
        "cholesterol": 180,
        "family_history": 0
    }
    pred_2, prob_2, explanation_2 = predict_and_explain(patient_2_data, loaded_model, loaded_features)
    print("\n--- Patient 2 Prediction and Explanation ---")
    print(f"Patient Data: {patient_2_data}")
    print(explanation_2)

    # Example patient 3: Moderate risk factors
    patient_3_data = {
        "age": 50,
        "bmi": 27.0,
        "blood_pressure": 135,
        "cholesterol": 210,
        "family_history": 0
    }
    pred_3, prob_3, explanation_3 = predict_and_explain(patient_3_data, loaded_model, loaded_features)
    print("\n--- Patient 3 Prediction and Explanation ---")
    print(f"Patient Data: {patient_3_data}")
    print(explanation_3)