import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "age": np.random.randint(30, 80, num_samples),
        "tumor_size": np.random.uniform(1.0, 10.0, num_samples),
        "general_health_score": np.random.randint(1, 10, num_samples),
        "previous_treatment_response": np.random.choice([0, 1], num_samples, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)

    # Simulate a non-critical condition and treatment recommendations
    def recommend_treatment(row):
        if row["tumor_size"] < 3 and row["general_health_score"] > 7:
            return "Observation"
        elif row["tumor_size"] < 5 and row["age"] < 60 and row["previous_treatment_response"] == 0:
            return "Surgery"
        elif row["tumor_size"] >= 5 and row["general_health_score"] > 5:
            return "Radiation"
        else:
            return "Chemotherapy"

    df["recommended_treatment"] = df.apply(recommend_treatment, axis=1)
    return df

def train_interpretable_model(df):
    features = df[["age", "tumor_size", "general_health_score", "previous_treatment_response"]]
    target = df["recommended_treatment"]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    return model, features.columns.tolist()

def get_decision_path_explanation(model, feature_names, sample):
    node_indicator = model.decision_path(sample.to_frame().T)
    leaf_id = model.apply(sample.to_frame().T)
    
    path = []
    for node_id in node_indicator.indices:
        if leaf_id[0] == node_id:
            continue

        if sample[feature_names[model.tree_.feature[node_id]]] <= model.tree_.threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        path.append(f"If {feature_names[model.tree_.feature[node_id]]} {threshold_sign} {model.tree_.threshold[node_id]:.2f}")

    return " and ".join(path)

def recommend_treatment_with_explanation(model, feature_names, patient_data):
    patient_df = pd.Series(patient_data, index=feature_names)
    prediction = model.predict(patient_df.to_frame().T)[0]
    explanation = get_decision_path_explanation(model, feature_names, patient_df)
    
    full_explanation = f"Based on the patient's data, the recommended treatment is: {prediction}.\n"
    full_explanation += f"Reasoning: {explanation}."
    
    return prediction, full_explanation

# --- Main Execution --- #
if __name__ == "__main__":
    # 1. Data Simulation
    print("Generating synthetic patient data...")
    patient_df = generate_synthetic_data(num_samples=500)
    print(f"Generated {len(patient_df)} samples.\n")

    # 2. Model Training
    print("Training Decision Tree model...")
    dt_model, features = train_interpretable_model(patient_df)
    print("Model training complete.\n")

    # 3. Prediction and Interpretation
    print("Demonstrating treatment recommendation and explanation for new patients:\n")

    # Example Patient 1: Likely 'Observation'
    patient1_data = {
        "age": 45,
        "tumor_size": 2.5,
        "general_health_score": 8,
        "previous_treatment_response": 0
    }
    pred1, explanation1 = recommend_treatment_with_explanation(dt_model, features, patient1_data)
    print(f"Patient 1 Data: {patient1_data}")
    print(explanation1)
    print("\n" + "-"*50 + "\n")

    # Example Patient 2: Likely 'Surgery'
    patient2_data = {
        "age": 55,
        "tumor_size": 4.0,
        "general_health_score": 6,
        "previous_treatment_response": 0
    }
    pred2, explanation2 = recommend_treatment_with_explanation(dt_model, features, patient2_data)
    print(f"Patient 2 Data: {patient2_data}")
    print(explanation2)
    print("\n" + "-"*50 + "\n")

    # Example Patient 3: Likely 'Radiation'
    patient3_data = {
        "age": 70,
        "tumor_size": 6.0,
        "general_health_score": 7,
        "previous_treatment_response": 1
    }
    pred3, explanation3 = recommend_treatment_with_explanation(dt_model, features, patient3_data)
    print(f"Patient 3 Data: {patient3_data}")
    print(explanation3)
    print("\n" + "-"*50 + "\n")