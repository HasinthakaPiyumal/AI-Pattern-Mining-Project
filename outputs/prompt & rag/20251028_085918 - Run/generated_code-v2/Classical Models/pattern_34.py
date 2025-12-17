import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    symptoms = {
        "fever": np.random.randint(0, 2, num_samples),
        "cough": np.random.randint(0, 2, num_samples),
        "fatigue": np.random.randint(0, 2, num_samples),
        "headache": np.random.randint(0, 2, num_samples),
        "sore_throat": np.random.randint(0, 2, num_samples),
        "nausea": np.random.randint(0, 2, num_samples),
        "rash": np.random.randint(0, 2, num_samples),
    }
    df = pd.DataFrame(symptoms)

    # Simple logic to assign diseases based on symptoms
    conditions = [
        (df["fever"] == 1) & (df["cough"] == 1) & (df["fatigue"] == 1), # Flu
        (df["fever"] == 1) & (df["sore_throat"] == 1) & (df["headache"] == 1), # Cold
        (df["rash"] == 1) & (df["fever"] == 1), # Measles
        (df["nausea"] == 1) & (df["fatigue"] == 1), # Food Poisoning
    ]
    choices = ["Flu", "Cold", "Measles", "Food Poisoning"]
    df["disease"] = np.select(conditions, choices, default="Healthy")
    return df

def train_and_evaluate_model(df):
    X = df.drop("disease", axis=1)
    y = df["disease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    return model, X.columns

def predict_disease(model, symptom_columns, new_symptoms):
    new_symptoms_df = pd.DataFrame([new_symptoms], columns=symptom_columns)
    prediction = model.predict(new_symptoms_df)
    return prediction[0]

if __name__ == "__main__":
    print("Generating synthetic patient data...")
    data = generate_synthetic_data(num_samples=2000)
    print("Data generated successfully. First 5 rows:")
    print(data.head())
    print("\nTraining and evaluating the Decision Tree Classifier...")
    trained_model, feature_columns = train_and_evaluate_model(data)

    print("\nMaking a prediction for a new patient...")
    # Example: Patient with fever, cough, fatigue
    new_patient_symptoms = {"fever": 1, "cough": 1, "fatigue": 1, "headache": 0, "sore_throat": 0, "nausea": 0, "rash": 0}
    predicted_disease = predict_disease(trained_model, feature_columns, new_patient_symptoms)
    print(f"New patient symptoms: {new_patient_symptoms}")
    print(f"Predicted Disease: {predicted_disease}")

    # Example: Patient with rash and fever
    new_patient_symptoms_2 = {"fever": 1, "cough": 0, "fatigue": 0, "headache": 0, "sore_throat": 0, "nausea": 0, "rash": 1}
    predicted_disease_2 = predict_disease(trained_model, feature_columns, new_patient_symptoms_2)
    print(f"\nNew patient symptoms: {new_patient_symptoms_2}")
    print(f"Predicted Disease: {predicted_disease_2}")