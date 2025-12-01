import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def simulate_patient_data(num_samples=100):
    np.random.seed(42)
    data = {
        "age": np.random.randint(30, 80, num_samples),
        "bmi": np.random.uniform(18.0, 35.0, num_samples),
        "blood_pressure": np.random.randint(100, 180, num_samples),
        "medication_dosage": np.random.uniform(5.0, 50.0, num_samples),
        "comorbidity_score": np.random.randint(0, 10, num_samples),
    }
    df = pd.DataFrame(data)

    # Simulate adverse event risk based on a non-linear relationship
    df["adverse_event_risk"] = (
        (df["age"] * 0.02)
        + (df["bmi"] * 0.05)
        + (df["blood_pressure"] * 0.01)
        - (df["medication_dosage"] * 0.03)
        + (df["comorbidity_score"] * 0.1)
        + np.random.normal(0, 0.5, num_samples)
    )
    df["adverse_event_risk"] = (df["adverse_event_risk"] > df["adverse_event_risk"].median()).astype(int)
    return df

def generate_ice_data(model, X_original, feature_to_vary, feature_range, target_class_idx=1):
    ice_curves = []
    for i in range(len(X_original)):
        instance = X_original.iloc[[i]].copy()
        instance_ice_predictions = []
        for val in feature_range:
            instance_vary = instance.copy()
            instance_vary[feature_to_vary] = val
            # Predict probability for the target class (e.g., adverse event risk)
            prediction = model.predict_proba(instance_vary)[:, target_class_idx][0]
            instance_ice_predictions.append(prediction)
        ice_curves.append(instance_ice_predictions)
    return np.array(ice_curves)

if __name__ == "__main__":
    # 1. Data Simulation Module
    patient_data = simulate_patient_data(num_samples=200)

    X = patient_data.drop("adverse_event_risk", axis=1)
    y = patient_data["adverse_event_risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Black-box Predictive Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 3. Individual Conditional Expectation (ICE) Plot Generator
    feature_to_vary = "medication_dosage"
    # Use the range from the original data for a realistic spectrum
    feature_min = patient_data[feature_to_vary].min()
    feature_max = patient_data[feature_to_vary].max()
    feature_range = np.linspace(feature_min, feature_max, 50)

    ice_data = generate_ice_data(model, X_test, feature_to_vary, feature_range)

    # 4. Visualization Module
    plt.figure(figsize=(10, 7))
    for i in range(ice_data.shape[0]):
        plt.plot(feature_range, ice_data[i, :], color="blue", alpha=0.3)

    # Plot the average (PDP-like) line as well for comparison
    plt.plot(feature_range, np.mean(ice_data, axis=0), color="red", linewidth=3, label="Average Prediction (PDP)")

    plt.xlabel(f"Varying {feature_to_vary} (Synthetic values)")
    plt.ylabel("Predicted Probability of Adverse Event")
    plt.title(f"Individual Conditional Expectation (ICE) Plot for {feature_to_vary}")
    plt.grid(True)
    plt.legend()
    plt.show()
