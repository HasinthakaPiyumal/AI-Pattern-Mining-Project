import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay


def generate_patient_data(n_samples=1000):
    np.random.seed(42)
    age = np.random.randint(18, 90, n_samples)
    comorbidities = np.random.randint(0, 5, n_samples)
    length_of_stay = np.random.randint(1, 30, n_samples)
    medication_adherence = np.random.rand(n_samples)

    # Simulate a non-linear relationship for readmission risk
    readmission_risk_proba = (
        0.1 * (age / 90)
        + 0.2 * (comorbidities / 5)
        + 0.05 * (length_of_stay / 30)
        - 0.1 * medication_adherence
        + 0.05 * np.sin(age / 10)
        + 0.05 * np.random.rand(n_samples)
    )
    readmission_risk = (readmission_risk_proba > 0.4).astype(int) # Binary classification

    data = pd.DataFrame({
        "age": age,
        "comorbidities": comorbidities,
        "length_of_stay": length_of_stay,
        "medication_adherence": medication_adherence,
        "readmission_risk": readmission_risk,
    })
    return data


def train_black_box_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    return model


def plot_partial_dependence(model, X, features_to_plot):
    fig, ax = plt.subplots(figsize=(12, 8))
    display = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=features_to_plot,
        kind="average",
        ax=ax,
        n_jobs=-1,
        grid_resolution=50,
    )
    fig.suptitle(f"Partial Dependence Plots for {', '.join(features_to_plot)}", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


if __name__ == "__main__":
    # 1. Data Generation/Loading Module
    patient_data = generate_patient_data()
    print("Generated patient data (first 5 rows):\n", patient_data.head())

    X = patient_data.drop("readmission_risk", axis=1)
    y = patient_data["readmission_risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Black-Box Model Module
    black_box_model = train_black_box_model(X_train, y_train)
    print(f"\nModel trained. Accuracy on test set: {black_box_model.score(X_test, y_test):.4f}")

    # 3. Partial Dependence Plot (PDP) Module
    print("\nGenerating Partial Dependence Plots...")

    # Example 1: Single feature PDP
    selected_features_1 = ["age"]
    plot_partial_dependence(black_box_model, X_train, selected_features_1)

    # Example 2: Another single feature PDP
    selected_features_2 = ["comorbidities"]
    plot_partial_dependence(black_box_model, X_train, selected_features_2)

    # Example 3: Two-feature PDP (interaction)
    selected_features_3 = [("age", "length_of_stay")]
    plot_partial_dependence(black_box_model, X_train, selected_features_3)

    print("PDP generation complete. Plots displayed (if running in an interactive environment).")