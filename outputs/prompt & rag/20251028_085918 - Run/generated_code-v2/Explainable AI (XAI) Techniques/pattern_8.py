import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    age = np.random.randint(20, 80, num_samples)
    cholesterol = np.random.normal(200, 30, num_samples)
    blood_pressure = np.random.normal(120, 15, num_samples)
    bmi = np.random.normal(25, 5, num_samples)
    smoking = np.random.randint(0, 2, num_samples)

    # Simulate a target variable (heart disease risk) with some interactions
    heart_disease_risk = (
        0.05 * age
        + 0.02 * cholesterol
        + 0.03 * blood_pressure
        + 0.1 * bmi
        + 10 * smoking
        + np.random.normal(0, 5, num_samples)
    )
    heart_disease_risk = (heart_disease_risk - heart_disease_risk.min()) / (heart_disease_risk.max() - heart_disease_risk.min())
    
    # Convert risk to a binary outcome (e.g., risk > 0.5)
    heart_disease = (heart_disease_risk > 0.5).astype(int)

    data = pd.DataFrame({
        "Age": age,
        "Cholesterol": cholesterol,
        "BloodPressure": blood_pressure,
        "BMI": bmi,
        "Smoking": smoking,
        "HeartDisease": heart_disease,
    })
    return data

def generate_ice_data(model, X_original, feature_name, num_grid_points=100):
    ice_curves_data = []
    feature_min = X_original[feature_name].min()
    feature_max = X_original[feature_name].max()
    feature_grid = np.linspace(feature_min, feature_max, num_grid_points)

    for i in range(len(X_original)):
        instance = X_original.iloc[[i]].copy()
        temp_df = pd.DataFrame(np.repeat(instance.values, num_grid_points, axis=0), columns=X_original.columns)
        temp_df[feature_name] = feature_grid
        
        # Predict probabilities for the positive class (class 1)
        predictions = model.predict_proba(temp_df)[:, 1]
        
        for j, val in enumerate(feature_grid):
            ice_curves_data.append({
                "Instance": i,
                feature_name: val,
                "PredictedProbability": predictions[j]
            })

    return pd.DataFrame(ice_curves_data)

def plot_ice_curves(ice_data, feature_name, target_label="Predicted Probability"):
    plt.figure(figsize=(10, 7))
    sns.lineplot(data=ice_data, x=feature_name, y="PredictedProbability", hue="Instance", palette="viridis", legend=False, alpha=0.7)
    
    plt.title(f"Individual Conditional Expectation (ICE) Plot for {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel(target_label)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Data Simulation
    patient_data = generate_synthetic_data(num_samples=500)
    X = patient_data.drop("HeartDisease", axis=1)
    y = patient_data["HeartDisease"]

    # Split data for training the black-box model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Black-box Model Training
    # Using a pipeline with StandardScaler and RandomForestClassifier
    model_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf_classifier", RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5))
    ])
    model_pipeline.fit(X_train, y_train)
    print(f"Model accuracy on test set: {model_pipeline.score(X_test, y_test):.4f}")

    # 3. Select a feature for ICE plot generation
    feature_to_explain = "Cholesterol"  # Example feature
    # feature_to_explain = "Age"

    # Get ICE data for the chosen feature on a subset of the data (e.g., X_test)
    # It's good practice to use a smaller subset for ICE plots for better visualization if there are many instances
    num_instances_for_ice = 50 # Plot ICE for 50 test instances
    X_ice = X_test.sample(n=num_instances_for_ice, random_state=1)

    ice_data_df = generate_ice_data(model_pipeline, X_ice, feature_to_explain, num_grid_points=100)

    # 4. Visualize ICE curves
    plot_ice_curves(ice_data_df, feature_to_explain, target_label="Predicted Heart Disease Probability")

    print(f"ICE plot generated for the feature: {feature_to_explain}")