import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


def generate_synthetic_data(n_samples=1000):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=42
    )
    feature_names = [
        "age", "blood_pressure", "number_of_previous_hospitalizations",
        "comorbidity_score", "medication_adherence"
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df["readmission_risk"] = y
    return df


def train_black_box_model(df, target_col="readmission_risk"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def generate_ice_data(
    model, 
    X_data, 
    instance_indices, 
    feature_to_vary, 
    n_grid_points=50
):
    ice_data = []
    feature_values = np.linspace(X_data[feature_to_vary].min(), X_data[feature_to_vary].max(), n_grid_points)

    for idx in instance_indices:
        instance = X_data.iloc[[idx]].copy()
        instance_predictions = []
        for val in feature_values:
            temp_instance = instance.copy()
            temp_instance[feature_to_vary] = val
            pred = model.predict_proba(temp_instance)[:, 1][0]  # Probability of positive class
            instance_predictions.append(pred)
        
        ice_data.append(pd.DataFrame({
            feature_to_vary: feature_values,
            "prediction": instance_predictions,
            "instance_id": f"Instance_{idx}"
        }))
    
    return pd.concat(ice_data)


def plot_ice_curves(ice_df, feature_to_vary, title="ICE Plot"):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=ice_df, x=feature_to_vary, y="prediction", hue="instance_id", palette="viridis", legend="full")
    plt.title(title)
    plt.xlabel(feature_to_vary)
    plt.ylabel("Predicted Readmission Risk")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Patient Instance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Generate synthetic patient data
    print("Generating synthetic patient data...")
    patient_df = generate_synthetic_data(n_samples=200) # Reduced samples for easier visualization
    print(f"Generated {len(patient_df)} patient records.")

    # 2. Train a dummy black-box model
    print("Training black-box RandomForestClassifier...")
    model, X_test, y_test = train_black_box_model(patient_df)
    print("Model trained.")

    # 3. Select a few individual patient instances from the dataset
    #    Let's pick 5 random instances from the test set
    num_instances_to_explain = 5
    selected_instance_indices = np.random.choice(X_test.index, num_instances_to_explain, replace=False)
    print(f"Selected {num_instances_to_explain} instances for explanation: {selected_instance_indices.tolist()}")

    # 4. Choose a specific feature to analyze
    feature_to_analyze = "blood_pressure"
    print(f"Analyzing feature: '{feature_to_analyze}'")

    # 5. Generate ICE data for selected instances
    print(f"Generating ICE data for feature '{feature_to_analyze}'...")
    ice_df = generate_ice_data(model, X_test, selected_instance_indices, feature_to_analyze)
    print("ICE data generated.")
    # print(ice_df.head())

    # 6. Visualize the ICE plot
    print("Plotting ICE curves...")
    plot_ice_curves(
        ice_df, 
        feature_to_analyze, 
        title=f"Individual Conditional Expectation (ICE) Plot for '{feature_to_analyze}'"
    )
    print("ICE plot displayed. Close the plot to exit.")