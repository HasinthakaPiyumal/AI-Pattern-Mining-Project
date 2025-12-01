import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Simulation Module
def simulate_patient_data(n_samples=1000):
    np.random.seed(42)
    data = {
        "age": np.random.randint(20, 90, n_samples),
        "diagnosis_code": np.random.randint(1, 10, n_samples),
        "length_of_stay": np.random.randint(1, 30, n_samples),
        "num_previous_admissions": np.random.randint(0, 5, n_samples),
        "comorbidity_index": np.random.rand(n_samples) * 10,
        "medication_adherence": np.random.rand(n_samples),
        "gender": np.random.choice([0, 1], n_samples),
        "insurance_type": np.random.choice([0, 1, 2], n_samples),
    }
    df = pd.DataFrame(data)

    # Simulate readmission likelihood based on some features
    df["readmitted"] = (
        (df["age"] > 65).astype(int) * 0.3
        + (df["num_previous_admissions"] > 1).astype(int) * 0.4
        + (df["length_of_stay"] > 10).astype(int) * 0.2
        + (df["medication_adherence"] < 0.5).astype(int) * 0.1
        + np.random.rand(n_samples) * 0.1
    ) > 0.5
    df["readmitted"] = df["readmitted"].astype(int)

    return df

# 2. Model Training Module
def train_readmission_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# 3. Permutation Feature Importance Module
def calculate_permutation_importance(model, X_test, y_test, metric):
    result = permutation_importance(model, X_test, y_test, scoring=metric, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()[::-1]
    
    importance_df = pd.DataFrame({
        "feature": X_test.columns[sorted_idx],
        "importance_mean": result.importances_mean[sorted_idx],
        "importance_std": result.importances_std[sorted_idx]
    })
    return importance_df

# 4. Results Visualization Module
def visualize_importance(importance_df, title="Permutation Feature Importance for Patient Readmission"):    
    plt.figure(figsize=(12, 7))
    sns.barplot(x="importance_mean", y="feature", data=importance_df, xerr=importance_df["importance_std"], palette="viridis")
    plt.xlabel("Mean Decrease in Model Performance (e.g., ROC AUC)")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# 5. Main Orchestration Script
if __name__ == "__main__":
    print("1. Simulating patient data...")
    patient_data = simulate_patient_data(n_samples=2000)
    
    X = patient_data.drop("readmitted", axis=1)
    y = patient_data["readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("2. Training readmission prediction model...")
    model = train_readmission_model(X_train, y_train)

    print("Evaluating initial model performance...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    baseline_roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Baseline ROC AUC on test set: {baseline_roc_auc:.4f}")

    print("3. Calculating Permutation Feature Importance...")
    feature_importance_df = calculate_permutation_importance(model, X_test, y_test, metric='roc_auc')
    print("\nFeature Importance Results:")
    print(feature_importance_df)

    print("4. Visualizing results...")
    visualize_importance(feature_importance_df)
    print("Process complete. Check the generated plot for feature importance.")