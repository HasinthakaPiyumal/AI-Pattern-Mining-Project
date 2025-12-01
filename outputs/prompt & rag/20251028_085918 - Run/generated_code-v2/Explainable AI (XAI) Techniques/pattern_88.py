import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "patient_id": np.arange(num_samples),
        "age": np.random.randint(20, 90, num_samples),
        "gender": np.random.choice(["Male", "Female"], num_samples),
        "systolic_bp": np.random.randint(100, 180, num_samples),
        "diastolic_bp": np.random.randint(60, 110, num_samples),
        "cholesterol": np.random.randint(150, 300, num_samples),
        "num_previous_admissions": np.random.randint(0, 5, num_samples),
        "diabetes": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        "heart_disease": np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        "length_of_stay": np.random.randint(1, 20, num_samples),
    }
    df = pd.DataFrame(data)

    # Introduce some correlation with readmission
    df["readmitted"] = 0
    df.loc[(df["age"] > 70) | (df["num_previous_admissions"] > 1) | (df["cholesterol"] > 250) | (df["diabetes"] == 1), "readmitted"] = 1
    # Add some random readmissions to make it more realistic
    random_readmissions = np.random.choice(df.index, size=int(num_samples * 0.1), replace=False)
    df.loc[random_readmissions, "readmitted"] = 1

    # Ensure binary target for classification
    df["readmitted"] = df["readmitted"].astype(int)

    return df

def train_black_box_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model

def calculate_permutation_importance(model, X_test, y_test, feature_names, metric_scorer):
    baseline_score = metric_scorer(y_test, model.predict_proba(X_test)[:, 1])
    
    importance_scores = {}
    for feature in feature_names:
        X_test_permuted = X_test.copy()
        X_test_permuted[feature] = np.random.permutation(X_test_permuted[feature])
        
        permuted_score = metric_scorer(y_test, model.predict_proba(X_test_permuted)[:, 1])
        importance_scores[feature] = baseline_score - permuted_score
        
    sorted_importance = sorted(importance_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_importance

def plot_feature_importance(importance_scores, title="Permutation Feature Importance"):
    features = [item[0] for item in importance_scores]
    scores = [item[1] for item in importance_scores]

    plt.figure(figsize=(10, 6))
    plt.barh(features, scores)
    plt.xlabel("Importance (Drop in ROC AUC)")
    plt.ylabel("Feature")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Generating synthetic patient data...")
    df = generate_synthetic_data(num_samples=2000)
    
    # Preprocessing: One-hot encode categorical features
    df_processed = pd.get_dummies(df.drop("patient_id", axis=1), columns=["gender"], drop_first=True)
    
    X = df_processed.drop("readmitted", axis=1)
    y = df_processed["readmitted"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    feature_names = X.columns.tolist()
    
    print(f"Training a RandomForestClassifier model with {len(feature_names)} features...")
    model = train_black_box_model(X_train, y_train)
    
    # Evaluate initial model performance
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    initial_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Initial Model ROC AUC on test set: {initial_auc:.4f}")
    
    print("Calculating permutation feature importance...")
    feature_importance = calculate_permutation_importance(model, X_test, y_test, feature_names, roc_auc_score)
    
    print("\nPermutation Feature Importance Scores:")
    for feature, score in feature_importance:
        print(f"{feature}: {score:.4f}")
        
    plot_feature_importance(feature_importance, "Permutation Feature Importance for Hospital Readmission")
    print("Program finished.")