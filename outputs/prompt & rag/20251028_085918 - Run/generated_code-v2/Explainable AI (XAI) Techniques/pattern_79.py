import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def simulate_patient_data(n_samples=1000):
    np.random.seed(42)
    data = {
        "age": np.random.randint(20, 90, n_samples),
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "num_previous_admissions": np.random.randint(0, 5, n_samples),
        "length_of_stay": np.random.randint(1, 30, n_samples),
        "diagnosis_group": np.random.choice(["Cardio", "Pulmonary", "Diabetic", "Ortho", "Other"], n_samples),
        "medication_adherence_score": np.random.uniform(0.5, 1.0, n_samples),
        "comorbidity_score": np.random.uniform(0.1, 0.9, n_samples),
        "readmitted": np.random.randint(0, 2, n_samples) # 0: No, 1: Yes
    }
    df = pd.DataFrame(data)
    
    # Make readmission slightly dependent on some features
    df["readmitted"] = df.apply(lambda row: 1 if (row["age"] > 70 and row["num_previous_admissions"] > 1 and row["diagnosis_group"] == "Cardio") or (row["length_of_stay"] < 5 and row["medication_adherence_score"] < 0.7) else row["readmitted"], axis=1)
    df["readmitted"] = df.apply(lambda row: 0 if (row["age"] < 40 and row["comorbidity_score"] < 0.3) else row["readmitted"], axis=1)
    
    return df

def calculate_permutation_importance(model, X_test, y_test, metric=accuracy_score, n_repeats=5):
    baseline_performance = metric(y_test, model.predict(X_test))
    
    feature_importances = {}
    for col in X_test.columns:
        original_col_values = X_test[col].copy()
        permuted_performances = []
        for _ in range(n_repeats):
            X_test_permuted = X_test.copy()
            X_test_permuted[col] = np.random.permutation(original_col_values)
            permuted_performances.append(metric(y_test, model.predict(X_test_permuted)))
        
        avg_permuted_performance = np.mean(permuted_performances)
        feature_importances[col] = baseline_performance - avg_permuted_performance
        
    return pd.DataFrame({"Feature": list(feature_importances.keys()), "Importance": list(feature_importances.values())}).sort_values(by="Importance", ascending=False)

if __name__ == "__main__":
    # 1. Data Ingestion and Preprocessing
    print("1. Simulating and Preprocessing Data...")
    df = simulate_patient_data(n_samples=2000)

    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    categorical_features = [col for col in X.columns if X[col].dtype == "object"]
    numerical_features = [col for col in X.columns if X[col].dtype != "object"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ])

    # 2. Black-box Machine Learning Model Training
    print("2. Training Black-box Machine Learning Model...")
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    model_pipeline.fit(X_train, y_train)

    # Get processed feature names for permutation importance
    encoded_feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
    X_test_processed = pd.DataFrame(model_pipeline.named_steps["preprocessor"].transform(X_test), columns=encoded_feature_names)

    # 3. Permutation Feature Importance Module
    print("3. Calculating Permutation Feature Importance...")
    permutation_importances = calculate_permutation_importance(
        model_pipeline.named_steps["classifier"], 
        X_test_processed, 
        y_test, 
        metric=f1_score # Using F1-score as the metric for a balanced view
    )

    # 4. Evaluation and Reporting
    print("4. Evaluating Model and Reporting Results...")
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    print(f"\nModel Performance on Test Set:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC AUC:  {roc_auc_score(y_test, y_proba):.4f}")

    print("\nPermutation Feature Importance (ranked by importance drop - F1-score):")
    print(permutation_importances.to_string(index=False))

    print("\nExplanation: A higher 'Importance' score indicates that permuting the values of that feature led to a larger drop in the model's F1-score, meaning the model relies heavily on that feature for accurate predictions of patient readmission risk.")