import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        "age": np.random.randint(20, 90, n_samples),
        "blood_pressure": np.random.randint(90, 180, n_samples),
        "diabetes": np.random.randint(0, 2, n_samples),
        "heart_disease": np.random.randint(0, 2, n_samples),
        "num_medications": np.random.randint(1, 15, n_samples),
        "length_of_stay": np.random.randint(1, 30, n_samples),
        "previous_admissions": np.random.randint(0, 5, n_samples),
        "lab_result_a": np.random.rand(n_samples) * 100,
        "lab_result_b": np.random.rand(n_samples) * 50,
        "readmitted": np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)
    df["readmitted"] = df.apply(lambda row: 1 if (row["age"] > 70 and row["diabetes"] == 1 and row["blood_pressure"] > 140) or (row["previous_admissions"] > 2) else 0, axis=1)
    return df

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def calculate_permutation_importance(model, X_test, y_test, scoring_function):
    baseline_predictions = model.predict(X_test)
    baseline_performance = scoring_function(y_test, baseline_predictions)

    importance_scores = {}

    for feature in X_test.columns:
        X_test_permuted = X_test.copy()
        
        # Ensure the column is mutable for shuffling
        if pd.api.types.is_numeric_dtype(X_test_permuted[feature]):
            shuffled_values = X_test_permuted[feature].values.copy()
            np.random.shuffle(shuffled_values)
            X_test_permuted[feature] = shuffled_values
        else:
            # Handle non-numeric types if necessary, though for this example, all are numeric
            shuffled_values = X_test_permuted[feature].values.copy()
            np.random.shuffle(shuffled_values)
            X_test_permuted[feature] = shuffled_values

        permuted_predictions = model.predict(X_test_permuted)
        permuted_performance = scoring_function(y_test, permuted_predictions)

        importance_score = baseline_performance - permuted_performance
        importance_scores[feature] = importance_score
    
    return pd.Series(importance_scores).sort_values(ascending=False)

if __name__ == "__main__":
    print("Generating synthetic patient data...")
    df = generate_synthetic_data(n_samples=2000)

    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Training a RandomForestClassifier model...")
    trained_model = train_model(X_train, y_train)

    print("Calculating Permutation Feature Importance...")
    feature_importances = calculate_permutation_importance(trained_model, X_test, y_test, accuracy_score)

    print("\n--- Permutation Feature Importances (Ranked) ---")
    print(feature_importances)

    print("\nInterpretation: A higher score indicates that shuffling the feature significantly decreased model performance, meaning the model relies heavily on that feature for accurate predictions.")