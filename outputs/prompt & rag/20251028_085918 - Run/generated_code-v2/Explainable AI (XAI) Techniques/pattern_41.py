import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "Age": np.random.randint(20, 70, num_samples),
        "BMI": np.random.uniform(18.0, 45.0, num_samples),
        "Glucose": np.random.uniform(70.0, 200.0, num_samples),
        "BloodPressure": np.random.uniform(60.0, 120.0, num_samples),
        "Insulin": np.random.uniform(15.0, 300.0, num_samples),
        "DiabetesPedigreeFunction": np.random.uniform(0.08, 2.4, num_samples),
        "Outcome": np.random.randint(0, 2, num_samples) # 0: No Diabetes, 1: Diabetes
    }
    df = pd.DataFrame(data)
    
    # Introduce some correlation to make the outcome more realistic
    df["Outcome"] = df.apply(lambda row: 1 if (row["Glucose"] > 140 and row["BMI"] > 30 and row["Age"] > 45) else row["Outcome"], axis=1)
    df["Outcome"] = df.apply(lambda row: 0 if (row["Glucose"] < 100 and row["BMI"] < 25 and row["Age"] < 35) else row["Outcome"], axis=1)
    df["Outcome"] = df["Outcome"].apply(lambda x: 1 if np.random.rand() < 0.1 else x) # Add some random positive cases
    df["Outcome"] = df["Outcome"].apply(lambda x: 0 if np.random.rand() < 0.1 else x) # Add some random negative cases

    return df

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    return model

def predict_and_explain(model, scaler, patient_data, feature_names, model_type="logistic_regression"):
    scaled_patient_data = scaler.transform(np.array(list(patient_data.values())).reshape(1, -1))
    prediction = model.predict(scaled_patient_data)[0]
    prediction_proba = model.predict_proba(scaled_patient_data)[0, 1]

    explanation = f"Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'} (Probability: {prediction_proba:.4f})\n"

    if model_type == "logistic_regression":
        coefficients = pd.Series(model.coef_[0], index=feature_names)
        explanation += "\nFactors influencing the prediction (Logistic Regression Coefficients):\n"
        for feature, coef in coefficients.items():
            explanation += f"  - {feature}: {coef:.4f} (Positive coefficient indicates increased risk, Negative indicates decreased risk)\n"
    elif model_type == "decision_tree":
        tree_rules = export_text(model, feature_names=list(feature_names))
        explanation += "\nDecision Tree Rules applied for prediction:\n"
        explanation += tree_rules

    return prediction, explanation


if __name__ == "__main__":
    print("--- Patient Risk Assessment System for Type 2 Diabetes ---")

    # 1. Data Generation
    df = generate_synthetic_data(num_samples=1500)
    print(f"Generated dataset with {df.shape[0]} samples and {df.shape[1] - 1} features.\n")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    feature_names = X.columns.tolist()

    # 2. Data Preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    # 3. Interpretable Model Training
    # Logistic Regression
    logistic_model = LogisticRegression(random_state=42)
    logistic_model = train_and_evaluate_model(logistic_model, X_train_scaled_df, y_train, X_test_scaled_df, y_test, "Logistic Regression")

    # Decision Tree Classifier
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42) # Limiting depth for better interpretability
    dt_model = train_and_evaluate_model(dt_model, X_train_scaled_df, y_train, X_test_scaled_df, y_test, "Decision Tree Classifier")

    # 5. Interpretability and Explanation - Displaying coefficients for Logistic Regression
    print("\n--- Logistic Regression Model Coefficients (Feature Importance) ---")
    coefficients_df = pd.DataFrame({"Feature": feature_names, "Coefficient": logistic_model.coef_[0]})
    coefficients_df["Impact"] = coefficients_df["Coefficient"].apply(lambda x: "Increases Risk" if x > 0 else ("Decreases Risk" if x < 0 else "No Impact"))
    print(coefficients_df.to_string(index=False))

    print("\n--- Decision Tree Rules (first few levels for brevity) ---")
    print(export_text(dt_model, feature_names=feature_names, max_depth=3))

    # 6. Prediction Function & 7. Main Execution Block - Demonstration for a new patient
    print("\n--- Demonstrating Prediction and Explanation for a New Patient ---")
    sample_patient_high_risk = {
        "Age": 55,
        "BMI": 32.5,
        "Glucose": 180,
        "BloodPressure": 95,
        "Insulin": 250,
        "DiabetesPedigreeFunction": 0.8
    }

    sample_patient_low_risk = {
        "Age": 30,
        "BMI": 22.0,
        "Glucose": 85,
        "BloodPressure": 70,
        "Insulin": 40,
        "DiabetesPedigreeFunction": 0.15
    }

    print("\n--- Sample Patient 1 (High Risk Profile) ---")
    print(sample_patient_high_risk)
    pred_lr_high, explanation_lr_high = predict_and_explain(logistic_model, scaler, sample_patient_high_risk, feature_names, "logistic_regression")
    print("\nLogistic Regression Explanation:")
    print(explanation_lr_high)
    pred_dt_high, explanation_dt_high = predict_and_explain(dt_model, scaler, sample_patient_high_risk, feature_names, "decision_tree")
    print("\nDecision Tree Explanation:")
    print(explanation_dt_high)

    print("\n--- Sample Patient 2 (Low Risk Profile) ---")
    print(sample_patient_low_risk)
    pred_lr_low, explanation_lr_low = predict_and_explain(logistic_model, scaler, sample_patient_low_risk, feature_names, "logistic_regression")
    print("\nLogistic Regression Explanation:")
    print(explanation_lr_low)
    pred_dt_low, explanation_dt_low = predict_and_explain(dt_model, scaler, sample_patient_low_risk, feature_names, "decision_tree")
    print("\nDecision Tree Explanation:")
    print(explanation_dt_low)
