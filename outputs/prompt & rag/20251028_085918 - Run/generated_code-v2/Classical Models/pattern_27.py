import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "age": np.random.randint(30, 80, num_samples),
        "gender": np.random.choice(["Male", "Female"], num_samples),
        "blood_pressure_sys": np.random.randint(100, 180, num_samples),
        "blood_pressure_dia": np.random.randint(60, 110, num_samples),
        "cholesterol": np.random.randint(150, 300, num_samples),
        "glucose": np.random.randint(70, 200, num_samples),
        "bmi": np.random.uniform(18.0, 35.0, num_samples),
        "smoking_status": np.random.choice(["Yes", "No"], num_samples),
        "exercise_hours_week": np.random.uniform(0.5, 10.0, num_samples),
        "family_history_cvd": np.random.choice(["Yes", "No"], num_samples),
        "risk_category": np.random.choice(["low", "medium", "high"], num_samples, p=[0.6, 0.3, 0.1])
    }
    df = pd.DataFrame(data)
    # Create a 'target' numerical column based on 'risk_category'
    df['target'] = df['risk_category'].map({"low": 0, "medium": 1, "high": 2})
    return df

def train_and_evaluate_models(df):
    X = df.drop(["risk_category", "target"], axis=1)
    y = df["target"]

    numerical_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(include=object).columns

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    trained_models = {}
    best_model_name = None
    best_roc_auc = -1

    print("\n--- Model Training and Evaluation ---")
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        # ROC AUC for multi-class using 'ovr' strategy
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average="weighted")

        print(f"{name} Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}\n")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name

    print(f"Best performing model based on ROC AUC: {best_model_name}")
    return trained_models[best_model_name], best_model_name, preprocessor

def predict_cvd_risk(model, preprocessor, new_patient_data):
    # Ensure new_patient_data is a DataFrame
    if isinstance(new_patient_data, dict):
        new_patient_data = pd.DataFrame([new_patient_data])
    
    # The model pipeline already includes the preprocessor, so just call predict
    prediction_numeric = model.predict(new_patient_data)
    
    # Map numerical prediction back to risk categories
    risk_map = {0: "low", 1: "medium", 2: "high"}
    predicted_risk = [risk_map[p] for p in prediction_numeric]
    
    return predicted_risk[0]

if __name__ == "__main__":
    print("Generating synthetic patient data...")
    df = generate_synthetic_data(num_samples=1500)
    print("Data head:\n", df.head())
    print("Data shape:", df.shape)

    best_model, best_model_name, preprocessor_fitted = train_and_evaluate_models(df)

    # Example of a new patient for prediction
    new_patient = {
        "age": 55,
        "gender": "Male",
        "blood_pressure_sys": 135,
        "blood_pressure_dia": 85,
        "cholesterol": 220,
        "glucose": 110,
        "bmi": 28.5,
        "smoking_status": "No",
        "exercise_hours_week": 4.0,
        "family_history_cvd": "Yes"
    }
    
    print(f"\n--- Making a prediction for a new patient using {best_model_name} ---")
    print("New Patient Data:", new_patient)
    predicted_risk = predict_cvd_risk(best_model, preprocessor_fitted, new_patient)
    print(f"Predicted Cardiovascular Disease Risk: {predicted_risk}")

    # Another example
    new_patient_2 = {
        "age": 70,
        "gender": "Female",
        "blood_pressure_sys": 160,
        "blood_pressure_dia": 95,
        "cholesterol": 280,
        "glucose": 180,
        "bmi": 32.1,
        "smoking_status": "Yes",
        "exercise_hours_week": 1.0,
        "family_history_cvd": "Yes"
    }
    print(f"\n--- Making a prediction for another new patient using {best_model_name} ---")
    print("New Patient Data:", new_patient_2)
    predicted_risk_2 = predict_cvd_risk(best_model, preprocessor_fitted, new_patient_2)
    print(f"Predicted Cardiovascular Disease Risk: {predicted_risk_2}")