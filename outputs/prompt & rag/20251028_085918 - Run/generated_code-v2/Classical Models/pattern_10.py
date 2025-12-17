import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "age": np.random.randint(20, 80, num_samples),
        "gender": np.random.choice(["Male", "Female"], num_samples),
        "cholesterol": np.random.normal(200, 30, num_samples),
        "blood_pressure": np.random.normal(120, 15, num_samples),
        "smoker": np.random.choice(["Yes", "No"], num_samples),
        "bmi": np.random.normal(25, 5, num_samples),
        "glucose": np.random.normal(100, 20, num_samples),
        "family_history": np.random.choice(["Yes", "No", np.nan], num_samples, p=[0.3, 0.6, 0.1]),
        "disease": np.random.randint(0, 2, num_samples) # 0 for no disease, 1 for disease
    }
    # Introduce some missing values for demonstration
    for col in ["cholesterol", "blood_pressure", "bmi"]:
        missing_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
        data[col][missing_indices] = np.nan

    return pd.DataFrame(data)

def preprocess_data(df):
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(include="object").columns.tolist()

    # Remove the target variable from features if present
    if "disease" in numerical_features:
        numerical_features.remove("disease")

    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ])
    
    return preprocessor, numerical_features, categorical_features

def train_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessor):
    models = {
        "Logistic Regression": LogisticRegression(solver="liblinear", random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    trained_pipelines = {}
    evaluation_results = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

        evaluation_results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        }
        trained_pipelines[name] = pipeline

    return trained_pipelines, evaluation_results

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

def make_prediction(model, new_data):
    # new_data should be a pandas DataFrame with the same structure as training data features
    return model.predict(new_data)

if __name__ == "__main__":
    # 1. Generate and Load Data
    print("Generating synthetic data...")
    raw_df = generate_synthetic_data(num_samples=1000)
    X = raw_df.drop("disease", axis=1)
    y = raw_df["disease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Preprocessing
    print("Initializing preprocessor...")
    preprocessor, _, _ = preprocess_data(X_train)
    
    # 3. Train and Evaluate Models
    print("Training and evaluating models...")
    trained_pipelines, evaluation_results = train_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessor)

    print("\n--- Model Evaluation Results ---")
    best_model_name = None
    best_accuracy = -1
    for name, metrics in evaluation_results.items():
        print(f"\nModel: {name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        if metrics["Accuracy"] > best_accuracy:
            best_accuracy = metrics["Accuracy"]
            best_model_name = name
    
    print(f"\nBest performing model (based on Accuracy): {best_model_name}")

    # 4. Save the best model
    model_filename = f"{best_model_name.replace(' ', '_').lower()}_disease_predictor.joblib"
    save_model(trained_pipelines[best_model_name], model_filename)
    print(f"\nSaved the best model to {model_filename}")

    # 5. Load the model and make a prediction on new data
    print("\nLoading the saved model and making a prediction...")
    loaded_model = load_model(model_filename)

    # Simulate new patient data (ensure it has the same columns as X_train)
    new_patient_data = pd.DataFrame({
        "age": [55],
        "gender": ["Male"],
        "cholesterol": [230],
        "blood_pressure": [135],
        "smoker": ["Yes"],
        "bmi": [28.5],
        "glucose": [110],
        "family_history": ["Yes"]
    })

    prediction = make_prediction(loaded_model, new_patient_data)
    disease_status = "Disease Present" if prediction[0] == 1 else "No Disease"
    print(f"Prediction for new patient: {disease_status}")

    # Another example for a patient with potentially lower risk
    new_patient_data_2 = pd.DataFrame({
        "age": [30],
        "gender": ["Female"],
        "cholesterol": [180],
        "blood_pressure": [110],
        "smoker": ["No"],
        "bmi": [22.0],
        "glucose": [85],
        "family_history": ["No"]
    })
    prediction_2 = make_prediction(loaded_model, new_patient_data_2)
    disease_status_2 = "Disease Present" if prediction_2[0] == 1 else "No Disease"
    print(f"Prediction for another new patient: {disease_status_2}")