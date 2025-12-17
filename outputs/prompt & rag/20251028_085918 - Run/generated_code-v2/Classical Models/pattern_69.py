import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os

# --- 1. Data Acquisition (Simulated) ---
def create_dummy_data(filename="dummy_heart_disease.csv"):
    np.random.seed(42)
    n_samples = 1000
    data = {
        "age": np.random.randint(20, 70, n_samples),
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "cholesterol": np.random.randint(150, 300, n_samples),
        "blood_pressure": np.random.randint(90, 180, n_samples),
        "glucose": np.random.randint(70, 200, n_samples),
        "smoking": np.random.choice(["Yes", "No"], n_samples),
        "heart_rate": np.random.randint(60, 100, n_samples),
        "cardio": np.random.randint(0, 2, n_samples) # 0: No disease, 1: Disease
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df

# --- Main Script ---
if __name__ == "__main__":
    data_file = "dummy_heart_disease.csv"
    if not os.path.exists(data_file):
        print(f"Creating dummy data file: {data_file}")
        df = create_dummy_data(data_file)
    else:
        print(f"Loading data from existing file: {data_file}")
        df = pd.read_csv(data_file)

    print("\nInitial Data Head:")
    print(df.head())
    print("\nData Info:")
    print(df.info())

    # Separate features and target
    X = df.drop("cardio", axis=1)
    y = df["cardio"]

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # --- 2. Data Preprocessing and Feature Engineering ---
    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # --- 3. Model Training and 4. Model Evaluation ---
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
    }

    trained_models = {}
    best_model_name = None
    best_roc_auc = -1

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        # Create a full pipeline including preprocessing and model
        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', model)])

        full_pipeline.fit(X_train, y_train)
        trained_models[name] = full_pipeline

        # Make predictions
        y_pred = full_pipeline.predict(X_test)
        y_proba = full_pipeline.predict_proba(X_test)[:, 1]

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print("Confusion Matrix:\n", conf_matrix)

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name

    print(f"\nBest performing model based on ROC AUC: {best_model_name}")

    # --- 5. Model Persistence ---
    best_pipeline = trained_models[best_model_name]
    model_filename = f"best_{best_model_name.lower().replace(' ', '_')}_model.joblib"
    joblib.dump(best_pipeline, model_filename)
    print(f"\nBest model and preprocessor saved to {model_filename}")

    # --- 6. Prediction/Inference Example ---
    print("\n--- Demonstrating Inference ---")

    # Load the saved model
    loaded_pipeline = joblib.load(model_filename)
    print(f"Model loaded from {model_filename}")

    # Example new patient data (should be a DataFrame with same columns as original X)
    new_patient_data = pd.DataFrame({
        "age": [55, 30, 68],
        "gender": ["Male", "Female", "Female"],
        "cholesterol": [250, 180, 290],
        "blood_pressure": [140, 110, 170],
        "glucose": [120, 85, 190],
        "smoking": ["Yes", "No", "Yes"],
        "heart_rate": [85, 70, 95]
    })

    print("\nNew Patient Data for Prediction:")
    print(new_patient_data)

    # Make prediction
    prediction_proba = loaded_pipeline.predict_proba(new_patient_data)[:, 1]
    prediction_class = loaded_pipeline.predict(new_patient_data)

    for i in range(len(new_patient_data)):
        print(f"\nPatient {i+1} (Age: {new_patient_data['age'].iloc[i]}, Gender: {new_patient_data['gender'].iloc[i]}):")
        print(f"  Probability of CVD: {prediction_proba[i]:.4f}")
        print(f"  Predicted Class: {'Disease' if prediction_class[i] == 1 else 'No Disease'}")