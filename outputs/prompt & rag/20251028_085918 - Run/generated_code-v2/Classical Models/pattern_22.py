import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "Age": np.random.randint(20, 80, num_samples),
        "Gender": np.random.choice(["Male", "Female"], num_samples),
        "Blood_Pressure": np.random.randint(90, 180, num_samples),
        "Cholesterol": np.random.randint(150, 300, num_samples),
        "Smoking": np.random.choice(["Yes", "No"], num_samples),
        "Exercise": np.random.choice(["High", "Medium", "Low"], num_samples),
        "Family_History": np.random.choice(["Yes", "No"], num_samples, p=[0.3, 0.7]),
        "Glucose": np.random.randint(70, 200, num_samples),
        "Disease_Presence": np.random.randint(0, 2, num_samples, p=[0.6, 0.4])
    }
    df = pd.DataFrame(data)
    # Introduce some missing values
    for col in ["Blood_Pressure", "Cholesterol", "Glucose"]:
        df.loc[df.sample(frac=0.05).index, col] = np.nan
    return df

def train_predict_and_save_model(data_path=None):
    if data_path:
        df = pd.read_csv(data_path)
    else:
        df = generate_synthetic_data()
        print("Generated synthetic data.")

    X = df.drop("Disease_Presence", axis=1)
    y = df["Disease_Presence"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numerical_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(include="object").columns

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

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("Training model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    y_pred = model.predict(X_test)

    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, "disease_prediction_model.joblib")
    joblib.dump(preprocessor, "preprocessor.joblib")
    print("Model and preprocessor saved as 'disease_prediction_model.joblib' and 'preprocessor.joblib'.")

def predict_new_patient(patient_data):
    try:
        model = joblib.load("disease_prediction_model.joblib")
        preprocessor = joblib.load("preprocessor.joblib") # Load preprocessor if it was saved separately or if it's part of the model pipeline
        
        # If the preprocessor is part of the model pipeline, direct loading isn't strictly needed for prediction
        # but it's good practice to ensure the same preprocessing steps are applied.
        # If the pipeline itself was saved, then it contains the preprocessor.
        
        new_patient_df = pd.DataFrame([patient_data])
        prediction = model.predict(new_patient_df)
        prediction_proba = model.predict_proba(new_patient_df)

        disease_status = "Present" if prediction[0] == 1 else "Absent"
        print(f"\nPrediction for new patient: Disease is {disease_status}.")
        print(f"Probability of Disease Absence: {prediction_proba[0][0]:.4f}")
        print(f"Probability of Disease Presence: {prediction_proba[0][1]:.4f}")
        return prediction[0], prediction_proba[0]

    except FileNotFoundError:
        print("Error: Model or preprocessor not found. Please train and save the model first.")
        return None, None

if __name__ == "__main__":
    # Run the full training, evaluation, and saving pipeline
    train_predict_and_save_model()

    # Example of predicting for a new patient
    new_patient_info = {
        "Age": 55,
        "Gender": "Male",
        "Blood_Pressure": 140,
        "Cholesterol": 220,
        "Smoking": "Yes",
        "Exercise": "Low",
        "Family_History": "Yes",
        "Glucose": 160
    }
    predict_new_patient(new_patient_info)