import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

def simulate_ehr_data(num_samples=1000):
    np.random.seed(42)

    data = {
        "age": np.random.randint(20, 90, num_samples),
        "gender": np.random.choice(["Male", "Female"], num_samples),
        "num_diagnoses": np.random.randint(1, 10, num_samples),
        "chronic_conditions": np.random.randint(0, 5, num_samples),
        "length_of_stay": np.random.randint(1, 30, num_samples),
        "num_procedures": np.random.randint(0, 8, num_samples),
        "discharge_disposition": np.random.choice(["Home", "Rehab", "SNF", "Other"], num_samples),
        "readmitted": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]) # 30% readmission rate
    }
    df = pd.DataFrame(data)
    return df

def train_readmission_model(df):
    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    categorical_features = ['gender', 'discharge_disposition']
    numerical_features = ['age', 'num_diagnoses', 'chronic_conditions', 'length_of_stay', 'num_procedures']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)

    print(f"Model Evaluation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"\nClassification Report:\n{report}")

    return model, preprocessor

def predict_readmission(model, preprocessor, new_patient_data):
    # Ensure new_patient_data is a DataFrame with the same columns as the training data
    if not isinstance(new_patient_data, pd.DataFrame):
        new_patient_data = pd.DataFrame([new_patient_data])
    
    # The full pipeline handles preprocessing, so just pass the raw data
    prediction = model.predict(new_patient_data)
    probability = model.predict_proba(new_patient_data)[:, 1]
    
    return prediction[0], probability[0]

if __name__ == "__main__":
    print("Simulating EHR data...")
    ehr_data = simulate_ehr_data()
    print("Data simulation complete. Sample data head:")
    print(ehr_data.head())

    print("\nTraining readmission prediction model...")
    trained_model, data_preprocessor = train_readmission_model(ehr_data)
    print("Model training and evaluation complete.")

    print("\nDemonstrating prediction on new data...")
    # Example new patient data (should match the structure of original features)
    new_patient = {
        "age": 75,
        "gender": "Female",
        "num_diagnoses": 4,
        "chronic_conditions": 2,
        "length_of_stay": 10,
        "num_procedures": 1,
        "discharge_disposition": "Home"
    }
    
    readmission_prediction, readmission_probability = predict_readmission(trained_model, data_preprocessor, new_patient)
    
    print(f"\nNew Patient Data: {new_patient}")
    print(f"Predicted Readmission: {'Yes' if readmission_prediction == 1 else 'No'}")
    print(f"Probability of Readmission: {readmission_probability:.4f}")

    new_patient_2 = {
        "age": 50,
        "gender": "Male",
        "num_diagnoses": 8,
        "chronic_conditions": 3,
        "length_of_stay": 25,
        "num_procedures": 5,
        "discharge_disposition": "Rehab"
    }
    readmission_prediction_2, readmission_probability_2 = predict_readmission(trained_model, data_preprocessor, new_patient_2)

    print(f"\nNew Patient Data: {new_patient_2}")
    print(f"Predicted Readmission: {'Yes' if readmission_prediction_2 == 1 else 'No'}")
    print(f"Probability of Readmission: {readmission_probability_2:.4f}")