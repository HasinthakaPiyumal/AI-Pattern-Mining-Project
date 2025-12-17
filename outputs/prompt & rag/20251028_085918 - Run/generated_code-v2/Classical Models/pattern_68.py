import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 90, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples),
        'medical_condition': np.random.choice(['Diabetes', 'Heart Disease', 'Hypertension', 'Asthma', 'None'], num_samples),
        'num_admissions_last_year': np.random.randint(0, 5, num_samples),
        'length_of_stay_days': np.random.randint(1, 30, num_samples),
        'lab_results_avg': np.random.normal(100, 15, num_samples),
        'medication_count': np.random.randint(1, 10, num_samples),
        'readmitted': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]) # 0: No, 1: Yes
    }
    return pd.DataFrame(data)

def train_and_evaluate_model():
    df = generate_synthetic_data()

    X = df.drop('readmitted', axis=1)
    y = df['readmitted']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    categorical_features = ['gender', 'medical_condition']
    numerical_features = ['age', 'num_admissions_last_year', 'length_of_stay_days', 'lab_results_avg', 'medication_count']

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

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model

if __name__ == '__main__':
    trained_model = train_and_evaluate_model()

    print("\n--- Demonstrating Prediction on New Data ---")
    new_patient_data = pd.DataFrame({
        'age': [75, 45],
        'gender': ['Female', 'Male'],
        'medical_condition': ['Heart Disease', 'Asthma'],
        'num_admissions_last_year': [2, 0],
        'length_of_stay_days': [10, 3],
        'lab_results_avg': [120, 85],
        'medication_count': [6, 2]
    })

    new_predictions = trained_model.predict(new_patient_data)
    new_probabilities = trained_model.predict_proba(new_patient_data)[:, 1]

    print("\nNew Patient Data:")
    print(new_patient_data)
    print("\nPredicted Readmission (0=No, 1=Yes):")
    print(new_predictions)
    print("\nPredicted Readmission Probability:")
    print(new_probabilities)
