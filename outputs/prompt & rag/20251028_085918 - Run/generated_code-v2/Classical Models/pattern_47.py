import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 70, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples),
        'fever': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'cough': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
        'fatigue': np.random.choice([0, 1], num_samples, p=[0.5, 0.5]),
        'blood_pressure': np.random.randint(90, 180, num_samples),
        'cholesterol': np.random.randint(150, 250, num_samples),
        'white_blood_cell_count': np.random.randint(4000, 12000, num_samples),
        'diabetes_history': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        'heart_disease_history': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
        'disease_present': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]) # Target variable
    }
    df = pd.DataFrame(data)
    
    # Introduce some correlation for disease_present
    df.loc[(df['fever'] == 1) & (df['cough'] == 1) & (df['white_blood_cell_count'] > 9000), 'disease_present'] = 1
    df.loc[(df['diabetes_history'] == 1) & (df['blood_pressure'] > 140), 'disease_present'] = 1
    df.loc[(df['heart_disease_history'] == 1) & (df['cholesterol'] > 200), 'disease_present'] = 1

    return df

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
        'Support Vector Machine': SVC(random_state=42, probability=True),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    trained_models = {}
    results = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': cr
        }
        trained_models[name] = pipeline

        print(f"{name} - Accuracy: {accuracy:.4f}")
        print(f"{name} - Precision: {precision:.4f}")
        print(f"{name} - Recall: {recall:.4f}")
        print(f"{name} - F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification Report:\n{cr}")

    return trained_models, results

def predict_new_patient_data(patient_data, model_pipeline):
    # patient_data should be a pandas DataFrame with one row
    prediction = model_pipeline.predict(patient_data)
    probability = model_pipeline.predict_proba(patient_data)[:, 1] if hasattr(model_pipeline.named_steps['classifier'], 'predict_proba') else None
    
    status = "Disease Present" if prediction[0] == 1 else "Disease Absent"
    return status, probability[0] if probability is not None else "N/A"

if __name__ == "__main__":
    print("--- Generating Synthetic Patient Data ---")
    df = generate_synthetic_data(num_samples=1000)
    # print(df.head())
    # print(df['disease_present'].value_counts())

    X = df.drop('disease_present', axis=1)
    y = df['disease_present']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define preprocessing steps
    numerical_features = ['age', 'blood_pressure', 'cholesterol', 'white_blood_cell_count']
    categorical_features = ['gender', 'fever', 'cough', 'fatigue', 'diabetes_history', 'heart_disease_history']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    print("\n--- Training and Evaluating Classical Models ---")
    trained_models, evaluation_results = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)

    print("\n--- Demonstrating Prediction on New Data ---")
    # Create a new patient record (example)
    new_patient = pd.DataFrame({
        'age': [55],
        'gender': ['Male'],
        'fever': [1],
        'cough': [1],
        'fatigue': [0],
        'blood_pressure': [160],
        'cholesterol': [230],
        'white_blood_cell_count': [10500],
        'diabetes_history': [1],
        'heart_disease_history': [0]
    })

    print(f"\nNew Patient Data:\n{new_patient.iloc[0].to_dict()}")

    for name, model_pipeline in trained_models.items():
        prediction_status, prediction_proba = predict_new_patient_data(new_patient, model_pipeline)
        print(f"Prediction using {name}: {prediction_status} (Probability: {prediction_proba:.4f})")

    print("\n--- System Workflow Completed ---")