import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def simulate_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'BMI': np.random.uniform(18, 40, n_samples),
        'Cholesterol': np.random.randint(150, 250, n_samples),
        'BloodPressure_Systolic': np.random.randint(90, 180, n_samples),
        'BloodPressure_Diastolic': np.random.randint(60, 120, n_samples),
        'Smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'DiabetesFamilyHistory': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'HeartDisease': np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
    }
    df = pd.DataFrame(data)
    df.loc[df['Age'] > 60, 'HeartDisease'] = np.random.choice([0, 1], sum(df['Age'] > 60), p=[0.4, 0.6])
    df.loc[df['BMI'] > 30, 'HeartDisease'] = np.random.choice([0, 1], sum(df['BMI'] > 30), p=[0.5, 0.5])
    df.loc[df['Cholesterol'] > 200, 'HeartDisease'] = np.random.choice([0, 1], sum(df['Cholesterol'] > 200), p=[0.5, 0.5])
    return df

def train_and_evaluate_models(df):
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numerical_features = ['Age', 'BMI', 'Cholesterol', 'BloodPressure_Systolic', 'BloodPressure_Diastolic']
    categorical_features = ['Gender', 'Smoking', 'DiabetesFamilyHistory']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    best_model = None
    best_accuracy = 0
    evaluation_results = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        evaluation_results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        }

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline

    return best_model, evaluation_results

def predict_disease(model, new_patient_data):
    prediction = model.predict(new_patient_data)
    probability = model.predict_proba(new_patient_data)[:, 1]
    return prediction[0], probability[0]

if __name__ == '__main__':
    print("Simulating patient data...")
    patient_df = simulate_data(n_samples=2000)
    print(f"Generated {len(patient_df)} patient records.")
    print("Sample data head:\n", patient_df.head())

    print("\nTraining and evaluating classical machine learning models...")
    best_model, results = train_and_evaluate_models(patient_df)

    print("\n--- Model Evaluation Results ---")
    for name, metrics in results.items():
        print(f"\nModel: {name}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  Confusion Matrix: {metrics['confusion_matrix']}")

    if best_model:
        print(f"\nBest performing model based on accuracy: {best_model.named_steps["classifier"].__class__.__name__}")

        print("\n--- Demonstrating Prediction on New Data ---")
        new_patient = pd.DataFrame({
            'Age': [55],
            'Gender': ['Female'],
            'BMI': [28.5],
            'Cholesterol': [210],
            'BloodPressure_Systolic': [135],
            'BloodPressure_Diastolic': [85],
            'Smoking': [0],
            'DiabetesFamilyHistory': [1],
        })

        disease_prediction, disease_probability = predict_disease(best_model, new_patient)
        print(f"New Patient Data:\n{new_patient.iloc[0].to_dict()}")
        risk_category = "High Risk" if disease_prediction == 1 else "Low Risk"
        print(f"Predicted Heart Disease Risk: {risk_category} (Probability: {disease_probability:.4f})")

        new_patient_high_risk = pd.DataFrame({
            'Age': [70],
            'Gender': ['Male'],
            'BMI': [32.1],
            'Cholesterol': [245],
            'BloodPressure_Systolic': [160],
            'BloodPressure_Diastolic': [100],
            'Smoking': [1],
            'DiabetesFamilyHistory': [1],
        })
        disease_prediction_high_risk, disease_probability_high_risk = predict_disease(best_model, new_patient_high_risk)
        risk_category_high_risk = "High Risk" if disease_prediction_high_risk == 1 else "Low Risk"
        print(f"\nNew Patient Data (High Risk Example):\n{new_patient_high_risk.iloc[0].to_dict()}")
        print(f"Predicted Heart Disease Risk: {risk_category_high_risk} (Probability: {disease_probability_high_risk:.4f})")

    else:
        print("No model was trained successfully.")