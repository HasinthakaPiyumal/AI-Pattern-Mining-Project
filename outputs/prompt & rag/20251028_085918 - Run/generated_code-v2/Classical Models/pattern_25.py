import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# 1. Data Ingestion and Preprocessing (Simulation)
def simulate_ehr_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 90, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'NumDiagnoses': np.random.randint(1, 15, num_samples),
        'NumMedications': np.random.randint(1, 30, num_samples),
        'LengthOfStay': np.random.randint(1, 30, num_samples),
        'PreviousAdmissions': np.random.randint(0, 5, num_samples),
        'DiagnosisCode': np.random.choice([f'ICD{i:03d}' for i in range(1, 20)], num_samples),
        'readmitted_within_30_days': np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
    }
    df = pd.DataFrame(data)
    
    # Introduce some missing values for demonstration
    for col in ['NumMedications', 'LengthOfStay']:
        df.loc[df.sample(frac=0.02).index, col] = np.nan
    df.loc[df.sample(frac=0.01).index, 'DiagnosisCode'] = np.nan

    return df

def preprocess_data(df):
    X = df.drop('readmitted_within_30_days', axis=1)
    y = df['readmitted_within_30_days']

    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor, X, y

# 3. Model Training and Selection
def train_evaluate_models(preprocessor, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    best_model = None
    best_roc_auc = -1
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # 4. Model Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"{name} Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")

        trained_models[name] = pipeline

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = pipeline

    return best_model, trained_models

# 6. Prediction Function
def predict_readmission(model, new_patient_data):
    # Ensure new_patient_data is a DataFrame and matches expected columns
    if isinstance(new_patient_data, dict):
        new_patient_data = pd.DataFrame([new_patient_data])
    
    prediction = model.predict(new_patient_data)
    probability = model.predict_proba(new_patient_data)[:, 1]
    
    return prediction[0], probability[0]

# Main execution flow
if __name__ == "__main__":
    print("Simulating EHR data...")
    df = simulate_ehr_data(num_samples=1000)
    # print("\nSimulated Data Head:")
    # print(df.head())
    # print("\nMissing Values:")
    # print(df.isnull().sum())

    print("\nPreprocessing data...")
    preprocessor, X, y = preprocess_data(df)
    
    print("\nTraining and evaluating models...")
    best_model, all_models = train_evaluate_models(preprocessor, X, y)

    if best_model:
        print(f"\nBest model (based on ROC-AUC) selected.")
        # 5. Model Persistence
        model_filename = 'best_readmission_model.joblib'
        joblib.dump(best_model, model_filename)
        print(f"Best model saved as '{model_filename}'")

        # Demonstrate prediction with a new patient
        print("\nDemonstrating prediction with new patient data...")
        new_patient_info = {
            'Age': 75,
            'Gender': 'Female',
            'NumDiagnoses': 8,
            'NumMedications': 20,
            'LengthOfStay': 10,
            'PreviousAdmissions': 1,
            'DiagnosisCode': 'ICD007'
        }
        
        predicted_readmission, prediction_proba = predict_readmission(best_model, new_patient_info)
        print(f"New Patient Info: {new_patient_info}")
        print(f"Predicted Readmission (0=No, 1=Yes): {predicted_readmission}")
        print(f"Readmission Probability: {prediction_proba:.4f}")
    else:
        print("No models were trained or selected.")