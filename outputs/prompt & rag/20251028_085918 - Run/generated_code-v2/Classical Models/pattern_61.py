import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- 1. Data Generation (Synthetic) ---
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'patient_id': range(num_samples),
        'age': np.random.randint(18, 90, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples, p=[0.49, 0.51]),
        'admission_type': np.random.choice(['Emergency', 'Elective', 'Urgent'], num_samples, p=[0.6, 0.2, 0.2]),
        'major_diagnosis': np.random.choice(['Cardiovascular', 'Respiratory', 'Diabetes', 'Cancer', 'Injury', 'Other'], num_samples, p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.3]),
        'num_procedures': np.random.randint(0, 5, num_samples),
        'lab_result_A': np.random.normal(100, 15, num_samples), # e.g., Blood Pressure
        'lab_result_B': np.random.normal(5.0, 1.0, num_samples),  # e.g., Glucose Level
        'length_of_stay': np.random.randint(1, 30, num_samples),
        'num_previous_admissions': np.random.randint(0, 6, num_samples),
        'readmitted': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]) # 0: No, 1: Yes
    }
    df = pd.DataFrame(data)
    
    # Introduce some correlation for readmission
    df.loc[df['length_of_stay'] > 15, 'readmitted'] = np.random.choice([0, 1], len(df[df['length_of_stay'] > 15]), p=[0.6, 0.4])
    df.loc[df['num_previous_admissions'] > 2, 'readmitted'] = np.random.choice([0, 1], len(df[df['num_previous_admissions'] > 2]), p=[0.5, 0.5])
    df.loc[df['age'] > 70, 'readmitted'] = np.random.choice([0, 1], len(df[df['age'] > 70]), p=[0.7, 0.3])
    
    return df

# --- 2. Data Preprocessing and Feature Engineering ---
def preprocess_data(df):
    # Define categorical and numerical features
    categorical_features = ['gender', 'admission_type', 'major_diagnosis']
    numerical_features = [
        'age', 'num_procedures', 'lab_result_A', 'lab_result_B',
        'length_of_stay', 'num_previous_admissions'
    ]

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# --- 3. Model Training and Selection ---
def train_models(X_train, y_train):
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, solver='liblinear'),
        'RandomForestClassifier': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBClassifier': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"Trained {name} model.")
    return trained_models

# --- 4. Model Evaluation ---
def evaluate_models(models, X_test, y_test):
    results = {}
    best_model_name = None
    best_roc_auc = -1

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }

        print(f"\n--- {name} Evaluation ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            
    print(f"\nBest performing model based on ROC AUC: {best_model_name} (ROC AUC: {best_roc_auc:.4f})")
    return results, best_model_name

# --- 5. Main Execution and Model Saving ---
if __name__ == '__main__':
    print("Generating synthetic data...")
    df = generate_synthetic_data(num_samples=2000)

    # Separate features (X) and target (y)
    X = df.drop(['patient_id', 'readmitted'], axis=1)
    y = df['readmitted']

    # Split data into training and testing sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Initializing preprocessor...")
    preprocessor = preprocess_data(X_train_raw)

    print("Fitting and transforming training data...")
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    print("Training models...")
    trained_models = train_models(X_train_processed, y_train)

    print("Evaluating models...")
    evaluation_results, best_model_name = evaluate_models(trained_models, X_test_processed, y_test)

    best_model = trained_models[best_model_name]
    
    # Save the preprocessor and the best model
    joblib.dump(preprocessor, 'readmission_preprocessor.joblib')
    joblib.dump(best_model, 'best_readmission_model.joblib')
    print(f"\nPreprocessor saved to 'readmission_preprocessor.joblib'")
    print(f"Best model ({best_model_name}) saved to 'best_readmission_model.joblib'")

    # --- 6. Prediction Service (Example) ---
    def predict_readmission(new_patient_data_raw):
        # Load the preprocessor and model
        loaded_preprocessor = joblib.load('readmission_preprocessor.joblib')
        loaded_model = joblib.load('best_readmission_model.joblib')
        
        # Ensure the input is a DataFrame with the correct columns
        new_df = pd.DataFrame([new_patient_data_raw])
        
        # Preprocess the new data
        new_data_processed = loaded_preprocessor.transform(new_df)
        
        # Make prediction
        prediction_proba = loaded_model.predict_proba(new_data_processed)[:, 1][0]
        prediction_class = loaded_model.predict(new_data_processed)[0]
        
        risk_label = "High Risk" if prediction_class == 1 else "Low Risk"
        
        return {
            'readmission_probability': float(prediction_proba),
            'readmission_risk': risk_label,
            'predicted_class': int(prediction_class)
        }

    print("\n--- Demonstrating Prediction Service ---")
    # Example new patient data
    example_patient = {
        'age': 75,
        'gender': 'Female',
        'admission_type': 'Emergency',
        'major_diagnosis': 'Cardiovascular',
        'num_procedures': 3,
        'lab_result_A': 130.0,
        'lab_result_B': 8.0,
        'length_of_stay': 20,
        'num_previous_admissions': 4
    }

    prediction = predict_readmission(example_patient)
    print(f"Prediction for example patient: {prediction}")
    
    example_patient_low_risk = {
        'age': 30,
        'gender': 'Male',
        'admission_type': 'Elective',
        'major_diagnosis': 'Other',
        'num_procedures': 0,
        'lab_result_A': 90.0,
        'lab_result_B': 4.0,
        'length_of_stay': 3,
        'num_previous_admissions': 0
    }
    prediction_low_risk = predict_readmission(example_patient_low_risk)
    print(f"Prediction for low risk example patient: {prediction_low_risk}")
