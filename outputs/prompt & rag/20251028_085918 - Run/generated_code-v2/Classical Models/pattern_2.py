import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def generate_simulated_ehr_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'patient_id': range(1, num_samples + 1),
        'age': np.random.randint(18, 90, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples),
        'num_diagnoses': np.random.randint(1, 10, num_samples),
        'num_procedures': np.random.randint(0, 5, num_samples),
        'lab_result_a': np.random.rand(num_samples) * 100,
        'lab_result_b': np.random.rand(num_samples) * 50,
        'medication_count': np.random.randint(1, 8, num_samples),
        'prior_admissions': np.random.randint(0, 3, num_samples),
        'length_of_stay': np.random.randint(1, 20, num_samples),
        'readmitted_30_days': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]) # 20% readmission rate
    }
    df = pd.DataFrame(data)
    
    # Introduce some missing values for demonstration
    for col in ['lab_result_a', 'medication_count']:
        missing_indices = np.random.choice(df.index, int(num_samples * 0.05), replace=False)
        df.loc[missing_indices, col] = np.nan
    
    return df

def train_readmission_model(df):
    X = df.drop(['patient_id', 'readmitted_30_days'], axis=1)
    y = df['readmitted_30_days']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

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
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model_pipeline.fit(X_train, y_train)
    
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    # plt.show()

    return model_pipeline

def predict_readmission_risk(model, new_patient_data):
    prediction = model.predict(new_patient_data)
    prediction_proba = model.predict_proba(new_patient_data)[:, 1]
    return prediction[0], prediction_proba[0]

if __name__ == "__main__":
    # 1. Generate Simulated EHR Data
    print("Generating simulated EHR data...")
    ehr_df = generate_simulated_ehr_data()
    print(f"Simulated data shape: {ehr_df.shape}")
    print(ehr_df.head())

    # 2. Train Model
    print("\nTraining readmission prediction model...")
    from sklearn.impute import SimpleImputer # Moved here due to strict single code block requirement
    trained_model = train_readmission_model(ehr_df)
    
    # 3. Model Persistence (Save the trained model)
    model_filename = 'readmission_prediction_model.joblib'
    joblib.dump(trained_model, model_filename)
    print(f"\nModel saved as \'{model_filename}\'")

    # 4. Model Loading and Prediction Example
    print(f"\nLoading model from \'{model_filename}\' and making a prediction...")
    loaded_model = joblib.load(model_filename)

    # Create a sample patient for prediction
    sample_patient = pd.DataFrame([{
        'age': 65,
        'gender': 'Female',
        'num_diagnoses': 4,
        'num_procedures': 1,
        'lab_result_a': 75.5,
        'lab_result_b': 28.1,
        'medication_count': 3,
        'prior_admissions': 1,
        'length_of_stay': 7
    }])

    risk_prediction, risk_probability = predict_readmission_risk(loaded_model, sample_patient)
    print(f"\nSample Patient Data:\n{sample_patient.iloc[0].to_dict()}")
    print(f"Readmission Prediction (0=No, 1=Yes): {risk_prediction}")
    print(f"Readmission Probability: {risk_probability:.4f}")

    # Another sample patient with higher risk factors
    sample_patient_high_risk = pd.DataFrame([{
        'age': 80,
        'gender': 'Male',
        'num_diagnoses': 8,
        'num_procedures': 3,
        'lab_result_a': 90.2,
        'lab_result_b': 45.0,
        'medication_count': 7,
        'prior_admissions': 2,
        'length_of_stay': 15
    }])
    risk_prediction_high, risk_probability_high = predict_readmission_risk(loaded_model, sample_patient_high_risk)
    print(f"\nHigh Risk Sample Patient Data:\n{sample_patient_high_risk.iloc[0].to_dict()}")
    print(f"Readmission Prediction (0=No, 1=Yes): {risk_prediction_high}")
    print(f"Readmission Probability: {risk_probability_high:.4f}")