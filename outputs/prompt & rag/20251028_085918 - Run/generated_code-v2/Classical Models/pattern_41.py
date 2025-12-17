import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import numpy as np

def create_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 90, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples),
        'num_diagnoses': np.random.randint(1, 10, num_samples),
        'num_medications': np.random.randint(5, 30, num_samples),
        'length_of_stay': np.random.randint(1, 20, num_samples),
        'lab_results_avg': np.random.rand(num_samples) * 100,
        'prior_admissions': np.random.randint(0, 5, num_samples),
        'medical_specialty': np.random.choice(['Cardiology', 'Internal Medicine', 'Surgery', 'Pediatrics', 'Oncology'], num_samples),
        'readmitted': np.random.randint(0, 2, num_samples) # 0 for No, 1 for Yes
    }
    # Introduce some missing values for demonstration
    for col in ['age', 'num_medications', 'lab_results_avg']:
        missing_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
        data[col][missing_indices] = np.nan

    return pd.DataFrame(data)

def train_and_evaluate_model(df):
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define categorical and numerical features
    categorical_features = ['gender', 'medical_specialty']
    numerical_features = ['age', 'num_diagnoses', 'num_medications', 'length_of_stay', 'lab_results_avg', 'prior_admissions']

    # Preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the full pipeline with preprocessor and RandomForestClassifier
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\nModel Evaluation on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Perform cross-validation
    cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='roc_auc')
    print(f"\nCross-validation ROC-AUC scores: {cv_scores}")
    print(f"Mean CV ROC-AUC: {np.mean(cv_scores):.4f}")

    return model_pipeline

def predict_readmission(model_path, new_patient_data):
    # Load the trained pipeline
    pipeline = joblib.load(model_path)

    # Convert new patient data to a DataFrame, ensuring consistent column order
    new_df = pd.DataFrame([new_patient_data])

    # Make prediction
    prediction = pipeline.predict(new_df)
    prediction_proba = pipeline.predict_proba(new_df)[:, 1]

    return prediction[0], prediction_proba[0]

if __name__ == "__main__":
    print("Generating synthetic patient data...")
    synthetic_df = create_synthetic_data(num_samples=2000)
    # print(synthetic_df.head())
    # print(synthetic_df.info())

    print("\nTraining and evaluating the readmission prediction model...")
    trained_pipeline = train_and_evaluate_model(synthetic_df)

    # Save the trained model pipeline
    model_filename = 'readmission_prediction_pipeline.joblib'
    joblib.dump(trained_pipeline, model_filename)
    print(f"\nModel pipeline saved to {model_filename}")

    # Demonstrate prediction with new data
    print("\nDemonstrating prediction with new patient data...")
    example_new_patient = {
        'age': 75,
        'gender': 'Female',
        'num_diagnoses': 8,
        'num_medications': 25,
        'length_of_stay': 15,
        'lab_results_avg': 88.5,
        'prior_admissions': 3,
        'medical_specialty': 'Cardiology'
    }
    
    # Another example with potentially lower risk
    example_low_risk_patient = {
        'age': 30,
        'gender': 'Male',
        'num_diagnoses': 1,
        'num_medications': 5,
        'length_of_stay': 2,
        'lab_results_avg': 30.2,
        'prior_admissions': 0,
        'medical_specialty': 'Pediatrics'
    }

    pred, proba = predict_readmission(model_filename, example_new_patient)
    print(f"New Patient (High Risk Example): Prediction = {'Readmitted' if pred == 1 else 'Not Readmitted'} (Probability of Readmission: {proba:.4f})")
    
    pred_low, proba_low = predict_readmission(model_filename, example_low_risk_patient)
    print(f"New Patient (Low Risk Example): Prediction = {'Readmitted' if pred_low == 1 else 'Not Readmitted'} (Probability of Readmission: {proba_low:.4f})")
