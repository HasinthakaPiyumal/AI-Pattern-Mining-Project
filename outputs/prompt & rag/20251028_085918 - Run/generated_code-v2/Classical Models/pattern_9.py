import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import xgboost as xgb

# --- 1. Data Ingestion & Preprocessing Module ---

def load_data(filepath):
    return pd.read_csv(filepath)

def handle_missing_values(df):
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def feature_engineering(df):
    df['age_band'] = pd.cut(df['Age'], bins=[0, 18, 45, 65, 100], labels=['Child', 'Adult', 'Senior', 'Elderly'], right=False)
    df['num_diagnoses_per_stay'] = df['Diagnosis'].map(df['Diagnosis'].value_counts())
    return df

def preprocess_features(df, training=True, preprocessors=None):
    if preprocessors is None:
        preprocessors = {
            'scaler': StandardScaler(),
            'encoder': OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        }

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Exclude target variable if it's in numerical_cols
    if 'Readmitted' in numerical_cols:
        numerical_cols.remove('Readmitted')
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if training:
        df_scaled = preprocessors['scaler'].fit_transform(df[numerical_cols])
        df_encoded = preprocessors['encoder'].fit_transform(df[categorical_cols])
    else:
        df_scaled = preprocessors['scaler'].transform(df[numerical_cols])
        df_encoded = preprocessors['encoder'].transform(df[categorical_cols])

    df_scaled = pd.DataFrame(df_scaled, columns=numerical_cols, index=df.index)
    df_encoded = pd.DataFrame(df_encoded, columns=preprocessors['encoder'].get_feature_names_out(categorical_cols), index=df.index)

    df_processed = pd.concat([df_scaled, df_encoded], axis=1)
    return df_processed, preprocessors

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# --- 2. Model Training Module ---

def train_model(X_train, y_train, model_name='LogisticRegression', hyperparameters={}):
    if model_name == 'LogisticRegression':
        model = LogisticRegression(random_state=42, solver='liblinear', **hyperparameters)
    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(random_state=42, **hyperparameters)
    elif model_name == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(random_state=42, **hyperparameters)
    elif model_name == 'XGBClassifier':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **hyperparameters)
    elif model_name == 'SVC':
        model = SVC(probability=True, random_state=42, **hyperparameters)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")
    
    model.fit(X_train, y_train)
    return model

def tune_hyperparameters(model, X_train, y_train, param_grid, cv=3):
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best ROC AUC score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# --- 3. Model Evaluation Module ---

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- 4. Prediction Module ---

def load_predictor(model_filename, preprocessors_filename):
    model = joblib.load(model_filename)
    preprocessors = joblib.load(preprocessors_filename)
    return model, preprocessors

def predict_readmission(model, preprocessors, new_data):
    # Ensure new_data is a DataFrame with the same columns as training data before preprocessing
    # For simplicity, assume new_data already has expected columns, just needs preprocessing
    
    # The columns must match the order and names used during training preprocessing
    # This part needs careful handling to match the columns after OHE and scaling
    
    # Create a DataFrame from new_data (if it's a Series or dict)
    if isinstance(new_data, dict):
        new_data_df = pd.DataFrame([new_data])
    elif isinstance(new_data, pd.Series):
        new_data_df = pd.DataFrame([new_data.to_dict()])
    else:
        new_data_df = new_data.copy()

    # Apply feature engineering to new data
    new_data_df = feature_engineering(new_data_df)
    
    processed_new_data, _ = preprocess_features(new_data_df, training=False, preprocessors=preprocessors)
    
    prediction_proba = model.predict_proba(processed_new_data)[:, 1]
    prediction_class = model.predict(processed_new_data)
    
    return prediction_class[0], prediction_proba[0]

# --- 5. Main Orchestration Script ---

if __name__ == "__main__":
    # Generate synthetic data for demonstration
    np.random.seed(42)
    data_size = 1000
    synthetic_data = {
        'PatientID': [f'P{i:04d}' for i in range(data_size)],
        'Age': np.random.randint(18, 90, data_size),
        'Gender': np.random.choice(['Male', 'Female'], data_size),
        'Diagnosis': np.random.choice(['Pneumonia', 'Heart Failure', 'Diabetes', 'Stroke', 'Appendicitis'], data_size),
        'Num_Previous_Admissions': np.random.randint(0, 5, data_size),
        'Length_of_Stay_Days': np.random.randint(1, 30, data_size),
        'Lab_Result_Creatinine': np.random.uniform(0.5, 2.0, data_size),
        'Medication': np.random.choice(['DrugA', 'DrugB', 'DrugC', 'DrugD'], data_size),
        'Readmitted': np.random.choice([0, 1], data_size, p=[0.7, 0.3]) # 30% readmission rate
    }
    df = pd.DataFrame(synthetic_data)
    df.to_csv('synthetic_ehr_data.csv', index=False)

    print("--- Starting Patient Readmission Prediction System ---")
    
    # 1. Data Ingestion & Preprocessing
    print("\n--- Data Ingestion & Preprocessing ---")
    df_raw = load_data('synthetic_ehr_data.csv')
    print("Raw Data Head:\n", df_raw.head())

    df_handled_missing = handle_missing_values(df_raw.copy())
    df_engineered = feature_engineering(df_handled_missing.copy())
    
    X = df_engineered.drop(['PatientID', 'Readmitted'], axis=1)
    y = df_engineered['Readmitted']

    X_processed, preprocessors = preprocess_features(X, training=True)
    X_train, X_test, y_train, y_test = split_data(X_processed, y)
    
    joblib.dump(preprocessors, 'preprocessors.joblib')
    print("Data preprocessing complete. Preprocessors saved.")

    # 2. Model Training
    print("\n--- Model Training (XGBoost) ---")
    # Example: Train an XGBoost Classifier
    xgb_model = train_model(X_train, y_train, model_name='XGBClassifier')
    save_model(xgb_model, 'xgb_readmission_model.joblib')
    
    # Example: Hyperparameter Tuning (for Logistic Regression as a quicker example)
    print("\n--- Hyperparameter Tuning (Logistic Regression) ---")
    lr_model = LogisticRegression(random_state=42, solver='liblinear')
    lr_param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    best_lr_model = tune_hyperparameters(lr_model, X_train, y_train, lr_param_grid)
    save_model(best_lr_model, 'tuned_lr_readmission_model.joblib')

    # 3. Model Evaluation
    print("\n--- Model Evaluation (XGBoost) ---")
    evaluate_model(xgb_model, X_test, y_test)
    
    print("\n--- Model Evaluation (Tuned Logistic Regression) ---")
    evaluate_model(best_lr_model, X_test, y_test)

    # 4. Prediction
    print("\n--- Prediction on New Data ---")
    loaded_model, loaded_preprocessors = load_predictor('xgb_readmission_model.joblib', 'preprocessors.joblib')

    # Simulate a new patient's data
    new_patient_data = pd.Series({
        'Age': 70,
        'Gender': 'Female',
        'Diagnosis': 'Heart Failure',
        'Num_Previous_Admissions': 2,
        'Length_of_Stay_Days': 10,
        'Lab_Result_Creatinine': 1.8,
        'Medication': 'DrugB'
    })

    predicted_class, predicted_proba = predict_readmission(loaded_model, loaded_preprocessors, new_patient_data)
    print(f"New Patient Prediction: {'Readmitted' if predicted_class == 1 else 'Not Readmitted'} (Probability: {predicted_proba:.4f})")

    print("\n--- Patient Readmission Prediction System Finished ---")