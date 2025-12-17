import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    # Simulate target variable if not present for demonstration
    if 'readmitted' not in data.columns:
        np.random.seed(42)
        data['readmitted'] = np.random.randint(0, 2, size=len(data))

    # Handle missing values (simple imputation for demonstration)
    for col in data.select_dtypes(include=np.number).columns:
        data[col] = data[col].fillna(data[col].median())
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Feature Engineering (example: age groups, length of stay - if 'admission_date' and 'discharge_date' exist)
    # Assuming 'age' column for age groups and 'time_in_hospital' for length of stay
    if 'age' in data.columns:
        data['age_group'] = pd.cut(data['age'], bins=[0, 18, 45, 65, 100], labels=['child', 'adult', 'senior', 'elderly'])
    if 'time_in_hospital' in data.columns:
        data['long_stay'] = (data['time_in_hospital'] > 7).astype(int)

    # Separate target variable
    X = data.drop('readmitted', axis=1)
    y = data['readmitted']

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include='object').columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(X[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)

    # Scale numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X[numerical_features])
    scaled_df = pd.DataFrame(scaled_features, columns=numerical_features, index=X.index)

    # Combine preprocessed features
    X_processed = pd.concat([scaled_df, encoded_df], axis=1)

    return X_processed, y, encoder, scaler

def train_models(X_train, y_train):
    models = {
        "LogisticRegression": LogisticRegression(solver='liblinear', random_state=42),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    param_grids = {
        "LogisticRegression": {'C': [0.1, 1, 10]},
        "RandomForestClassifier": {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
        "XGBClassifier": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    }

    trained_models = {}
    best_model = None
    best_score = -1
    best_model_name = ""

    for name, model in models.items():
        print(f"Training {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        trained_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best ROC AUC for {name}: {grid_search.best_score_:.4f}")

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_model_name = name

    print(f"\nBest overall model: {best_model_name} with ROC AUC: {best_score:.4f}")
    return best_model, best_model_name

def evaluate_models(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- Evaluation for {model_name} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def predict_readmission(model, new_data_df, encoder, scaler, original_columns):
    # Ensure new data has same columns as training data before preprocessing
    # Create a dummy DataFrame with all original columns and fill new_data_df into it
    # This handles cases where new_data_df might be missing columns present in training
    processed_new_data = pd.DataFrame(columns=original_columns)
    for col in original_columns:
        if col in new_data_df.columns:
            processed_new_data[col] = new_data_df[col]
        else:
            # Fill with a default/mode value, or nan for numerical and impute later
            if col in scaler.feature_names_in_:
                processed_new_data[col] = np.nan # Will be imputed by median in preprocessing
            elif col in encoder.feature_names_in_:
                 processed_new_data[col] = 'unknown' # Or mode of training data
            else:
                processed_new_data[col] = np.nan # For other engineered features later
    
    # Apply same preprocessing steps
    categorical_features_new = processed_new_data.select_dtypes(include='object').columns
    numerical_features_new = processed_new_data.select_dtypes(include=np.number).columns

    # Handle missing values in new data for numerical columns (median from training)
    for col in numerical_features_new:
        if col in scaler.feature_names_in_:
            median_val = pd.Series(scaler.mean_, index=scaler.feature_names_in_)[col] # Using mean as proxy for median after scale fit
            processed_new_data[col] = processed_new_data[col].fillna(median_val)
        else:
             processed_new_data[col] = processed_new_data[col].fillna(processed_new_data[col].median())

    # Feature Engineering for new data (if applicable)
    if 'age' in processed_new_data.columns:
        processed_new_data['age_group'] = pd.cut(processed_new_data['age'], bins=[0, 18, 45, 65, 100], labels=['child', 'adult', 'senior', 'elderly'])
    if 'time_in_hospital' in processed_new_data.columns:
        processed_new_data['long_stay'] = (processed_new_data['time_in_hospital'] > 7).astype(int)

    encoded_features_new = encoder.transform(processed_new_data[categorical_features_new])
    encoded_feature_names_new = encoder.get_feature_names_out(categorical_features_new)
    encoded_df_new = pd.DataFrame(encoded_features_new, columns=encoded_feature_names_new, index=new_data_df.index)

    scaled_features_new = scaler.transform(processed_new_data[numerical_features_new])
    scaled_df_new = pd.DataFrame(scaled_features_new, columns=numerical_features_new, index=new_data_df.index)

    X_new_processed = pd.concat([scaled_df_new, encoded_df_new], axis=1)

    prediction = model.predict(X_new_processed)
    prediction_proba = model.predict_proba(X_new_processed)[:, 1]
    return prediction, prediction_proba

if __name__ == "__main__":
    # Create dummy data for demonstration
    np.random.seed(42)
    data = {
        'patient_id': range(100),
        'age': np.random.randint(20, 90, 100),
        'gender': np.random.choice(['Male', 'Female'], 100),
        'ethnicity': np.random.choice(['Caucasian', 'AfricanAmerican', 'Asian', 'Other'], 100),
        'time_in_hospital': np.random.randint(1, 15, 100),
        'num_lab_procedures': np.random.randint(10, 80, 100),
        'num_medications': np.random.randint(5, 30, 100),
        'diagnosis': np.random.choice(['Diabetes', 'HeartDisease', 'Pneumonia', 'Flu', 'Other'], 100),
        'num_diagnoses': np.random.randint(1, 10, 100),
        'discharge_disposition': np.random.choice(['Home', 'Rehab', 'Expired'], 100),
        'readmitted': np.random.randint(0, 2, 100) # 0 for no, 1 for yes
    }
    dummy_df = pd.DataFrame(data)
    dummy_df.to_csv("dummy_patient_data.csv", index=False)
    print("Dummy data generated to dummy_patient_data.csv")

    # 1. Data Ingestion and Preprocessing
    filepath = "dummy_patient_data.csv"
    X_processed, y, encoder, scaler = load_and_preprocess_data(filepath)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    
    # Store original columns for prediction function
    original_data_for_cols = pd.read_csv(filepath)
    original_columns_for_prediction = original_data_for_cols.drop('readmitted', axis=1).columns

    print("\nData preprocessing complete.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # 2. Model Training
    best_model, best_model_name = train_models(X_train, y_train)

    # 3. Model Evaluation
    evaluate_models(best_model, X_test, y_test, best_model_name)

    # 4. Model Persistence
    model_filename = f"{best_model_name.lower().replace(' ', '_')}_readmission_predictor.joblib"
    joblib.dump(best_model, model_filename)
    joblib.dump(encoder, 'onehot_encoder.joblib')
    joblib.dump(scaler, 'standard_scaler.joblib')
    joblib.dump(original_columns_for_prediction, 'original_columns.joblib')
    print(f"\nBest model, encoder, scaler, and original columns saved to {model_filename}, onehot_encoder.joblib, standard_scaler.joblib, and original_columns.joblib")

    # Demonstrate Prediction on new data
    print("\n--- Demonstrating Prediction on New Data ---")
    loaded_model = joblib.load(model_filename)
    loaded_encoder = joblib.load('onehot_encoder.joblib')
    loaded_scaler = joblib.load('standard_scaler.joblib')
    loaded_original_columns = joblib.load('original_columns.joblib')

    new_patient_data = pd.DataFrame([
        {
            'patient_id': 101,
            'age': 75,
            'gender': 'Female',
            'ethnicity': 'Caucasian',
            'time_in_hospital': 10,
            'num_lab_procedures': 60,
            'num_medications': 25,
            'diagnosis': 'HeartDisease',
            'num_diagnoses': 7,
            'discharge_disposition': 'Home'
        },
        {
            'patient_id': 102,
            'age': 30,
            'gender': 'Male',
            'ethnicity': 'Asian',
            'time_in_hospital': 3,
            'num_lab_procedures': 20,
            'num_medications': 8,
            'diagnosis': 'Flu',
            'num_diagnoses': 2,
            'discharge_disposition': 'Rehab'
        }
    ])

    predictions, probabilities = predict_readmission(loaded_model, new_patient_data, loaded_encoder, loaded_scaler, loaded_original_columns)
    new_patient_data['predicted_readmission'] = predictions
    new_patient_data['readmission_probability'] = probabilities
    print(new_patient_data[['patient_id', 'predicted_readmission', 'readmission_probability']])
