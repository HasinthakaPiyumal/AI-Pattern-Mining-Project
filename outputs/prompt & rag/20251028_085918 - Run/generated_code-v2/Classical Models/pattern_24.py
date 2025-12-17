import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# --- Data Module (data_loader.py content) ---
def load_data(filepath=None):
    if filepath:
        df = pd.read_csv(filepath)
    else:
        # Simulate a dataset for demonstration if no filepath is provided
        np.random.seed(42)
        num_samples = 1000
        data = {
            'age': np.random.randint(25, 80, num_samples),
            'gender': np.random.choice(['Male', 'Female'], num_samples),
            'cholesterol': np.random.randint(150, 300, num_samples),
            'blood_pressure': np.random.randint(90, 180, num_samples),
            'smoking': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
            'diabetes': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
            'obesity': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
            'cardio_disease': np.zeros(num_samples, dtype=int)
        }
        df = pd.DataFrame(data)

        df.loc[(df['age'] > 60) & (df['cholesterol'] > 220) & (df['blood_pressure'] > 140), 'cardio_disease'] = 1
        df.loc[(df['smoking'] == 1) & (df['age'] > 50), 'cardio_disease'] = 1
        df.loc[(df['diabetes'] == 1) & (df['obesity'] == 1), 'cardio_disease'] = 1
        df.loc[df['cardio_disease'] == 0, 'cardio_disease'] = np.random.choice([0, 1], sum(df['cardio_disease'] == 0), p=[0.9, 0.1])
        df.loc[df['cardio_disease'] == 1, 'cardio_disease'] = np.random.choice([0, 1], sum(df['cardio_disease'] == 1), p=[0.1, 0.9])

    print(f"Data loaded. Shape: {df.shape}")
    return df

# --- Preprocessing Module (preprocessor.py content) ---
class DataPreprocessor:
    def __init__(self):
        self.numerical_features = ['age', 'cholesterol', 'blood_pressure']
        self.categorical_features = ['gender', 'smoking', 'diabetes', 'obesity']
        self.preprocessor = None

    def fit(self, X, y=None):
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        self.preprocessor.fit(X)
        print("Data preprocessor fitted.")
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        X_processed = self.preprocessor.transform(X)
        print("Data transformed.")
        return X_processed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

# --- Model Training Module (model_trainer.py content) ---
def train_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    model = LogisticRegression(solver='liblinear', random_state=random_state)
    model.fit(X_train, y_train)
    print("Model training complete.")

    metrics = evaluate_model(model, X_test, y_test)
    print("Model evaluation complete.")
    return model, metrics, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    print(f"Model Metrics: {metrics}")
    return metrics

def save_model_artifacts(model, preprocessor, model_path="logistic_regression_model.joblib", preprocessor_path="data_preprocessor.joblib"):
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {preprocessor_path}")

# --- Prediction Module (predictor.py content) ---
def make_prediction(raw_data_dict, model_path="logistic_regression_model.joblib", preprocessor_path="data_preprocessor.joblib"):
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
    except FileNotFoundError:
        raise FileNotFoundError("Model or preprocessor file not found. Ensure training was performed and artifacts were saved.")

    raw_data_df = pd.DataFrame([raw_data_dict])
    
    processed_data = preprocessor.transform(raw_data_df)

    prediction = model.predict(processed_data)[0]
    prediction_proba = model.predict_proba(processed_data)[:, 1][0]

    print(f"Prediction: {'Cardiovascular Disease' if prediction == 1 else 'No Cardiovascular Disease'}")
    print(f"Probability of Cardiovascular Disease: {prediction_proba:.4f}")
    return prediction, prediction_proba

# --- Main Application Script (main.py content) ---
def main():
    print("\n--- Starting Predictive Diagnosis System ---")

    df = load_data()
    
    X = df.drop('cardio_disease', axis=1)
    y = df['cardio_disease']

    preprocessor = DataPreprocessor()
    X_processed = preprocessor.fit_transform(X)

    model, metrics, X_test, y_test = train_model(X_processed, y)
    print(f"\nTraining Metrics: {metrics}")

    model_path = "logistic_regression_model.joblib"
    preprocessor_path = "data_preprocessor.joblib"
    save_model_artifacts(model, preprocessor, model_path, preprocessor_path)

    print("\n--- Demonstrating Prediction ---")
    new_patient_data = {
        'age': 65,
        'gender': 'Female',
        'cholesterol': 280,
        'blood_pressure': 155,
        'smoking': 1,
        'diabetes': 0,
        'obesity': 1
    }
    print(f"New Patient Data: {new_patient_data}")
    prediction, probability = make_prediction(new_patient_data, model_path, preprocessor_path)

    new_patient_data_2 = {
        'age': 30,
        'gender': 'Male',
        'cholesterol': 180,
        'blood_pressure': 110,
        'smoking': 0,
        'diabetes': 0,
        'obesity': 0
    }
    print(f"\nNew Patient Data: {new_patient_data_2}")
    prediction_2, probability_2 = make_prediction(new_patient_data_2, model_path, preprocessor_path)

    print("\n--- System Finished ---")

if __name__ == "__main__":
    main()