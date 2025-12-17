import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib

class DiabetesDiagnosticSystem:
    def __init__(self):
        self.scaler = None
        self.model = None

    def generate_synthetic_data(self, num_samples=1000):
        np.random.seed(42)
        data = {
            'Age': np.random.randint(20, 70, num_samples),
            'BMI': np.random.uniform(18.0, 40.0, num_samples),
            'BloodGlucose': np.random.uniform(70, 200, num_samples),
            'BloodPressure': np.random.uniform(80, 180, num_samples),
            'Insulin': np.random.uniform(0, 300, num_samples),
            'Pregnancies': np.random.randint(0, 10, num_samples),
            'SkinThickness': np.random.uniform(10, 60, num_samples),
            'DiabetesPedigreeFunction': np.random.uniform(0.07, 2.5, num_samples),
        }
        df = pd.DataFrame(data)

        # Simulate diabetes outcome based on a combination of features
        df['Outcome'] = ((df['BloodGlucose'] > 140) * 0.4 + 
                         (df['BMI'] > 30) * 0.3 + 
                         (df['Age'] > 50) * 0.2 + 
                         (df['Insulin'] > 150) * 0.1 > 0.5).astype(int)
        
        # Introduce some noise and balance
        num_diabetes = df['Outcome'].sum()
        if num_diabetes < num_samples / 3:
            # Increase diabetes cases slightly for better training
            high_risk_indices = df[(df['BloodGlucose'] > 120) & (df['BMI'] > 28)].sample(frac=0.3, random_state=42).index
            df.loc[high_risk_indices, 'Outcome'] = 1
        
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df

    def preprocess_data(self, df, target_column='Outcome'):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle missing values (simple imputation for demonstration)
        X = X.fillna(X.mean())

        # Feature Scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # Data Splitting
        X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train, model_type='logistic_regression'):
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, solver='liblinear')
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError("Invalid model_type. Choose 'logistic_regression' or 'random_forest'.")

        self.model.fit(X_train, y_train)
        print(f"{model_type.replace('_', ' ').title()} model trained successfully.")

    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print("\n--- Model Evaluation ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

    def save_model_and_scaler(self, model_filepath='diabetes_model.joblib', scaler_filepath='scaler.joblib'):
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not trained/fitted yet.")
        joblib.dump(self.model, model_filepath)
        joblib.dump(self.scaler, scaler_filepath)
        print(f"Model saved to {model_filepath}")
        print(f"Scaler saved to {scaler_filepath}")

    def load_model_and_scaler(self, model_filepath='diabetes_model.joblib', scaler_filepath='scaler.joblib'):
        self.model = joblib.load(model_filepath)
        self.scaler = joblib.load(scaler_filepath)
        print(f"Model loaded from {model_filepath}")
        print(f"Scaler loaded from {scaler_filepath}")

    def predict_diabetes(self, new_patient_data):
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not loaded/trained yet.")
        
        # Ensure new_patient_data is a DataFrame with correct columns
        if isinstance(new_patient_data, dict):
            new_patient_data = pd.DataFrame([new_patient_data])
        elif not isinstance(new_patient_data, pd.DataFrame):
            raise ValueError("new_patient_data must be a dictionary or pandas DataFrame.")

        # The column order must be consistent with training data
        # For demonstration, assume input dict/DataFrame has correct columns
        # In a real system, you'd ensure column order matches X.columns from preprocessing
        
        new_patient_data_scaled = self.scaler.transform(new_patient_data)
        prediction = self.model.predict(new_patient_data_scaled)[0]
        prediction_proba = self.model.predict_proba(new_patient_data_scaled)[0, 1]
        
        return {
            'prediction': 'Diabetes' if prediction == 1 else 'No Diabetes',
            'probability_diabetes': prediction_proba
        }

if __name__ == '__main__':
    system = DiabetesDiagnosticSystem()

    print("\n--- Generating Synthetic Data ---")
    df = system.generate_synthetic_data(num_samples=2000)
    print("Synthetic data generated. First 5 rows:")
    print(df.head())
    print(f"Dataset shape: {df.shape}")
    print(f"Diabetes cases: {df['Outcome'].sum()}, No Diabetes cases: {(df.shape[0] - df['Outcome'].sum())}")

    print("\n--- Preprocessing Data ---")
    X_train, X_test, y_train, y_test = system.preprocess_data(df)
    print(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    print("\n--- Training Logistic Regression Model ---")
    system.train_model(X_train, y_train, model_type='logistic_regression')
    system.evaluate_model(X_test, y_test)
    system.save_model_and_scaler('logistic_regression_model.joblib', 'scaler.joblib')

    print("\n--- Training Random Forest Model ---")
    system.train_model(X_train, y_train, model_type='random_forest')
    system.evaluate_model(X_test, y_test)
    system.save_model_and_scaler('random_forest_model.joblib', 'scaler.joblib')

    print("\n--- Demonstrating Prediction with Loaded Model ---")
    # Load the Random Forest model for prediction
    prediction_system = DiabetesDiagnosticSystem()
    prediction_system.load_model_and_scaler('random_forest_model.joblib', 'scaler.joblib')

    # Example new patient data
    new_patient = {
        'Age': 55,
        'BMI': 32.5,
        'BloodGlucose': 160,
        'BloodPressure': 130,
        'Insulin': 200,
        'Pregnancies': 2,
        'SkinThickness': 35,
        'DiabetesPedigreeFunction': 0.85
    }
    
    prediction_result = prediction_system.predict_diabetes(new_patient)
    print(f"\nNew Patient Data: {new_patient}")
    print(f"Prediction: {prediction_result['prediction']} (Probability of Diabetes: {prediction_result['probability_diabetes']:.4f})")

    new_patient_low_risk = {
        'Age': 30,
        'BMI': 22.0,
        'BloodGlucose': 90,
        'BloodPressure': 110,
        'Insulin': 50,
        'Pregnancies': 0,
        'SkinThickness': 20,
        'DiabetesPedigreeFunction': 0.15
    }
    prediction_result_low_risk = prediction_system.predict_diabetes(new_patient_low_risk)
    print(f"\nNew Patient Data: {new_patient_low_risk}")
    print(f"Prediction: {prediction_result_low_risk['prediction']} (Probability of Diabetes: {prediction_result_low_risk['probability_diabetes']:.4f})")
