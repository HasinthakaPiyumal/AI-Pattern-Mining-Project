import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import joblib


class DiseasePredictionSystem:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.features = []
        self.target = ''

    def simulate_data(self, n_samples=1000):
        np.random.seed(42)
        data = {
            'age': np.random.randint(20, 70, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'cholesterol': np.random.randint(150, 250, n_samples),
            'blood_pressure_systolic': np.random.randint(100, 160, n_samples),
            'blood_pressure_diastolic': np.random.randint(60, 100, n_samples),
            'bmi': np.random.uniform(18.0, 35.0, n_samples),
            'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'diabetes_history': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'disease_risk': np.random.randint(0, 2, n_samples)  # 0: Low Risk, 1: High Risk
        }
        df = pd.DataFrame(data)

        # Introduce some correlation for 'disease_risk'
        df.loc[(df['age'] > 50) & (df['cholesterol'] > 200) & (df['smoking'] == 1), 'disease_risk'] = 1
        df.loc[(df['bmi'] > 30) & (df['diabetes_history'] == 1), 'disease_risk'] = 1
        df.loc[(df['age'] < 30) & (df['smoking'] == 0) & (df['cholesterol'] < 180), 'disease_risk'] = 0

        return df

    def preprocess_data(self, df, target_column='disease_risk'):
        self.target = target_column
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Identify numerical and categorical features
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include='object').columns.tolist()
        self.features = numerical_features + categorical_features # Store feature names for prediction

        # Scale numerical features
        self.scaler = StandardScaler()
        X[numerical_features] = self.scaler.fit_transform(X[numerical_features])

        # One-hot encode categorical features
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_features = self.encoder.fit_transform(X[categorical_features])
        encoded_feature_names = self.encoder.get_feature_names_out(categorical_features)
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)

        # Combine preprocessed features
        X_processed = pd.concat([X[numerical_features], encoded_df], axis=1)

        return X_processed, y

    def train_model(self, X_train, y_train, model_type='RandomForest'):
        if model_type == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(random_state=42, solver='liblinear')
        elif model_type == 'SVC':
            from sklearn.svm import SVC
            self.model = SVC(probability=True, random_state=42)
        elif model_type == 'DecisionTree':
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier(random_state=42)
        else:
            raise ValueError("Unsupported model type. Choose from 'RandomForest', 'LogisticRegression', 'SVC', 'DecisionTree'.")

        self.model.fit(X_train, y_train)
        print(f"{model_type} model trained successfully.")

    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print(f"\n--- Model Evaluation ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    def predict(self, new_data):
        if self.model is None or self.scaler is None or self.encoder is None:
            raise ValueError("Model and preprocessors have not been initialized. Please train the model first.")

        # Ensure new_data is a DataFrame and has the same columns as the training data
        if not isinstance(new_data, pd.DataFrame):
            new_data = pd.DataFrame([new_data])

        # Separate numerical and categorical features
        numerical_features = new_data.select_dtypes(include=np.number).columns.tolist()
        categorical_features = new_data.select_dtypes(include='object').columns.tolist()

        # Scale numerical features
        new_data[numerical_features] = self.scaler.transform(new_data[numerical_features])

        # One-hot encode categorical features
        encoded_features = self.encoder.transform(new_data[categorical_features])
        encoded_feature_names = self.encoder.get_feature_names_out(categorical_features)
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=new_data.index)

        # Combine preprocessed features, ensuring order matches training data
        X_processed_new = pd.concat([new_data[numerical_features], encoded_df], axis=1)

        # Reindex to ensure all columns present during training are present in new data
        # and fill missing (if any new category appears, it will be 0) and remove extra columns
        missing_cols = set(self.model.feature_names_in_) - set(X_processed_new.columns)
        for c in missing_cols:
            X_processed_new[c] = 0
        X_processed_new = X_processed_new[self.model.feature_names_in_]

        prediction = self.model.predict(X_processed_new)
        probability = self.model.predict_proba(X_processed_new)[:, 1]

        return {"prediction": int(prediction[0]), "probability": float(probability[0])}

    def save_model(self, filename="disease_prediction_model.joblib"):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'encoder': self.encoder,
            'features': self.features
        }, filename)
        print(f"Model, scaler, and encoder saved to {filename}")

    def load_model(self, filename="disease_prediction_model.joblib"):
        loaded_components = joblib.load(filename)
        self.model = loaded_components['model']
        self.scaler = loaded_components['scaler']
        self.encoder = loaded_components['encoder']
        self.features = loaded_components['features']
        print(f"Model, scaler, and encoder loaded from {filename}")


if __name__ == "__main__":
    system = DiseasePredictionSystem()

    # 1. Simulate Data
    print("Simulating data...")
    raw_df = system.simulate_data(n_samples=1000)
    print(f"Raw data head:\n{raw_df.head()}")

    # 2. Preprocess Data
    print("\nPreprocessing data...")
    X, y = system.preprocess_data(raw_df)
    print(f"Processed X head:\n{X.head()}")
    print(f"y head:\n{y.head()}")

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining with {len(X_train)} samples, testing with {len(X_test)} samples.")

    # 3. Train Model
    print("\nTraining RandomForestClassifier...")
    system.train_model(X_train, y_train, model_type='RandomForest')

    # 4. Evaluate Model
    system.evaluate_model(X_test, y_test)

    # 5. Save Model
    system.save_model()

    # 6. Load Model and Make a Prediction
    print("\nLoading model for prediction...")
    new_system = DiseasePredictionSystem()
    new_system.load_model()

    # Example new patient data
    new_patient_data = {
        'age': 60,
        'gender': 'Female',
        'cholesterol': 230,
        'blood_pressure_systolic': 150,
        'blood_pressure_diastolic': 95,
        'bmi': 32.5,
        'smoking': 1,
        'diabetes_history': 1
    }
    print(f"\nPredicting for new patient: {new_patient_data}")
    prediction_result = new_system.predict(new_patient_data)
    print(f"Prediction: {prediction_result['prediction']} (0: Low Risk, 1: High Risk)")
    print(f"Probability of High Risk: {prediction_result['probability']:.4f}")

    # Another example with low risk profile
    new_patient_data_low_risk = {
        'age': 25,
        'gender': 'Male',
        'cholesterol': 170,
        'blood_pressure_systolic': 110,
        'blood_pressure_diastolic': 70,
        'bmi': 22.0,
        'smoking': 0,
        'diabetes_history': 0
    }
    print(f"\nPredicting for new patient (low risk profile): {new_patient_data_low_risk}")
    prediction_result_low_risk = new_system.predict(new_patient_data_low_risk)
    print(f"Prediction: {prediction_result_low_risk['prediction']} (0: Low Risk, 1: High Risk)")
    print(f"Probability of High Risk: {prediction_result_low_risk['probability']:.4f}")