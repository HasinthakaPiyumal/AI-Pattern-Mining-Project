import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib
import warnings
warnings.filterwarnings("ignore")

class ChronicDiseasePredictor:
    def __init__(self, model_type="RandomForestClassifier"):
        self.model_type = model_type
        self.preprocessor = None
        self.model = None
        self.categorical_features = None
        self.numerical_features = None

    def _create_mock_data(self, num_samples=1000):
        np.random.seed(42)
        data = {
            'Age': np.random.randint(25, 80, num_samples),
            'Gender': np.random.choice(['Male', 'Female'], num_samples),
            'BMI': np.random.uniform(18.0, 40.0, num_samples),
            'BloodPressure_Systolic': np.random.randint(100, 180, num_samples),
            'BloodPressure_Diastolic': np.random.randint(60, 110, num_samples),
            'Cholesterol': np.random.randint(150, 280, num_samples),
            'Glucose': np.random.randint(70, 200, num_samples),
            'Smoking': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
            'Alcohol': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
            'ExerciseHoursWeek': np.random.uniform(0.5, 10.0, num_samples),
            'FamilyHistory_Diabetes': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
            'FamilyHistory_HeartDisease': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
            'ChronicDisease': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]) # Target variable
        }
        df = pd.DataFrame(data)
        # Introduce some correlation for ChronicDisease
        df['ChronicDisease'] = np.where(
            (df['Age'] > 50) & (df['BMI'] > 28) & (df['Glucose'] > 120) & (df['FamilyHistory_Diabetes'] == 1),
            1,
            df['ChronicDisease']
        )
        df['ChronicDisease'] = np.where(
            (df['BloodPressure_Systolic'] > 140) & (df['Cholesterol'] > 220) & (df['Smoking'] == 1) & (df['Age'] > 60),
            1,
            df['ChronicDisease']
        )
        return df

    def data_ingestion_and_preprocessing(self, df, target_column='ChronicDisease'):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        self.numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_features = X.select_dtypes(include='object').columns.tolist()

        # Create a preprocessor pipeline for numerical and categorical features
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ], 
            remainder='passthrough'
        )
        
        return X, y

    def feature_engineering(self, X):
        # For this example, feature engineering is primarily handled by the preprocessor
        # and direct selection of relevant columns. Additional feature engineering
        # such as interaction terms can be added here if needed.
        # Example: X['Age_BMI_Interaction'] = X['Age'] * X['BMI']
        return X

    def train_model(self, X_train, y_train):
        # Apply preprocessing and then train the model
        if self.model_type == "LogisticRegression":
            classifier = LogisticRegression(random_state=42)
        elif self.model_type == "SVC":
            classifier = SVC(probability=True, random_state=42)
        elif self.model_type == "DecisionTreeClassifier":
            classifier = DecisionTreeClassifier(random_state=42)
        elif self.model_type == "RandomForestClassifier":
            classifier = RandomForestClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.model = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', classifier)])
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            raise RuntimeError("Model not trained. Please train the model first.")

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"\n--- Model Evaluation ({self.model_type}) ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}

    def save_model(self, model_path="chronic_disease_model.joblib", preprocessor_path="preprocessor.joblib"):
        if self.model is None:
            raise RuntimeError("Model not trained. Cannot save.")
        joblib.dump(self.model, model_path)
        # The preprocessor is part of the pipeline, so saving the model saves the preprocessor too.
        # However, it's good practice to save feature names if needed for external checks.
        joblib.dump({
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }, preprocessor_path)
        print(f"\nModel saved to {model_path}")
        print(f"Feature metadata saved to {preprocessor_path}")

    def load_model(self, model_path="chronic_disease_model.joblib", preprocessor_path="preprocessor.joblib"):
        self.model = joblib.load(model_path)
        feature_metadata = joblib.load(preprocessor_path)
        self.numerical_features = feature_metadata['numerical_features']
        self.categorical_features = feature_metadata['categorical_features']
        print(f"Model loaded from {model_path}")
        print(f"Feature metadata loaded from {preprocessor_path}")

    def predict_likelihood(self, new_patient_data_df):
        if self.model is None:
            raise RuntimeError("Model not loaded or trained. Cannot make predictions.")

        # Ensure new data has the same columns as training data, filling missing ones if necessary
        expected_columns = self.numerical_features + self.categorical_features
        for col in expected_columns:
            if col not in new_patient_data_df.columns:
                new_patient_data_df[col] = np.nan # Or appropriate default/imputation
        new_patient_data_df = new_patient_data_df[expected_columns]
        
        # The pipeline handles preprocessing automatically during predict
        prediction_proba = self.model.predict_proba(new_patient_data_df)[:, 1]
        return prediction_proba

if __name__ == "__main__":
    print("Initializing Chronic Disease Predictor...")
    predictor = ChronicDiseasePredictor(model_type="RandomForestClassifier") # You can change the model type here

    # 1. Create Mock Data
    print("Creating mock EHR data...")
    ehr_df = predictor._create_mock_data(num_samples=2000)
    print(f"Mock data created with {len(ehr_df)} samples.")

    # 2. Data Ingestion and Preprocessing
    print("Preprocessing data and splitting into training/testing sets...")
    X, y = predictor.data_ingestion_and_preprocessing(ehr_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # 3. Feature Engineering (already integrated into preprocessor or direct selection)
    # For this example, X_train and X_test are ready after initial preprocessing.

    # 4. Model Training
    print(f"Training {predictor.model_type} model...")
    predictor.train_model(X_train, y_train)
    print("Model training complete.")

    # 5. Model Evaluation
    eval_results = predictor.evaluate_model(X_test, y_test)

    # 6. Model Persistence
    predictor.save_model()

    # --- Demonstrate Prediction API --- 
    print("\n--- Demonstrating Prediction API ---")
    
    # Load the trained model and preprocessor (simulating a new session)
    new_predictor_instance = ChronicDiseasePredictor()
    new_predictor_instance.load_model()

    # Create some new patient data for prediction
    new_patient_data = pd.DataFrame({
        'Age': [65, 40, 70],
        'Gender': ['Female', 'Male', 'Female'],
        'BMI': [32.5, 24.1, 29.8],
        'BloodPressure_Systolic': [155, 120, 140],
        'BloodPressure_Diastolic': [95, 80, 85],
        'Cholesterol': [240, 180, 210],
        'Glucose': [160, 90, 115],
        'Smoking': [1, 0, 0],
        'Alcohol': [0, 1, 0],
        'ExerciseHoursWeek': [2.0, 7.5, 4.0],
        'FamilyHistory_Diabetes': [1, 0, 1],
        'FamilyHistory_HeartDisease': [1, 0, 0]
    })

    print("New patient data for prediction:")
    print(new_patient_data)

    # Get predictions
    likelihoods = new_predictor_instance.predict_likelihood(new_patient_data)

    print("\nPredicted likelihood of chronic disease (0 to 1):")
    for i, likelihood in enumerate(likelihoods):
        print(f"Patient {i+1}: {likelihood:.4f}")

    print("\nPrediction demonstration complete.")

    # Example with a different model type
    print("\n--- Demonstrating with Logistic Regression Model ---")
    lr_predictor = ChronicDiseasePredictor(model_type="LogisticRegression")
    X_lr, y_lr = lr_predictor.data_ingestion_and_preprocessing(ehr_df)
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)
    lr_predictor.train_model(X_train_lr, y_train_lr)
    lr_predictor.evaluate_model(X_test_lr, y_test_lr)
    lr_predictor.save_model(model_path="lr_chronic_disease_model.joblib", preprocessor_path="lr_preprocessor.joblib")

    new_lr_predictor_instance = ChronicDiseasePredictor()
    new_lr_predictor_instance.load_model(model_path="lr_chronic_disease_model.joblib", preprocessor_path="lr_preprocessor.joblib")
    lr_likelihoods = new_lr_predictor_instance.predict_likelihood(new_patient_data)
    print("\nPredicted likelihood (Logistic Regression):")
    for i, likelihood in enumerate(lr_likelihoods):
        print(f"Patient {i+1}: {likelihood:.4f}")
