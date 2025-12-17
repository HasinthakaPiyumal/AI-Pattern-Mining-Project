import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 90, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Ethnicity': np.random.choice(['White', 'Black', 'Asian', 'Other'], n_samples),
        'AdmissionType': np.random.choice(['Emergency', 'Elective', 'Urgent'], n_samples),
        'NumDiagnoses': np.random.randint(1, 10, n_samples),
        'NumProcedures': np.random.randint(0, 5, n_samples),
        'LengthOfStay': np.random.randint(1, 30, n_samples),
        'HasDiabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'HasHeartDisease': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'LabResults_Creatinine': np.random.normal(1.0, 0.3, n_samples),
        'LabResults_Hemoglobin': np.random.normal(13.5, 1.5, n_samples),
        'Readmitted': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]) # Target variable
    }
    df = pd.DataFrame(data)
    
    # Introduce some missing values for demonstration
    for col in ['LabResults_Creatinine', 'Ethnicity']:
        df.loc[df.sample(frac=0.05).index, col] = np.nan
        
    # Make readmission somewhat dependent on other features
    df['Readmitted'] = np.where(
        (df['LengthOfStay'] > 15) | (df['NumDiagnoses'] > 5) | (df['HasHeartDisease'] == 1),
        np.random.choice([0, 1], n_samples, p=[0.5, 0.5]), # Higher chance of readmission
        df['Readmitted'] # Keep existing value otherwise
    )
    df['Readmitted'] = np.where(df['Readmitted'] > 0.5, 1, 0) # Ensure binary

    return df

class PatientReadmissionPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None

    def preprocess_data(self, df):
        numerical_features = df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = df.select_dtypes(include='object').columns.tolist()
        
        # Remove target from features if present
        if 'Readmitted' in numerical_features:
            numerical_features.remove('Readmitted')

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
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        return self.preprocessor.fit_transform(df)
    
    def get_feature_names_out(self, preprocessor, input_features):
        output_features = []
        for name, transformer, features in preprocessor.transformers_:
            if name == 'remainder' or transformer == 'passthrough':
                output_features.extend(features)
            elif hasattr(transformer, 'get_feature_names_out'):
                if hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
                    oh_transformer = transformer.named_steps['onehot']
                    oh_feature_names = oh_transformer.get_feature_names_out(features)
                    output_features.extend(oh_feature_names)
                else:
                    output_features.extend(transformer.get_feature_names_out(features))
            else:
                # For simple transformers like scalers, the feature names remain the same
                output_features.extend(features)
        return output_features

    def train(self, X_train, y_train):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"\n--- Model Evaluation ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"------------------------")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "roc_auc": roc_auc}

    def predict(self, new_patient_data_df):
        if self.preprocessor is None or self.model is None:
            raise RuntimeError("Preprocessor or model has not been fitted/trained yet.")
        
        # Apply the *fitted* preprocessor to new data
        processed_new_data = self.preprocessor.transform(new_patient_data_df)
        prediction = self.model.predict(processed_new_data)
        prediction_proba = self.model.predict_proba(processed_new_data)[:, 1]
        return prediction, prediction_proba

# Main execution block
if __name__ == "__main__":
    from sklearn.impute import SimpleImputer # Import SimpleImputer here
    print("Generating synthetic patient data...")
    df = generate_synthetic_data(n_samples=2000)
    print(f"Synthetic data generated with {len(df)} samples and {len(df.columns)} features.")
    print("Data head:\n", df.head())
    print("\nMissing values:\n", df.isnull().sum())

    # Separate features and target
    X = df.drop('Readmitted', axis=1)
    y = df['Readmitted']

    # Initialize and preprocess data
    predictor = PatientReadmissionPredictor()
    
    # Split data BEFORE fitting the preprocessor to avoid data leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nPreprocessing training data...")
    X_train_processed = predictor.preprocess_data(X_train_raw)
    
    # Transform test data using the *fitted* preprocessor
    print("Preprocessing test data...")
    X_test_processed = predictor.preprocessor.transform(X_test_raw)
    
    # Get feature names after preprocessing
    predictor.feature_names = predictor.get_feature_names_out(predictor.preprocessor, X_train_raw.columns)

    print("\nTraining the XGBoost model...")
    predictor.train(X_train_processed, y_train)

    print("\nEvaluating the model on the test set...")
    predictor.evaluate(X_test_processed, y_test)

    print("\nDemonstrating prediction on new patient data...")
    # Create some new synthetic patient data for prediction
    new_patient_data = {
        'Age': [75, 45, 60],
        'Gender': ['Female', 'Male', 'Female'],
        'Ethnicity': ['White', 'Black', 'Other'],
        'AdmissionType': ['Emergency', 'Elective', 'Urgent'],
        'NumDiagnoses': [7, 2, 4],
        'NumProcedures': [3, 1, 0],
        'LengthOfStay': [20, 5, 12],
        'HasDiabetes': [1, 0, 0],
        'HasHeartDisease': [1, 0, 1],
        'LabResults_Creatinine': [2.1, 0.8, 1.3],
        'LabResults_Hemoglobin': [10.5, 14.2, 12.8]
    }
    new_patient_df = pd.DataFrame(new_patient_data)
    
    # Introduce a missing value to test imputer
    new_patient_df.loc[0, 'Ethnicity'] = np.nan

    print("New patient data for prediction:\n", new_patient_df)
    
    predictions, probabilities = predictor.predict(new_patient_df)
    print("\nPredictions (0: Not Readmitted, 1: Readmitted):", predictions)
    print("Readmission Probabilities:", [f'{p:.4f}' for p in probabilities])

    # Add predictions to the new patient data for better visualization
    new_patient_df['Predicted_Readmitted'] = predictions
    new_patient_df['Readmission_Probability'] = probabilities
    print("\nNew patient data with predictions:\n", new_patient_df[['Age', 'LengthOfStay', 'HasHeartDisease', 'Predicted_Readmitted', 'Readmission_Probability']])
