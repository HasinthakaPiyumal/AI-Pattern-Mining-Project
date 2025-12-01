import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

class DataHandler:
    def load_data(self, filepath=None):
        if filepath:
            return pd.read_csv(filepath)
        else:
            # Create a synthetic dataset for demonstration
            np.random.seed(42)
            data = {
                'age': np.random.randint(20, 90, 1000),
                'gender': np.random.choice(['Male', 'Female'], 1000),
                'num_procedures': np.random.randint(0, 5, 1000),
                'num_medications': np.random.randint(1, 20, 1000),
                'time_in_hospital': np.random.randint(1, 15, 1000),
                'diagnosis_code': np.random.choice([f'D{i}' for i in range(1, 10)], 1000),
                'num_lab_procedures': np.random.randint(1, 60, 1000),
                'had_emergency': np.random.choice([0, 1], 1000, p=[0.7, 0.3]),
                'readmitted': np.random.choice([0, 1], 1000, p=[0.6, 0.4])
            }
            df = pd.DataFrame(data)
            return df

    def preprocess_data(self, df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        numerical_features = X.select_dtypes(include=np.number).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Fit and transform on X
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after one-hot encoding
        new_column_names = []
        for name, transformer, features in preprocessor.transformers_:
            if name == 'num':
                new_column_names.extend(features)
            elif name == 'cat':
                new_column_names.extend(transformer.get_feature_names_out(features))

        X_processed_df = pd.DataFrame(X_processed, columns=new_column_names)

        return X_processed_df, y, new_column_names

class ModelTrainer:
    def train_model(self, X_train, y_train):
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        return {"accuracy": accuracy, "f1_score": f1, "roc_auc": roc_auc}

class FeatureImportanceCalculator:
    def calculate_permutation_importance(self, model, X_test, y_test, feature_names, scoring='accuracy'):
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring=scoring)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        })
        importance_df = importance_df.sort_values(by='importance_mean', ascending=False)
        return importance_df

class Visualizer:
    def plot_feature_importance(self, importance_df):
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance_mean', y='feature', data=importance_df)
        plt.title('Permutation Feature Importance for Readmission Prediction')
        plt.xlabel('Mean Importance (Drop in Accuracy)')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Starting Healthcare Patient Readmission Risk Prediction with PFI...")

    # 1. Data Ingestion and Preprocessing
    data_handler = DataHandler()
    df = data_handler.load_data()  # Loads synthetic data if no filepath given
    
    target_column = 'readmitted'
    X_processed, y, feature_names = data_handler.preprocess_data(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Dataset split into train ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples).")
    print(f"Number of features after preprocessing: {X_train.shape[1]}")

    # 2. Model Training
    model_trainer = ModelTrainer()
    model = model_trainer.train_model(X_train, y_train)
    print("Model training complete.")

    # 3. Model Evaluation
    evaluation_metrics = model_trainer.evaluate_model(model, X_test, y_test)
    print("\nModel Evaluation Metrics:")
    for metric, value in evaluation_metrics.items():
        print(f"  {metric.replace('_', ' ').capitalize()}: {value:.4f}")

    # 4. Permutation Feature Importance Calculation
    pfi_calculator = FeatureImportanceCalculator()
    importance_df = pfi_calculator.calculate_permutation_importance(model, X_test, y_test, feature_names, scoring='accuracy')
    print("\nPermutation Feature Importance calculated.")
    print("\nTop 10 Features by Importance (Mean Drop in Accuracy):")
    print(importance_df.head(10).to_string(index=False))

    # 5. Reporting and Visualization
    visualizer = Visualizer()
    visualizer.plot_feature_importance(importance_df)

    print("Process complete. Feature importance plot displayed.")
