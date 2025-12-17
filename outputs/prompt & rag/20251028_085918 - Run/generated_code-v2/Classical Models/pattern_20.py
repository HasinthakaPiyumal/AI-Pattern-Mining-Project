import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class DiagnosticAssistant:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Support Vector Machine": SVC(probability=True, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }
        self.trained_models = {}

    def preprocess_data(self, df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle missing values (simple imputation with mean for numerical, could be more complex)
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].mean())
            else:
                # For categorical, simple mode imputation or more advanced methods
                X[col] = X[col].fillna(X[col].mode()[0])
        
        # One-hot encode categorical features if any
        X = pd.get_dummies(X, drop_first=True)

        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns), y

    def train_models(self, X_train, y_train):
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            print(f"{name} trained.")

    def evaluate_models(self, X_test, y_test):
        results = {}
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 'N/A'

            results[name] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "ROC-AUC": roc_auc
            }
        return results

    def predict_disease(self, new_patient_data_df):
        # Ensure new_patient_data_df has the same columns as the training data after preprocessing
        # This is a simplified example; in a real app, you'd need to align columns carefully
        
        # For demonstration, we'll assume the new data needs to be scaled using the fitted scaler
        # In a real application, you'd align columns and then scale.
        new_patient_data_scaled = self.scaler.transform(new_patient_data_df.values.reshape(1, -1))

        predictions = {}
        for name, model in self.trained_models.items():
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(new_patient_data_scaled)[:, 1][0]
                predictions[name] = f"{prob:.4f}"
            else:
                pred = model.predict(new_patient_data_scaled)[0]
                predictions[name] = "Positive" if pred == 1 else "Negative"
        return predictions

# Example Usage:
if __name__ == "__main__":
    # 1. Generate Dummy Data (replace with your actual data loading)
    np.random.seed(42)
    data_size = 1000
    dummy_data = pd.DataFrame({
        'Age': np.random.randint(20, 80, data_size),
        'BloodPressure': np.random.randint(90, 180, data_size),
        'Cholesterol': np.random.randint(150, 300, data_size),
        'Glucose': np.random.randint(70, 200, data_size),
        'BMI': np.random.uniform(18.0, 40.0, data_size),
        'Smoker': np.random.choice([0, 1], data_size),
        'FamilyHistory': np.random.choice([0, 1], data_size),
        'ExerciseHoursWeekly': np.random.uniform(0.5, 10.0, data_size),
        'TargetDisease': np.random.choice([0, 1], data_size, p=[0.7, 0.3]) # 0 for no disease, 1 for disease
    })

    assistant = DiagnosticAssistant()

    # 2. Preprocess Data
    X, y = assistant.preprocess_data(dummy_data.copy(), 'TargetDisease')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Models
    assistant.train_models(X_train, y_train)

    # 4. Evaluate Models
    evaluation_results = assistant.evaluate_models(X_test, y_test)
    print("\n--- Model Evaluation Results ---")
    for model_name, metrics in evaluation_results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # 5. Make a Prediction for a New Patient
    print("\n--- New Patient Prediction ---")
    # Create a new patient DataFrame, ensure column order and types match training data
    new_patient_data = pd.DataFrame([[55, 130, 220, 110, 28.5, 0, 1, 5.0]], 
                                    columns=['Age', 'BloodPressure', 'Cholesterol', 'Glucose', 'BMI', 'Smoker', 'FamilyHistory', 'ExerciseHoursWeekly'])
    
    # In a real scenario, you'd apply the same preprocessing steps to new_patient_data as done during training.
    # For this simplified example, we'll manually align and scale the raw input.
    # Note: `preprocess_data` on a single row might be tricky without careful handling of one-hot encoding for unseen categories.
    # For now, we assume numerical inputs matching the scaled features.

    # To make this robust, for new prediction, you should feed raw data through a similar `preprocess_data` function
    # which uses the *fitted* scaler and one-hot encoder from training.
    
    # A more robust approach for prediction might involve storing the 'columns' used in X during training
    # and then ensuring `new_patient_data` aligns with those columns before scaling.
    
    # For simplicity, let's assume `new_patient_data` is already prepared to be directly scaled.
    # In a real application, you'd need to convert this into the exact feature vector (with dummy variables, etc.)
    # that the model expects.
    
    # Here's a simplified way to create the correct input format for prediction, assuming numerical features only for now.
    # This needs to be robust for categorical features if present.
    
    # Create a dummy DataFrame to get the column order for preprocessing
    # This is a bit of a hack for the example, a robust solution would save feature names.
    dummy_for_cols = dummy_data.drop(columns=['TargetDisease'])
    dummy_for_cols = pd.get_dummies(dummy_for_cols, drop_first=True)
    
    # Ensure new_patient_data has the same columns and order as the training data before scaling
    new_patient_processed = pd.DataFrame(columns=dummy_for_cols.columns)
    new_patient_processed.loc[0] = 0 # Initialize with zeros
    
    # Fill in the actual values from new_patient_data
    for col in new_patient_data.columns:
        if col in new_patient_processed.columns:
            new_patient_processed[col] = new_patient_data[col].iloc[0]

    # Scale the new patient data using the *fitted* scaler
    # This requires the input to be in the correct shape and column order
    # For simplicity, assuming new_patient_processed now has the correct numerical features and order

    # For accurate prediction, the `preprocess_data` method should be refactored to handle new single-row data for prediction correctly
    # For now, let's provide a warning that this is a simplified example for prediction input.

    # *** IMPORTANT NOTE FOR PREDICTION: ***
    # The current `predict_disease` method assumes `new_patient_data_df` is already scaled 
    # and has the correct shape. In a real application, you would need to apply the 
    # same preprocessing steps (handling categorical, imputation, scaling) to 
    # the raw new patient data using the *fitted* scaler and any learned encoders.
    # A robust solution would involve saving the preprocessor state.

    # For this example, let's manually create a scaled input for a new patient
    # (This assumes you know the scaled values or have a separate preprocessor for new data)
    example_new_patient_scaled = assistant.scaler.transform(new_patient_data.values.reshape(1, -1))
    
    # Call predict_disease with the correctly formatted (scaled) input
    # Note: The `predict_disease` method expects a DataFrame, so we wrap it.
    # This is a point that would need refinement in a production system.
    prediction_results = assistant.predict_disease(pd.DataFrame(example_new_patient_scaled, columns=X.columns)) 

    print("Likelihood of disease for new patient:")
    for model_name, prob in prediction_results.items():
        print(f"  {model_name}: {prob}")