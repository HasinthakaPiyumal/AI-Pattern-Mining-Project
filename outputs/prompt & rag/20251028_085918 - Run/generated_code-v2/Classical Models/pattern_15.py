import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def load_and_preprocess_data(filepath='dummy_patient_data.csv'):
    # Create a dummy dataset if the file doesn't exist
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Creating dummy data as {filepath} not found.")
        data = {
            'Age': np.random.randint(20, 80, 100),
            'Gender': np.random.choice(['Male', 'Female'], 100),
            'BMI': np.random.uniform(18, 40, 100),
            'BloodPressure': np.random.randint(90, 180, 100),
            'Cholesterol': np.random.randint(150, 250, 100),
            'Glucose': np.random.randint(70, 200, 100),
            'Smoking': np.random.choice([0, 1], 100),
            'ExerciseHoursWeek': np.random.uniform(0, 10, 100),
            'FamilyHistory': np.random.choice([0, 1], 100),
            'Disease': np.random.choice([0, 1], 100) # 0: No Disease, 1: Disease
        }
        df = pd.DataFrame(data)
        # Introduce some missing values for demonstration
        for col in ['BMI', 'Cholesterol']:
            df.loc[df.sample(frac=0.1).index, col] = np.nan
        df.to_csv(filepath, index=False)

    # Handle missing values (e.g., fill with mean for numerical columns)
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    
    # Encode categorical features (Gender)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, X.columns

def train_models(X_train, y_train):
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, solver='liblinear'),
        'SVC': SVC(probability=True, random_state=42),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0] * len(y_test) # Handle models without predict_proba

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }
    return results

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    return joblib.load(filename)

def make_prediction(model, scaler, feature_names, patient_data):
    # patient_data should be a dictionary like {'Age': 50, 'Gender': 'Male', ...}
    patient_df = pd.DataFrame([patient_data])
    
    # Ensure gender is encoded consistently
    patient_df['Gender'] = patient_df['Gender'].map({'Male': 0, 'Female': 1})

    # Align columns if patient_data doesn't have all feature_names (fill with 0 or mean of training data if possible)
    # For simplicity, we assume patient_data has all relevant features in correct order/names
    # A more robust solution would involve recreating the DataFrame with all feature_names and filling missing ones
    
    # Ensure the order of columns matches the training data
    patient_df_aligned = patient_df[feature_names] # This assumes patient_data has all feature_names

    patient_scaled = scaler.transform(patient_df_aligned)
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0, 1] if hasattr(model, 'predict_proba') else None
    
    return 'Disease Present' if prediction == 1 else 'No Disease', probability

if __name__ == "__main__":
    data_filepath = 'patient_data.csv'
    X, y, scaler, feature_names = load_and_preprocess_data(data_filepath)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trained_models = train_models(X_train, y_train)
    
    evaluation_results = evaluate_models(trained_models, X_test, y_test)
    
    print("\n--- Model Evaluation Results ---")
    for name, metrics in evaluation_results.items():
        print(f"\nModel: {name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            
    # Example of saving and loading a model
    best_model_name = 'LogisticRegression' # Assuming Logistic Regression is chosen as the best for this example
    save_model(trained_models[best_model_name], f'{best_model_name}_model.joblib')
    loaded_model = load_model(f'{best_model_name}_model.joblib')
    
    # Example of making a prediction for a new patient
    print("\n--- Example Prediction ---")
    new_patient_data = {
        'Age': 60,
        'Gender': 'Female',
        'BMI': 32.5,
        'BloodPressure': 140,
        'Cholesterol': 220,
        'Glucose': 180,
        'Smoking': 1,
        'ExerciseHoursWeek': 2.0,
        'FamilyHistory': 1
    }
    
    prediction, probability = make_prediction(loaded_model, scaler, feature_names, new_patient_data)
    print(f"Patient data: {new_patient_data}")
    print(f"Predicted Disease Status: {prediction}")
    if probability is not None:
        print(f"Prediction Probability (Disease Present): {probability:.4f}")
