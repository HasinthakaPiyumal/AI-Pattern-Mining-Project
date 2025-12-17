import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# 1. Data Ingestion & Storage: Simulate patient data
def generate_synthetic_data(num_samples=1000):
    data = {
        'Age': np.random.randint(20, 80, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'BMI': np.random.uniform(18, 40, num_samples),
        'BloodPressure_Systolic': np.random.randint(100, 180, num_samples),
        'BloodPressure_Diastolic': np.random.randint(60, 120, num_samples),
        'Cholesterol': np.random.randint(150, 250, num_samples),
        'Glucose': np.random.randint(70, 200, num_samples),
        'Smoking': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'AlcoholConsumption': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        'PhysicalActivity': np.random.randint(1, 7, num_samples), # days per week
        'FamilyHistory_Diabetes': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'FamilyHistory_HeartDisease': np.random.choice([0, 1], num_samples, p=[0.75, 0.25]),
        'Diabetes': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]) # Target variable
    }
    df = pd.DataFrame(data)
    
    # Introduce some correlations for 'Diabetes'
    df.loc[df['Glucose'] > 140, 'Diabetes'] = 1
    df.loc[(df['BMI'] > 30) & (df['Age'] > 50), 'Diabetes'] = 1
    df.loc[(df['Smoking'] == 1) & (df['Cholesterol'] > 200), 'Diabetes'] = 1
    df.loc[df['FamilyHistory_Diabetes'] == 1, 'Diabetes'] = 1
    
    df['Diabetes'] = df['Diabetes'].astype(int)
    return df

# 2. Data Preprocessing Module
def preprocess_data(df):
    X = df.drop('Diabetes', axis=1)
    y = df['Diabetes']

    categorical_features = ['Gender']
    numerical_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return preprocessor, X_train, X_test, y_train, y_test, numerical_features, categorical_features

# 3. Model Training Module
def train_models(preprocessor, X_train, y_train):
    models = {
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='liblinear', random_state=42))
        ]),
        'SVC': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(probability=True, random_state=42))
        ]),
        'Decision Tree': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} trained.")
    return trained_models

# 4. Model Evaluation Module
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model.named_steps['classifier'], 'predict_proba') else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 'N/A'

    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}" if roc_auc != 'N/A' else f"ROC AUC: {roc_auc}")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}

# 5. Model Serialization & Deserialization
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

# 6. Prediction & Interpretation Module
def make_prediction_and_interpret(model, new_patient_data_df, preprocessor_transformer, numerical_features, categorical_features, model_name):
    # Ensure the new data has the same columns as the training data, even if empty
    # Create an empty DataFrame with all expected columns for consistent preprocessing
    all_features = numerical_features + categorical_features
    for col in all_features:
        if col not in new_patient_data_df.columns:
            new_patient_data_df[col] = np.nan # or a default value suitable for the feature type

    # Reorder columns to match training data expectation if necessary (handled by preprocessor in pipeline for known features)
    # The pipeline's preprocessor step will handle new data appropriately based on its fitted state

    prediction_proba = model.predict_proba(new_patient_data_df)[:, 1]
    prediction = model.predict(new_patient_data_df)[0]
    
    print(f"\n--- Prediction for New Patient (using {model_name}) ---")
    print(f"Likelihood of Diabetes: {prediction_proba[0]:.4f}")
    print(f"Predicted Diabetes Status: {'Positive' if prediction == 1 else 'Negative'}")

    # Basic Interpretation
    if model_name == 'Logistic Regression':
        classifier = model.named_steps['classifier']
        # Get feature names after one-hot encoding
        ohe_feature_names = preprocessor_transformer.named_transformers_['cat'].get_feature_names_out(categorical_features)
        processed_feature_names = numerical_features + list(ohe_feature_names)
        
        coefficients = pd.Series(classifier.coef_[0], index=processed_feature_names)
        print("\nLogistic Regression Coefficients (Impact on log-odds of Diabetes):")
        print(coefficients.sort_values(ascending=False).head(5))
        print(coefficients.sort_values(ascending=True).head(5))
    elif model_name == 'Decision Tree':
        classifier = model.named_steps['classifier']
        # Get feature names after preprocessing
        ohe_feature_names = preprocessor_transformer.named_transformers_['cat'].get_feature_names_out(categorical_features)
        processed_feature_names = numerical_features + list(ohe_feature_names)

        feature_importances = pd.Series(classifier.feature_importances_, index=processed_feature_names)
        print("\nDecision Tree Feature Importances (Higher means more important):")
        print(feature_importances.sort_values(ascending=False).head(5))
    else:
        print("\nInterpretation for SVC is less direct. Consider other techniques for explainability.")


if __name__ == "__main__":
    print("Generating synthetic patient data...")
    df = generate_synthetic_data(num_samples=1000)
    print("Data head:")
    print(df.head())

    print("\nPreprocessing data...")
    preprocessor, X_train, X_test, y_train, y_test, numerical_features, categorical_features = preprocess_data(df)
    
    print("\nTraining models...")
    trained_models = train_models(preprocessor, X_train, y_train)

    best_model_name = None
    best_roc_auc = -1
    
    print("\nEvaluating models...")
    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        if metrics['roc_auc'] != 'N/A' and metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_model_name = name
    
    if best_model_name:
        print(f"\nBest performing model based on ROC AUC: {best_model_name}")
        best_model = trained_models[best_model_name]
        save_model(best_model, f"{best_model_name.replace(' ', '_').lower()}_model.joblib")

        # Example of loading model and making a new prediction
        loaded_model = load_model(f"{best_model_name.replace(' ', '_').lower()}_model.joblib")

        # Simulate new patient data
        new_patient = pd.DataFrame([{
            'Age': 55,
            'Gender': 'Female',
            'BMI': 32.5,
            'BloodPressure_Systolic': 145,
            'BloodPressure_Diastolic': 90,
            'Cholesterol': 210,
            'Glucose': 160,
            'Smoking': 0,
            'AlcoholConsumption': 1,
            'PhysicalActivity': 2,
            'FamilyHistory_Diabetes': 1,
            'FamilyHistory_HeartDisease': 0
        }])
        
        make_prediction_and_interpret(loaded_model, new_patient, preprocessor, numerical_features, categorical_features, best_model_name)
    else:
        print("No models were evaluated with ROC AUC, or an error occurred.")