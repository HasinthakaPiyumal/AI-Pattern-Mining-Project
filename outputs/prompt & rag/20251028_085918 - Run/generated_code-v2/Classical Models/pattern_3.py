import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# 1. Simulate Data Ingestion (creating a synthetic dataset)
def create_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(25, 80, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'BMI': np.random.normal(28, 5, num_samples),
        'HbA1c': np.random.normal(6.0, 1.0, num_samples), # Glycated hemoglobin
        'BloodPressure_Systolic': np.random.randint(100, 180, num_samples),
        'BloodPressure_Diastolic': np.random.randint(60, 110, num_samples),
        'Cholesterol': np.random.normal(200, 40, num_samples),
        'Smoking': np.random.choice(['Yes', 'No'], num_samples, p=[0.2, 0.8]),
        'FamilyHistoryDiabetes': np.random.choice(['Yes', 'No'], num_samples, p=[0.3, 0.7]),
        'PhysicalActivity': np.random.choice(['Low', 'Medium', 'High'], num_samples, p=[0.3, 0.5, 0.2]),
        'DiabetesRisk': np.random.randint(0, 2, num_samples) # 0: Low Risk, 1: High Risk
    }
    df = pd.DataFrame(data)

    # Introduce some missing values for demonstration
    for col in ['BMI', 'HbA1c', 'Cholesterol']:
        df.loc[np.random.choice(df.index, int(num_samples * 0.05), replace=False), col] = np.nan
    df.loc[np.random.choice(df.index, int(num_samples * 0.02), replace=False), 'Smoking'] = np.nan

    # Make DiabetesRisk slightly correlated with other features
    df.loc[df['HbA1c'] > 6.5, 'DiabetesRisk'] = 1
    df.loc[df['BMI'] > 30, 'DiabetesRisk'] = 1
    df.loc[(df['BMI'] < 25) & (df['HbA1c'] < 5.7), 'DiabetesRisk'] = 0

    # Ensure binary classification target
    df['DiabetesRisk'] = df['DiabetesRisk'].astype(int)

    return df

print("Generating synthetic patient data...")
df = create_synthetic_data(num_samples=1500)
print(f"Dataset shape: {df.shape}")
print("First 5 rows of the dataset:")
print(df.head())
print("\nMissing values before preprocessing:")
print(df.isnull().sum())

# Define features and target
X = df.drop('DiabetesRisk', axis=1)
y = df['DiabetesRisk']

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# 2. Data Preprocessing Module
# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

# 3. Model Training Module
# Create the full pipeline: preprocessing + model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

print("\nTraining the Logistic Regression model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the model
print("\nEvaluating the model on the test set...")
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# Save the trained model pipeline
model_filename = 'diabetes_risk_model.joblib'
joblib.dump(model_pipeline, model_filename)
print(f"\nModel saved to {model_filename}")

# 4. Prediction and Output Module - Demonstrate prediction on new data
print("\nDemonstrating prediction on a new hypothetical patient record...")
new_patient_data = pd.DataFrame({
    'Age': [55],
    'Gender': ['Female'],
    'BMI': [31.5],
    'HbA1c': [7.1],
    'BloodPressure_Systolic': [145],
    'BloodPressure_Diastolic': [90],
    'Cholesterol': [230],
    'Smoking': ['No'],
    'FamilyHistoryDiabetes': ['Yes'],
    'PhysicalActivity': ['Low']
})

# Ensure column order matches training data (important for prediction)
# It's better to ensure this through feature names from preprocessor, but for simple demo, direct df creation is fine.

# Load the saved model (for demonstration purposes, in a real scenario this would be a separate step)
loaded_model = joblib.load(model_filename)

new_patient_prediction = loaded_model.predict(new_patient_data)
new_patient_proba = loaded_model.predict_proba(new_patient_data)[:, 1]

risk_label = "High Risk" if new_patient_prediction[0] == 1 else "Low Risk"
print(f"Predicted Diabetes Risk for new patient: {risk_label}")
print(f"Probability of High Risk: {new_patient_proba[0]:.4f}")

print("\nAnother hypothetical patient (low risk profile):")
new_patient_data_2 = pd.DataFrame({
    'Age': [30],
    'Gender': ['Male'],
    'BMI': [22.0],
    'HbA1c': [5.5],
    'BloodPressure_Systolic': [110],
    'BloodPressure_Diastolic': [70],
    'Cholesterol': [170],
    'Smoking': ['No'],
    'FamilyHistoryDiabetes': ['No'],
    'PhysicalActivity': ['High']
})

new_patient_prediction_2 = loaded_model.predict(new_patient_data_2)
new_patient_proba_2 = loaded_model.predict_proba(new_patient_data_2)[:, 1]

risk_label_2 = "High Risk" if new_patient_prediction_2[0] == 1 else "Low Risk"
print(f"Predicted Diabetes Risk for new patient 2: {risk_label_2}")
print(f"Probability of High Risk: {new_patient_proba_2[0]:.4f}")
