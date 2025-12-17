import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Data Layer: Create a synthetic dataset for demonstration
np.random.seed(42)
num_samples = 1000

data = {
    'age': np.random.randint(25, 75, num_samples),
    'gender': np.random.choice(['Male', 'Female'], num_samples),
    'blood_pressure_systolic': np.random.randint(100, 180, num_samples),
    'blood_pressure_diastolic': np.random.randint(60, 120, num_samples),
    'cholesterol': np.random.randint(150, 250, num_samples),
    'glucose': np.random.randint(70, 200, num_samples),
    'bmi': np.random.uniform(18.0, 35.0, num_samples),
    'family_history': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]), # 0: No, 1: Yes
    'smoking': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]), # 0: No, 1: Yes
    'disease_present': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]) # Target variable
}

# Introduce some correlation for 'disease_present'
data['disease_present'] = (
    (data['age'] > 50).astype(int) + 
    (data['bmi'] > 28).astype(int) + 
    (data['glucose'] > 120).astype(int) + 
    (data['family_history'] == 1).astype(int)
) > 2.0
data['disease_present'] = data['disease_present'].astype(int)

df = pd.DataFrame(data)

# Save to CSV for consistency with architecture description (optional, but good for testing)
df.to_csv('patient_data.csv', index=False)

print("Synthetic patient data created and saved to 'patient_data.csv'")

# Load the data (as described in the architecture)
df = pd.read_csv('patient_data.csv')

# Separate features and target
X = df.drop('disease_present', axis=1)
y = df['disease_present']

# Identify numerical and categorical features
numerical_features = ['age', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'cholesterol', 'glucose', 'bmi']
categorical_features = ['gender', 'family_history', 'smoking'] # Assuming family_history and smoking are categorical even if 0/1

# 2. Data Preprocessing Layer
# Create a column transformer for preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline that includes preprocessing and the model
# 3. Model Layer
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
print("\nTraining the Logistic Regression model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# 4. Model Evaluation
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

print("\nModel Evaluation on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# 5. Prediction and Interpretation Layer

# Example of a new patient data for prediction
new_patient_data = pd.DataFrame({
    'age': [60],
    'gender': ['Male'],
    'blood_pressure_systolic': [150],
    'blood_pressure_diastolic': [95],
    'cholesterol': [230],
    'glucose': [160],
    'bmi': [30.5],
    'family_history': [1],
    'smoking': [1]
})

print("\nPredicting for a new patient:")
print(new_patient_data)

# Predict disease likelihood for the new patient
new_patient_prediction_proba = model_pipeline.predict_proba(new_patient_data)[:, 1]
new_patient_prediction = model_pipeline.predict(new_patient_data)[0]

print(f"\nPredicted disease likelihood: {new_patient_prediction_proba[0]:.4f}")
print(f"Predicted disease presence: {'Yes' if new_patient_prediction == 1 else 'No'}")

# Interpretation: Feature importance using coefficients (for Logistic Regression)
print("\nInterpreting Feature Importance (Logistic Regression Coefficients):")

# Get feature names after one-hot encoding
onehot_features = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_features = numerical_features + list(onehot_features)

# Get coefficients from the logistic regression model
coefficients = model_pipeline.named_steps['classifier'].coef_[0]

feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

print(feature_importance)

print("\nPositive coefficients indicate features that increase the likelihood of disease.")
print("Negative coefficients indicate features that decrease the likelihood of disease.")
print("The magnitude of the coefficient indicates the strength of the influence.")