
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Simulate Data Ingestion
# Create a dummy dataset for demonstration purposes
np.random.seed(42)
data_size = 1000

dummy_data = {
    'age': np.random.randint(20, 90, data_size),
    'gender': np.random.choice(['Male', 'Female'], data_size),
    'ethnicity': np.random.choice(['White', 'Black', 'Asian', 'Other'], data_size),
    'num_diagnoses': np.random.randint(1, 10, data_size),
    'num_medications': np.random.randint(1, 20, data_size),
    'length_of_stay': np.random.randint(1, 30, data_size),
    'prior_admissions': np.random.randint(0, 5, data_size),
    'comorbidity_score': np.random.uniform(0.1, 0.9, data_size),
    'readmitted': np.random.choice([0, 1], data_size, p=[0.7, 0.3]) # 0: No, 1: Yes
}

df = pd.DataFrame(dummy_data)

print("Original DataFrame head:")
print(df.head())
print("\n")

# Define features (X) and target (y)
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Identify categorical and numerical features
categorical_features = ['gender', 'ethnicity']
numerical_features = ['age', 'num_diagnoses', 'num_medications', 'length_of_stay', 'prior_admissions', 'comorbidity_score']

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 2. Model Selection and Training
# Create a full pipeline with preprocessing and RandomForestClassifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training the RandomForestClassifier...")
model_pipeline.fit(X_train, y_train)
print("Training complete.\n")

# 3. Model Evaluation
print("Evaluating the model...")
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Model Evaluation on Test Set:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")
print(f"  ROC AUC: {roc_auc:.4f}\n")

# 4. Prediction Endpoint (Demonstration)
print("Demonstrating prediction for new patient data...")
# Create a sample new patient data point (similar structure to original data)
new_patient_data = pd.DataFrame({
    'age': [75],
    'gender': ['Female'],
    'ethnicity': ['White'],
    'num_diagnoses': [8],
    'num_medications': [15],
    'length_of_stay': [10],
    'prior_admissions': [2],
    'comorbidity_score': [0.85]
})

predicted_readmission = model_pipeline.predict(new_patient_data)
predicted_proba = model_pipeline.predict_proba(new_patient_data)[:, 1]

if predicted_readmission[0] == 1:
    print(f"New patient is predicted to be READMITTED (Probability: {predicted_proba[0]:.4f}).")
else:
    print(f"New patient is predicted NOT to be readmitted (Probability: {predicted_proba[0]:.4f}).")
