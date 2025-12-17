import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. Create a Synthetic Dataset for Demonstration
# In a real-world scenario, you would load your medical data here.
np.random.seed(42)
data_size = 1000

data = {
    'Age': np.random.randint(20, 80, data_size),
    'Gender': np.random.choice(['Male', 'Female'], data_size),
    'BMI': np.random.uniform(18.0, 40.0, data_size),
    'BloodPressure_Systolic': np.random.randint(90, 180, data_size),
    'BloodPressure_Diastolic': np.random.randint(60, 120, data_size),
    'Glucose': np.random.randint(70, 200, data_size),
    'Cholesterol': np.random.randint(150, 300, data_size),
    'Smoker': np.random.choice([0, 1], data_size, p=[0.7, 0.3]),
    'FamilyHistory': np.random.choice([0, 1], data_size, p=[0.6, 0.4]),
    'Diabetes': np.random.choice([0, 1], data_size, p=[0.85, 0.15]) # Target variable
}

df = pd.DataFrame(data)

# Introduce some missing values for demonstration of preprocessing
for col in ['BMI', 'Glucose']:
    missing_indices = np.random.choice(df.index, int(data_size * 0.05), replace=False)
    df.loc[missing_indices, col] = np.nan

print("Sample of the synthetic dataset:")
print(df.head())
print("\nMissing values before preprocessing:")
print(df.isnull().sum())

# 2. Define Features (X) and Target (y)
X = df.drop('Diabetes', axis=1)
y = df['Diabetes']

# 3. Preprocessing Setup
# Identify numerical and categorical features
numerical_features = ['Age', 'BMI', 'BloodPressure_Systolic', 'BloodPressure_Diastolic', 'Glucose', 'Cholesterol']
categorical_features = ['Gender', 'Smoker', 'FamilyHistory']

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Create the Full Machine Learning Pipeline
# This pipeline will first preprocess the data and then train a RandomForestClassifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining data shape: {X_train.shape}, Target shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Target shape: {y_test.shape}")

# 6. Train the Model
print("\nTraining the RandomForestClassifier...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# 7. Make Predictions and Evaluate the Model
y_pred = model_pipeline.predict(X_test)

print("\nModel Evaluation:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example of making a prediction for a new patient
print("\n--- Predicting for a New Patient ---")
new_patient_data = pd.DataFrame({
    'Age': [55],
    'Gender': ['Female'],
    'BMI': [31.5],
    'BloodPressure_Systolic': [145],
    'BloodPressure_Diastolic': [90],
    'Glucose': [180],
    'Cholesterol': [250],
    'Smoker': [1],
    'FamilyHistory': [1]
})

predicted_disease = model_pipeline.predict(new_patient_data)
predicted_proba = model_pipeline.predict_proba(new_patient_data)

print(f"New patient data:\n{new_patient_data.to_string(index=False)}")
print(f"Prediction: {'Diabetes Likely' if predicted_disease[0] == 1 else 'No Diabetes'}")
print(f"Probability of No Diabetes: {predicted_proba[0][0]:.2f}")
print(f"Probability of Diabetes: {predicted_proba[0][1]:.2f}")