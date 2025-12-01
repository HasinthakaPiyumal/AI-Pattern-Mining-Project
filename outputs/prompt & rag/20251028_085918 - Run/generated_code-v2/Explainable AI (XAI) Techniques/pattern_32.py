import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

def generate_synthetic_data(num_samples=1000):
    data = {
        'Age': np.random.randint(20, 80, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'Blood_Pressure': np.random.randint(90, 180, num_samples),
        'Cholesterol': np.random.randint(150, 300, num_samples),
        'Smoking_Status': np.random.choice(['Never', 'Former', 'Current'], num_samples, p=[0.6, 0.2, 0.2]),
        'Medical_Condition': np.random.choice(['None', 'Diabetes', 'Heart Disease', 'Asthma'], num_samples, p=[0.5, 0.2, 0.15, 0.15]),
        'Lab_Result_A': np.random.uniform(0.5, 5.0, num_samples),
        'Lab_Result_B': np.random.uniform(10, 100, num_samples),
    }
    df = pd.DataFrame(data)

    # Introduce some correlation for disease progression
    df['Disease_Progression'] = 0
    df.loc[(df['Age'] > 60) & (df['Cholesterol'] > 220), 'Disease_Progression'] = 1
    df.loc[(df['Blood_Pressure'] > 140) & (df['Smoking_Status'] == 'Current'), 'Disease_Progression'] = 1
    df.loc[(df['Medical_Condition'] == 'Diabetes') & (df['Lab_Result_A'] > 3.0), 'Disease_Progression'] = 1
    df.loc[np.random.rand(num_samples) < 0.1, 'Disease_Progression'] = 1 # Random noise
    
    df['Disease_Progression'] = df['Disease_Progression'].astype(int)

    return df

# Generate data
df = generate_synthetic_data()

X = df.drop('Disease_Progression', axis=1)
y = df['Disease_Progression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing steps
numerical_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include='object').columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create the pipeline with preprocessing and Logistic Regression model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# --- Interpretation --- 
# Get feature names after one-hot encoding
encoded_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = list(numerical_features) + list(encoded_feature_names)

# Get coefficients from the Logistic Regression model
coefficients = model_pipeline.named_steps['classifier'].coef_[0]
intercept = model_pipeline.named_steps['classifier'].intercept_[0]

print("Model Coefficients (feature importance for interpretation):")
for feature, coef in zip(all_feature_names, coefficients):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {intercept:.4f}")

print("\nInterpretation: A positive coefficient indicates that as the feature value increases (or if the categorical feature is present), the log-odds of disease progression increase. A negative coefficient indicates the opposite.")

# --- Prediction Example --- 
print("\n--- Prediction Example ---")
new_patient_data = pd.DataFrame([{
    'Age': 65,
    'Gender': 'Female',
    'Blood_Pressure': 150,
    'Cholesterol': 250,
    'Smoking_Status': 'Current',
    'Medical_Condition': 'Diabetes',
    'Lab_Result_A': 3.5,
    'Lab_Result_B': 80,
}])

predicted_risk_proba = model_pipeline.predict_proba(new_patient_data)[:, 1][0]
predicted_class = model_pipeline.predict(new_patient_data)[0]

print(f"New Patient Data:\n{new_patient_data.to_string(index=False)}")
print(f"Predicted Probability of Disease Progression: {predicted_risk_proba:.4f}")
print(f"Predicted Class (0: No Progression, 1: Progression): {predicted_class}")

print("\nTo understand this prediction, medical professionals can refer to the coefficients above. For example, a 'Current' Smoking Status or 'Diabetes' Medical Condition with high 'Age' and 'Cholesterol' values contribute positively to the risk, as reflected by their respective positive coefficients.")
