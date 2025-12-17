import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Data Simulation/Loading
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(20, 80, n_samples),
    'bmi': np.random.uniform(18.0, 40.0, n_samples),
    'blood_pressure': np.random.randint(90, 180, n_samples),
    'cholesterol': np.random.randint(150, 300, n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'family_history_diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'smoker': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'diabetes_risk': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]) # Target variable
}

df = pd.DataFrame(data)

# Introduce some correlation for diabetes_risk for more realistic data
df.loc[df['bmi'] > 30, 'diabetes_risk'] = np.random.choice([0, 1], len(df[df['bmi'] > 30]), p=[0.3, 0.7])
df.loc[df['age'] > 60, 'diabetes_risk'] = np.random.choice([0, 1], len(df[df['age'] > 60]), p=[0.4, 0.6])
df.loc[df['family_history_diabetes'] == 1, 'diabetes_risk'] = np.random.choice([0, 1], len(df[df['family_history_diabetes'] == 1]), p=[0.35, 0.65])

X = df.drop('diabetes_risk', axis=1)
y = df['diabetes_risk']

# 2. Data Preprocessing Pipeline
numerical_features = ['age', 'bmi', 'blood_pressure', 'cholesterol']
categorical_features = ['gender', 'family_history_diabetes', 'smoker']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 3. Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Pipeline
logistic_regression_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])
logistic_regression_pipeline.fit(X_train, y_train)

# Random Forest Classifier Pipeline
random_forest_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
random_forest_pipeline.fit(X_train, y_train)

# 4. Model Evaluation
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

evaluate_model(logistic_regression_pipeline, X_test, y_test, "Logistic Regression")
evaluate_model(random_forest_pipeline, X_test, y_test, "Random Forest Classifier")

# 5. Prediction Functionality
def predict_patient_risk(model, new_patient_data: dict):
    new_df = pd.DataFrame([new_patient_data])
    prediction = model.predict(new_df)
    probability = model.predict_proba(new_df)[:, 1]
    return {"prediction": int(prediction[0]), "probability_of_diabetes": float(probability[0])}

# Example of using the prediction functionality
print("\n--- Demonstrating Prediction Functionality ---")
new_patient_1 = {
    'age': 55,
    'bmi': 31.5,
    'blood_pressure': 145,
    'cholesterol': 220,
    'gender': 'Female',
    'family_history_diabetes': 1,
    'smoker': 0
}

new_patient_2 = {
    'age': 30,
    'bmi': 22.0,
    'blood_pressure': 110,
    'cholesterol': 180,
    'gender': 'Male',
    'family_history_diabetes': 0,
    'smoker': 0
}

print("Predicting for Patient 1 (using Random Forest):")
prediction_1 = predict_patient_risk(random_forest_pipeline, new_patient_1)
print(f"Prediction: {prediction_1['prediction']}, Probability of Diabetes: {prediction_1['probability_of_diabetes']:.4f}")

print("\nPredicting for Patient 2 (using Logistic Regression):")
prediction_2 = predict_patient_risk(logistic_regression_pipeline, new_patient_2)
print(f"Prediction: {prediction_2['prediction']}, Probability of Diabetes: {prediction_2['probability_of_diabetes']:.4f}")
