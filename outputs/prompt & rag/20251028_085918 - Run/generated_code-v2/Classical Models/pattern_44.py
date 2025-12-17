import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# 1. Synthetic Data Generation
patients_data = {
    'Age': np.random.randint(20, 70, 200),
    'BMI': np.random.uniform(18.0, 45.0, 200),
    'Glucose': np.random.randint(70, 200, 200),
    'BloodPressure': np.random.randint(60, 120, 200),
    'Insulin': np.random.randint(0, 300, 200),
    'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.5, 200),
    'SmokingStatus': np.random.choice(['Never', 'Current', 'Ex-smoker'], 200),
    'Outcome': np.random.choice([0, 1], 200, p=[0.7, 0.3]) # 0: No Diabetes, 1: Diabetes
}
df = pd.DataFrame(patients_data)

# Introduce some NaN values for demonstration of handling missing data (optional, for robustness)
# df.loc[df.sample(frac=0.05).index, 'BMI'] = np.nan

# Separate features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Define numerical and categorical features
numerical_features = ['Age', 'BMI', 'Glucose', 'BloodPressure', 'Insulin', 'DiabetesPedigreeFunction']
categorical_features = ['SmokingStatus']

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create full pipelines for each model, including preprocessing
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', LogisticRegression(random_state=42, solver='liblinear'))])
pipeline_svc = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', SVC(random_state=42))])
pipeline_dt = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', DecisionTreeClassifier(random_state=42))])

models = {
    'Logistic Regression': pipeline_lr,
    'Support Vector Classifier': pipeline_svc,
    'Decision Tree Classifier': pipeline_dt
}

print("\n--- Model Training and Evaluation ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 4. Prediction Module (Demonstration)
print("\n--- Prediction Demonstration ---")
# Create a new synthetic patient record for prediction
new_patient_data = pd.DataFrame({
    'Age': [35],
    'BMI': [28.5],
    'Glucose': [150],
    'BloodPressure': [85],
    'Insulin': [120],
    'DiabetesPedigreeFunction': [0.6],
    'SmokingStatus': ['Current']
})

# Use one of the trained models for prediction (e.g., Logistic Regression)
print("Predicting for a new patient using Logistic Regression model:")
new_patient_prediction = pipeline_lr.predict(new_patient_data)
predicted_class = "Diabetes" if new_patient_prediction[0] == 1 else "No Diabetes"
print(f"The predicted outcome for the new patient is: {predicted_class}")

# You can also get probability estimates if the model supports it (e.g., Logistic Regression)
if hasattr(pipeline_lr.named_steps['classifier'], 'predict_proba'):
    new_patient_proba = pipeline_lr.predict_proba(new_patient_data)
    print(f"Probability of No Diabetes: {new_patient_proba[0][0]:.4f}")
    print(f"Probability of Diabetes: {new_patient_proba[0][1]:.4f}")