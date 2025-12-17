import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Synthetic Data Generation (for demonstration)
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(30, 80, n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'cholesterol': np.random.randint(150, 300, n_samples),
    'blood_pressure_systolic': np.random.randint(100, 180, n_samples),
    'blood_pressure_diastolic': np.random.randint(60, 110, n_samples),
    'bmi': np.random.uniform(18.0, 35.0, n_samples),
    'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'diabetes': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'cardio_disease': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
}
df = pd.DataFrame(data)

# 2. Data Preprocessing
X = df.drop('cardio_disease', axis=1)
y = df['cardio_disease']

numerical_features = ['age', 'cholesterol', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'bmi']
categorical_features = ['gender', 'smoking', 'diabetes']

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Model Training and Evaluation

models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
    'Support Vector Machine': SVC(random_state=42, probability=True),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'MLP Classifier': MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(100, 50), early_stopping=True)
}

results = {}

for name, model in models.items():
    print(f"\n--- Training and Evaluating {name} ---")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0] * len(y_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    results[name] = {'model': pipeline, 'accuracy': accuracy, 'roc_auc': roc_auc}

# 4. Model Selection (Conceptual - here we just iterate and evaluate)
# In a real scenario, you would choose the best performing model based on specific metrics.

# 5. Prediction Example (using Logistic Regression pipeline as an example)
print("\n--- Example Prediction using Logistic Regression ---")
# Assume Logistic Regression was chosen as the best model for this example
logistic_regression_pipeline = results['Logistic Regression']['model']

new_patient_data = pd.DataFrame([{
    'age': 55,
    'gender': 'Female',
    'cholesterol': 220,
    'blood_pressure_systolic': 130,
    'blood_pressure_diastolic': 85,
    'bmi': 28.5,
    'smoking': 0,
    'diabetes': 0
}])

prediction = logistic_regression_pipeline.predict(new_patient_data)[0]
prediction_proba = logistic_regression_pipeline.predict_proba(new_patient_data)[0, 1]

print(f"New patient data:\n{new_patient_data}")
print(f"Predicted Cardiovascular Disease: {'Yes' if prediction == 1 else 'No'}")
print(f"Probability of Cardiovascular Disease: {prediction_proba:.4f}")