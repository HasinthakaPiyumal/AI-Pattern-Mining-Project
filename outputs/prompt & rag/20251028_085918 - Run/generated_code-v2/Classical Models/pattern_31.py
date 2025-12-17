import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

# 1. Data Ingestion (Simulated Synthetic Data)
np.random.seed(42)
n_samples = 1000

data = {
    'Age': np.random.randint(20, 70, n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'BMI': np.random.normal(25, 5, n_samples),
    'BloodPressure_Systolic': np.random.randint(100, 180, n_samples),
    'BloodPressure_Diastolic': np.random.randint(60, 120, n_samples),
    'Cholesterol': np.random.randint(150, 250, n_samples),
    'Glucose': np.random.randint(70, 200, n_samples),
    'Smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'ExerciseHoursWeek': np.random.uniform(0, 10, n_samples),
    'FamilyHistory_Diabetes': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'Target_Disease': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]) # 0: No Disease, 1: Disease
}
df = pd.DataFrame(data)

# Introduce some missing values for demonstration
for col in ['BMI', 'Cholesterol', 'Glucose']:
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, col] = np.nan

X = df.drop('Target_Disease', axis=1)
y = df['Target_Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Data Preprocessing Module

# Define numerical and categorical features
numerical_features = ['Age', 'BMI', 'BloodPressure_Systolic', 'BloodPressure_Diastolic', 'Cholesterol', 'Glucose', 'ExerciseHoursWeek']
categorical_features = ['Gender', 'Smoking', 'FamilyHistory_Diabetes']

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

# 3. Model Training Module and 4. Model Evaluation Module
models = {
    'LogisticRegression': {
        'model': LogisticRegression(solver='liblinear', random_state=42),
        'params': {
            'classifier__C': [0.1, 1.0, 10.0]
        }
    },
    'SVC': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['linear', 'rbf']
        }
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'classifier__max_depth': [5, 10, 15],
            'classifier__min_samples_split': [2, 5, 10]
        }
    }
}

best_model = None
best_score = -1
model_performance = {}

for model_name, config in models.items():
    print(f"\nTraining {model_name}...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', config['model'])
    ])

    grid_search = GridSearchCV(pipeline, config['params'], cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)
    y_proba = grid_search.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    model_performance[model_name] = {
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }

    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    if roc_auc > best_score:
        best_score = roc_auc
        best_model = grid_search.best_estimator_

print("\n--- Model Performance Summary ---")
for model_name, metrics in model_performance.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

if best_model:
    print(f"\nBest performing model (based on ROC-AUC) is: {best_model.named_steps['classifier'].__class__.__name__}")
    # 5. Model Persistence
    joblib.dump(best_model, 'best_disease_prediction_model.joblib')
    print("Best model saved as 'best_disease_prediction_model.joblib'")

    # Example of loading and making a prediction
    loaded_model = joblib.load('best_disease_prediction_model.joblib')
    print("\nLoaded model for example prediction.")
    # Create a dummy new patient record for prediction
    new_patient_data = pd.DataFrame({
        'Age': [55],
        'Gender': ['Male'],
        'BMI': [32.1],
        'BloodPressure_Systolic': [140],
        'BloodPressure_Diastolic': [90],
        'Cholesterol': [210],
        'Glucose': [130],
        'Smoking': [1],
        'ExerciseHoursWeek': [2.5],
        'FamilyHistory_Diabetes': [1]
    })
    prediction = loaded_model.predict(new_patient_data)
    prediction_proba = loaded_model.predict_proba(new_patient_data)[:, 1]
    print(f"New patient prediction: {'Disease' if prediction[0] == 1 else 'No Disease'}")
    print(f"Probability of Disease: {prediction_proba[0]:.4f}")
else:
    print("No best model found.")