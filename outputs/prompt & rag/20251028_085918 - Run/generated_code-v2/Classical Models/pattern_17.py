import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

# 1. Data Simulation
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(25, 85, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'Creatinine': np.random.uniform(0.7, 3.0, num_samples), # mg/dL
        'GFR': np.random.uniform(15, 120, num_samples), # mL/min/1.73m^2
        'Hypertension': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'Diabetes': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        'Albuminuria': np.random.choice([0, 1, 2, 3], num_samples, p=[0.6, 0.2, 0.1, 0.1]), # 0: None, 1: Micro, 2: Macro
        'Family_History_CKD': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
        'Smoker': np.random.choice([0, 1], num_samples, p=[0.75, 0.25]),
        'CKD_Progression': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]) # Target variable: 0=No Progression, 1=Progression
    }
    df = pd.DataFrame(data)

    # Introduce some correlations for realism
    df.loc[df['Creatinine'] > 1.5, 'GFR'] = df.loc[df['Creatinine'] > 1.5, 'GFR'] * np.random.uniform(0.5, 0.8, (df['Creatinine'] > 1.5).sum())
    df.loc[df['GFR'] < 60, 'CKD_Progression'] = np.random.choice([0, 1], (df['GFR'] < 60).sum(), p=[0.3, 0.7])
    df.loc[df['Age'] > 60, 'CKD_Progression'] = np.random.choice([0, 1], (df['Age'] > 60).sum(), p=[0.5, 0.5])
    df.loc[(df['Hypertension'] == 1) | (df['Diabetes'] == 1), 'CKD_Progression'] = np.random.choice([0, 1], ((df['Hypertension'] == 1) | (df['Diabetes'] == 1)).sum(), p=[0.4, 0.6])

    # Introduce some missing values
    for col in ['Creatinine', 'GFR', 'Albuminuria']:
        df.loc[df.sample(frac=0.05).index, col] = np.nan

    return df

df = generate_synthetic_data()

# 2. Data Ingestion and Preprocessing
X = df.drop('CKD_Progression', axis=1)
y = df['CKD_Progression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define numerical and categorical features
numerical_features = ['Age', 'Creatinine', 'GFR']
categorical_features = ['Gender', 'Hypertension', 'Diabetes', 'Albuminuria', 'Family_History_CKD', 'Smoker']

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

# 3. Model Training
models = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, solver='liblinear'),
        'params': {
            'classifier__C': [0.1, 1.0, 10.0]
        }
    },
    'SVC': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['linear', 'rbf']
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1]
        }
    }
}

best_model = None
best_score = -1
best_model_name = ""
best_pipeline = None

for model_name, config in models.items():
    print(f"\nTraining {model_name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', config['model'])])

    grid_search = GridSearchCV(pipeline, config['params'], cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best ROC AUC score for {model_name} on validation: {grid_search.best_score_:.4f}")

    y_pred = grid_search.best_estimator_.predict(X_test)
    y_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]

    current_roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC on test set for {model_name}: {current_roc_auc:.4f}")
    print(f"Accuracy on test set for {model_name}: {accuracy_score(y_test, y_pred):.4f}")

    if current_roc_auc > best_score:
        best_score = current_roc_auc
        best_model = grid_search.best_estimator_
        best_model_name = model_name
        best_pipeline = grid_search.best_estimator_

print(f"\nBest performing model: {best_model_name} with ROC AUC: {best_score:.4f} on test set.")

# 4. Model Evaluation (Detailed for the best model)
print(f"\nDetailed Evaluation for {best_model_name}:")
y_pred_best = best_pipeline.predict(X_test)
y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_best):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_best):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_best):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_best):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# 5. Model Persistence
model_filename = 'ckd_progression_model.joblib'
joblib.dump(best_pipeline, model_filename)
print(f"\nBest model and preprocessor saved as {model_filename}")

# 6. Prediction/Inference Module
def predict_ckd_progression(new_patient_data, model_path='ckd_progression_model.joblib'):
    loaded_pipeline = joblib.load(model_path)

    # Convert new_patient_data to DataFrame, ensuring consistent column order
    new_df = pd.DataFrame([new_patient_data])

    # The loaded pipeline includes the preprocessor, so just pass the raw data
    prediction_proba = loaded_pipeline.predict_proba(new_df)[:, 1]
    prediction_class = loaded_pipeline.predict(new_df)[0]

    return {
        'predicted_progression': int(prediction_class),
        'probability_of_progression': float(prediction_proba[0])
    }

print("\n--- Demonstrating Inference ---")
# Example new patient data
new_patient = {
    'Age': 70,
    'Gender': 'Female',
    'Creatinine': 2.1,
    'GFR': 45.0,
    'Hypertension': 1,
    'Diabetes': 1,
    'Albuminuria': 2,
    'Family_History_CKD': 1,
    'Smoker': 0
}

inference_result = predict_ckd_progression(new_patient)
print(f"New Patient Data: {new_patient}")
print(f"Inference Result: {inference_result}")

new_patient_low_risk = {
    'Age': 35,
    'Gender': 'Male',
    'Creatinine': 0.9,
    'GFR': 95.0,
    'Hypertension': 0,
    'Diabetes': 0,
    'Albuminuria': 0,
    'Family_History_CKD': 0,
    'Smoker': 0
}

inference_result_low_risk = predict_ckd_progression(new_patient_low_risk)
print(f"\nNew Patient Data (Low Risk): {new_patient_low_risk}")
print(f"Inference Result (Low Risk): {inference_result_low_risk}")
