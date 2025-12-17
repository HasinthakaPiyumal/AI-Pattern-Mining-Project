import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

# --- 1. Simulate Data ---
# In a real scenario, this would be loaded from an EHR database or CSV.
np.random.seed(42)
num_patients = 1000
data = {
    'age': np.random.randint(20, 90, num_patients),
    'num_diagnoses': np.random.randint(1, 15, num_patients),
    'num_prior_admissions': np.random.randint(0, 5, num_patients),
    'length_of_stay_prev': np.random.randint(1, 30, num_patients),
    'gender': np.random.choice(['Male', 'Female'], num_patients),
    'primary_diagnosis': np.random.choice(['Cardiovascular', 'Respiratory', 'Diabetes', 'Cancer', 'Injury'], num_patients),
    'medication_count': np.random.randint(1, 10, num_patients),
    'readmitted': np.random.randint(0, 2, num_patients) # 0: No, 1: Yes
}
df = pd.DataFrame(data)

# Introduce some missing values for demonstration
for col in ['num_diagnoses', 'length_of_stay_prev']:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

print("--- Raw Data Sample ---")
print(df.head())
print(f"Missing values before imputation:\n{df.isnull().sum()}")

# --- 2. Data Preprocessing and Feature Engineering Module ---

# Define numerical and categorical features
numerical_features = ['age', 'num_diagnoses', 'num_prior_admissions', 'length_of_stay_prev', 'medication_count']
categorical_features = ['gender', 'primary_diagnosis']

# Create a preprocessing pipeline
# Numerical transformer: Impute missing values with median, then scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical transformer: One-hot encode
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ], 
    remainder='passthrough' # Keep other columns (if any) that are not processed
)

# Feature Engineering (example: comorbidity score)
def calculate_comorbidity_score(df_patient):
    score = 0
    if 'Cardiovascular' in df_patient['primary_diagnosis_encoded'] or 'Diabetes' in df_patient['primary_diagnosis_encoded']:
        score += 1
    if df_patient['num_diagnoses'] > 5:
        score += 1
    return score

# Prepare data for modeling
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Data Splitting Complete ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# --- 3. Model Training Module ---

# Define models
models = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Store trained models and their performance
trained_models = {}
model_performance = {}

print("\n--- Model Training and Evaluation ---")

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create a full pipeline including preprocessing and the model
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
    
    model_pipeline.fit(X_train, y_train)
    trained_models[name] = model_pipeline
    
    # --- 4. Model Evaluation Module ---
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    model_performance[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    print(f"{name} Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")


# Example of Hyperparameter Tuning (using GridSearchCV on RandomForest)
print("\n--- Hyperparameter Tuning (Random Forest) ---")
param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_leaf': [1, 2, 4]
}

rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best parameters for Random Forest: {grid_search.best_params_}")
print(f"Best ROC-AUC score (Random Forest): {grid_search.best_score_:.4f}")

best_rf_model = grid_search.best_estimator_

# Evaluate the best Random Forest model
y_pred_best_rf = best_rf_model.predict(X_test)
y_proba_best_rf = best_rf_model.predict_proba(X_test)[:, 1]

accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
roc_auc_best_rf = roc_auc_score(y_test, y_proba_best_rf)

print(f"Best Random Forest Model Test Accuracy: {accuracy_best_rf:.4f}")
print(f"Best Random Forest Model Test ROC-AUC: {roc_auc_best_rf:.4f}")


# --- 5. Prediction and Inference Module ---

# Select the best performing model based on ROC-AUC (or another preferred metric)
best_model_name = max(model_performance, key=lambda name: model_performance[name]['roc_auc'])
final_model = trained_models[best_model_name]
print(f"\nSelected best model: {best_model_name} (ROC-AUC: {model_performance[best_model_name]['roc_auc']:.4f})")

# Save the final model
model_filename = f'{best_model_name.replace(" ", "_").lower()}_readmission_predictor.joblib'
joblib.dump(final_model, model_filename)
print(f"Model saved to {model_filename}")

# Load the model back (for demonstration of inference)
loaded_model = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# Simulate new patient data for prediction
new_patient_data = pd.DataFrame({
    'age': [75, 45, 60],
    'num_diagnoses': [8, 2, 6],
    'num_prior_admissions': [2, 0, 1],
    'length_of_stay_prev': [15, 3, 7],
    'gender': ['Female', 'Male', 'Female'],
    'primary_diagnosis': ['Cardiovascular', 'Injury', 'Diabetes'],
    'medication_count': [6, 2, 4]
})

print("\n--- New Patient Data for Prediction ---")
print(new_patient_data)

# Make predictions on new patient data
new_patient_predictions = loaded_model.predict(new_patient_data)
new_patient_probabilities = loaded_model.predict_proba(new_patient_data)[:, 1]

print("\n--- Predictions for New Patients ---")
for i, (pred, proba) in enumerate(zip(new_patient_predictions, new_patient_probabilities)):
    status = "readmitted" if pred == 1 else "not readmitted"
    print(f"Patient {i+1}: Predicted {status} (Probability: {proba:.4f})")
