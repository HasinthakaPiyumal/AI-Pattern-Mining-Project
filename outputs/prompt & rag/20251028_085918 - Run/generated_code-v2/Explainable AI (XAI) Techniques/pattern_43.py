import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1. Data Preparation Module: Generate synthetic patient data
np.random.seed(42)
n_samples = 1000

data = {
    "age": np.random.randint(20, 90, n_samples),
    "num_medications": np.random.randint(1, 15, n_samples),
    "num_procedures": np.random.randint(0, 7, n_samples),
    "time_in_hospital": np.random.randint(1, 20, n_samples),
    "diabetes": np.random.randint(0, 2, n_samples),
    "heart_disease": np.random.randint(0, 2, n_samples),
    "hypertension": np.random.randint(0, 2, n_samples),
    "emergency_visit_last_year": np.random.randint(0, 5, n_samples),
    "readmitted": np.random.randint(0, 2, n_samples) # Target variable: 0 or 1
}

df = pd.DataFrame(data)

# Introduce some correlation for 'readmitted' for a more realistic scenario
df['readmitted'] = (df['num_medications'] * 0.1 + df['time_in_hospital'] * 0.2 + df['age'] * 0.05 + np.random.randn(n_samples) * 0.5 > 2).astype(int)

X = df.drop("readmitted", axis=1)
y = df["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. Pre-trained Black-Box Model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 3. & 4. Permutation Module and Performance Evaluation Module

# Baseline performance
y_pred_baseline = model.predict_proba(X_test)[:, 1]
baseline_roc_auc = roc_auc_score(y_test, y_pred_baseline)

feature_importances = {}

for feature in X_test.columns:
    X_test_permuted = X_test.copy()
    X_test_permuted[feature] = np.random.permutation(X_test_permuted[feature])

    y_pred_permuted = model.predict_proba(X_test_permuted)[:, 1]
    permuted_roc_auc = roc_auc_score(y_test, y_pred_permuted)

    importance = baseline_roc_auc - permuted_roc_auc
    feature_importances[feature] = importance

# 5. Reporting Module for Feature Importance Scores
sorted_importance = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)

print("Permutation Feature Importances (ROC AUC drop):")
for feature, importance in sorted_importance:
    print(f"  {feature}: {importance:.4f}")