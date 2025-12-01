import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Data Generation
np.random.seed(42)
num_samples = 1000

data = {
    'age': np.random.randint(30, 80, num_samples),
    'cholesterol': np.random.randint(150, 300, num_samples),
    'blood_pressure': np.random.randint(90, 180, num_samples),
    'bmi': np.random.uniform(18.0, 35.0, num_samples),
    'smoking': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
    'gender': np.random.choice([0, 1], num_samples, p=[0.5, 0.5]) # 0 for female, 1 for male
}
df = pd.DataFrame(data)

# Simulate CVD risk - a simplified linear combination with some noise
df['cvd_risk'] = (0.05 * df['age'] + 0.02 * df['cholesterol'] + 0.03 * df['blood_pressure'] +
                  0.1 * df['bmi'] + 0.5 * df['smoking'] + 0.2 * df['gender'] +
                  np.random.normal(0, 0.5, num_samples))

# Convert to binary classification problem (e.g., risk > threshold)
risk_threshold = df['cvd_risk'].median()
df['cvd_positive'] = (df['cvd_risk'] > risk_threshold).astype(int)

X = df[['age', 'cholesterol', 'blood_pressure', 'bmi', 'smoking', 'gender']]
y = df['cvd_positive']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Data Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrame for feature naming consistency
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\n--- Interpretable Cardiovascular Disease Risk Prediction System ---")

# 3. Model Selection & Training - Logistic Regression
print("\n--- Training Logistic Regression Model ---")
log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train_scaled_df, y_train)

# 4. Model Evaluation - Logistic Regression
y_pred_lr = log_reg_model.predict(X_test_scaled_df)
y_prob_lr = log_reg_model.predict_proba(X_test_scaled_df)[:, 1]

accuracy_lr = accuracy_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_prob_lr)

print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")
print(f"Logistic Regression ROC AUC: {roc_auc_lr:.4f}")

# 6. Interpretability Module - Logistic Regression
print("\n--- Logistic Regression Model Interpretability (Coefficients) ---")
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': log_reg_model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)
print(coefficients)
print("\nInterpretation: A positive coefficient indicates that as the feature value increases, the log-odds of CVD risk increase. A negative coefficient indicates the opposite.")

# 3. Model Selection & Training - Decision Tree Classifier
print("\n--- Training Decision Tree Classifier Model ---")
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42) # Limiting depth for interpretability
dt_model.fit(X_train_scaled_df, y_train)

# 4. Model Evaluation - Decision Tree Classifier
y_pred_dt = dt_model.predict(X_test_scaled_df)
y_prob_dt = dt_model.predict_proba(X_test_scaled_df)[:, 1]

accuracy_dt = accuracy_score(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, y_prob_dt)

print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")
print(f"Decision Tree ROC AUC: {roc_auc_dt:.4f}")

# 6. Interpretability Module - Decision Tree Classifier
print("\n--- Decision Tree Model Interpretability ---")
print("Feature Importances:")
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(feature_importances)

print("\nText-based Decision Tree Rules (max_depth=4 for readability):\n")
r = export_text(dt_model, feature_names=list(X_train.columns))
print(r)
print("\nInterpretation: The tree shows a series of IF-THEN rules. Each path from the root to a leaf represents a decision rule, and the leaf indicates the predicted class (CVD positive or negative) and the proportion of samples belonging to that class.")
