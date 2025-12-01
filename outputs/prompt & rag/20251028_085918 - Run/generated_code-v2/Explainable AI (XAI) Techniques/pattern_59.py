import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Simulate Data (as no CSV is provided)
np.random.seed(42)
n_samples = 1000
data = {
    'age': np.random.randint(20, 70, n_samples),
    'bmi': np.random.uniform(18, 40, n_samples),
    'glucose': np.random.uniform(70, 200, n_samples),
    'blood_pressure': np.random.uniform(60, 180, n_samples),
    'insulin': np.random.uniform(10, 800, n_samples),
    'diabetes': np.random.randint(0, 2, n_samples) 
}
df = pd.DataFrame(data)

# Introduce some correlation for diabetes
df.loc[df['glucose'] > 140, 'diabetes'] = 1
df.loc[df['bmi'] > 30, 'diabetes'] = 1
df.loc[(df['glucose'] > 120) & (df['bmi'] > 28) & (df['age'] > 45), 'diabetes'] = 1

X = df.drop('diabetes', axis=1)
y = df['diabetes']

# 2. Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training

# Logistic Regression
log_reg_model = LogisticRegression(solver='liblinear', random_state=42)
log_reg_model.fit(X_train_scaled, y_train)

# Decision Tree
dec_tree_model = DecisionTreeClassifier(max_depth=5, random_state=42) 
dec_tree_model.fit(X_train, y_train) 

# 4. Model Evaluation
print("--- Logistic Regression Model Evaluation ---")
y_pred_log_reg = log_reg_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_log_reg))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))

print("\n--- Decision Tree Model Evaluation ---")
y_pred_dec_tree = dec_tree_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_dec_tree):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_dec_tree))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dec_tree))

# 5. Interpretation Layer
print("\n--- Logistic Regression Feature Importance ---")
feature_names = X.columns
coefficients = log_reg_model.coef_[0]
for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
    print(f"Feature '{name}': Coefficient = {coef:.4f}")

print("\nExplanation: Positive coefficients indicate that an increase in the feature value is associated with a higher likelihood of diabetes, while negative coefficients indicate a lower likelihood (holding other features constant).")

print("\n--- Decision Tree Rules ---")
r = export_text(dec_tree_model, feature_names=list(feature_names))
print(r)

print("\nExplanation: The Decision Tree provides a set of 'if-then-else' rules that explain how the model arrives at a diagnosis. Each path from the root to a leaf node represents a distinct decision rule.")