import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Data Simulation/Acquisition Module
def simulate_patient_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'BMI': np.random.normal(25, 5, n_samples),
        'Smoker': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'BloodPressure': np.random.normal(120, 15, n_samples),
        'Cholesterol': np.random.normal(200, 30, n_samples),
        'FamilyHistory': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'RiskCategory': np.random.choice(['Low', 'High'], n_samples, p=[0.65, 0.35]) # Target variable
    }
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    for col in ['BMI', 'BloodPressure', 'Cholesterol']:
        missing_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
        df.loc[missing_indices, col] = np.nan
        
    return df

df = simulate_patient_data()

X = df.drop('RiskCategory', axis=1)
y = df['RiskCategory']

# 2. Data Preprocessing Module
# Define categorical and numerical features
numerical_features = ['Age', 'BMI', 'BloodPressure', 'Cholesterol']
categorical_features = ['Gender', 'Smoker', 'FamilyHistory']

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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Model Training Module (Decision Tree as primary example)
# Create a full pipeline with preprocessor and classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42)) # Limiting depth for interpretability
])

model_pipeline.fit(X_train, y_train)

# 4. Model Evaluation Module
y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1] # Probability of 'High' risk

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='High')
recall = recall_score(y_test, y_pred, pos_label='High')
f1 = f1_score(y_test, y_pred, pos_label='High')
roc_auc = roc_auc_score(y_test, y_prob, pos_label='High')

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (High Risk): {precision:.4f}")
print(f"Recall (High Risk): {recall:.4f}")
print(f"F1-Score (High Risk): {f1:.4f}")
print(f"ROC AUC Score (High Risk): {roc_auc:.4f}")

# 5. Interpretability Module
print("\n--- Model Interpretability (Decision Tree) ---")

# Get feature names after one-hot encoding
encoded_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(encoded_feature_names)

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(model_pipeline.named_steps['classifier'], 
          feature_names=all_feature_names,  
          class_names=model_pipeline.classes_, 
          filled=True, 
          rounded=True,
          fontsize=8)
plt.title("Decision Tree for Patient Risk Assessment")
plt.savefig("decision_tree_visualization.png")
plt.show()

# Display Feature Importances
feature_importances = model_pipeline.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(importance_df)

# 6. Prediction and Deployment Module (Saving Model)
model_filename = "patient_risk_model.joblib"
joblib.dump(model_pipeline, model_filename)
print(f"\nModel saved to {model_filename}")

# 7. Load Model and Make New Predictions (Example)
loaded_model = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# Example new patient data
new_patient_data = pd.DataFrame({
    'Age': [55],
    'Gender': ['Male'],
    'BMI': [31.5],
    'Smoker': [1],
    'BloodPressure': [145],
    'Cholesterol': [230],
    'FamilyHistory': [1]
})

new_patient_risk = loaded_model.predict(new_patient_data)
new_patient_prob = loaded_model.predict_proba(new_patient_data)[:, 1] # Probability of 'High' risk

print(f"\nNew Patient Data:\n{new_patient_data}")
print(f"Predicted Risk Category: {new_patient_risk[0]}")
print(f"Probability of High Risk: {new_patient_prob[0]:.4f}")
