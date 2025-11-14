#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import lime
import lime.lime_tabular

# --- 1. Synthetic Data Generation ---
print("1. Generating synthetic patient data...")
np.random.seed(42)
n_samples = 500

data = {
    'Age': np.random.randint(20, 80, n_samples),
    'Blood_Pressure': np.random.randint(90, 180, n_samples),
    'Cholesterol': np.random.randint(150, 300, n_samples),
    'Fever': np.random.uniform(98.0, 104.0, n_samples),
    'Sore_Throat': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'Fatigue': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    'Cough': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    'Diagnosis': np.random.randint(0, 2, n_samples) # 0: Healthy, 1: Disease X (e.g., Flu-like illness)
}
df = pd.DataFrame(data)

# Introduce some correlations for a more realistic (though simple) model
df.loc[df['Fever'] > 100.5, 'Diagnosis'] = np.random.choice([0, 1], sum(df['Fever'] > 100.5), p=[0.2, 0.8])
df.loc[df['Sore_Throat'] == 1, 'Diagnosis'] = np.random.choice([0, 1], sum(df['Sore_Throat'] == 1), p=[0.3, 0.7])
df.loc[(df['Fever'] > 100.5) & (df['Cough'] == 1), 'Diagnosis'] = 1
df.loc[df['Age'] > 60, 'Diagnosis'] = np.random.choice([0, 1], sum(df['Age'] > 60), p=[0.4, 0.6])

X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

feature_names = X.columns.tolist()
class_names = ['Healthy', 'Disease X']

# --- 2. Train a Simulated AI Diagnosis Model ---
print("2. Training a RandomForestClassifier model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model accuracy on test set: {model.score(X_test, y_test):.2f}")

# --- 3. Explanation Generation Module ---

# --- 3.1. Local Explanations (LIME) ---
print("3.1. Generating Local Explanations (LIME) for a specific patient...")
# Select a random patient from the test set for explanation
patient_idx = np.random.randint(0, len(X_test))
single_patient_data = X_test.iloc[patient_idx]
single_patient_true_diagnosis = y_test.iloc[patient_idx]
single_patient_prediction = model.predict(single_patient_data.to_frame().T)[0]

print(f"\nPatient {patient_idx} (True Diagnosis: {class_names[single_patient_true_diagnosis]}, Predicted Diagnosis: {class_names[single_patient_prediction]})\n")

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

explanation = explainer.explain_instance(
    data_row=single_patient_data.values,
    predict_fn=model.predict_proba,
    num_features=5
)

print("LIME Explanation for the selected patient:")
for feature, weight in explanation.as_list():
    print(f"  {feature}: {weight:.4f}")

fig_lime, ax_lime = plt.subplots(figsize=(10, 6))
explanation.as_pyplot_figure(ax=ax_lime)
ax_lime.set_title(f"LIME Explanation for Patient {patient_idx} (Predicted: {class_names[single_patient_prediction]})")
plt.tight_layout()
plt.show()


# --- 3.2. Global Explanations (Partial Dependence Plots & Permutation Feature Importance) ---
print("\n3.2. Generating Global Explanations...")

# Partial Dependence Plots (PDPs)
print("Generating Partial Dependence Plots (PDPs)...")
features_for_pdp = ['Age', 'Fever', 'Cholesterol', ('Age', 'Fever')] # Example features, can be adjusted

fig_pdp, ax_pdp = plt.subplots(figsize=(15, 7), ncols=3, nrows=1)
if not isinstance(ax_pdp, np.ndarray):
    ax_pdp = np.array([ax_pdp]) # Ensure ax_pdp is an array for consistent indexing

PartialDependenceDisplay.from_estimator(
    estimator=model,
    X=X_train,
    features=[feature_names.index(f) if isinstance(f, str) else tuple(feature_names.index(sf) for sf in f) for f in features_for_pdp[:3]],
    feature_names=feature_names,
    target=1, # Probability of 'Disease X'
    ax=ax_pdp[:3], # Assign to the first 3 subplots
    kind='average'
)
fig_pdp.suptitle('Partial Dependence Plots (Probability of Disease X)', y=1.02)
plt.tight_layout()
plt.show()

# Permutation Feature Importance
print("Generating Permutation Feature Importance...")
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()

fig_perm_imp, ax_perm_imp = plt.subplots(figsize=(12, 7))
sns.barplot(x=result.importances_mean[sorted_idx], y=np.array(feature_names)[sorted_idx], ax=ax_perm_imp)
ax_perm_imp.set_title("Permutation Feature Importance")
ax_perm_imp.set_xlabel("Mean Decrease in Accuracy")
plt.tight_layout()
plt.show()


# --- 3.3. Instance-Specific Conditional Expectations (ICE Plots) ---
print("\n3.3. Generating Instance-Specific Conditional Expectations (ICE Plots)...")

# Select a feature to plot ICE for
ice_feature = 'Fever'
ice_feature_idx = feature_names.index(ice_feature)

fig_ice, ax_ice = plt.subplots(figsize=(10, 6))

PartialDependenceDisplay.from_estimator(
    estimator=model,
    X=X_test, # Using test data for ICE plots
    features=[ice_feature_idx],
    feature_names=feature_names,
    target=1, # Probability of 'Disease X'
    kind='individual', # This is key for ICE plots
    ax=ax_ice,
    # pd_display_kwargs={'line_kw': {'alpha': 0.3}}
)
ax_ice.set_title(f"ICE Plots for {ice_feature} (Probability of Disease X)")
plt.tight_layout()
plt.show()

print("\nDemonstration complete. The plots show various aspects of model interpretability.")
print("For a full interactive application, Streamlit or Gradio would be used to wrap these functionalities.")
