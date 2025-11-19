import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

# 1. Data Simulation
np.random.seed(42)
num_patients = 1000

age = np.random.randint(20, 80, num_patients)
symptom_a = np.random.randint(0, 2, num_patients) # 0: no, 1: yes
symptom_b = np.random.randint(0, 2, num_patients)
test_result_x = np.random.rand(num_patients) * 100 # continuous score
test_result_y = np.random.rand(num_patients) * 50 # continuous score

# Simulate a disease based on a combination of features
disease_probability = (age * 0.01) + (symptom_a * 0.3) + (symptom_b * 0.2) + (test_result_x * 0.005) - (test_result_y * 0.01)
disease = (disease_probability + np.random.rand(num_patients) * 0.5 > 0.7).astype(int)

data = pd.DataFrame({
    'age': age,
    'symptom_a': symptom_a,
    'symptom_b': symptom_b,
    'test_result_x': test_result_x,
    'test_result_y': test_result_y,
    'disease': disease
})

X = data.drop('disease', axis=1)
y = data['disease']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Disease Prediction Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model accuracy on test set: {model.score(X_test, y_test):.2f}\n")

# 3. Local Model Interpretability (LIME)
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=['No Disease', 'Disease'],
    mode='classification'
)

# Choose an instance to explain (e.g., the first instance in the test set)
sample_instance_index = 0
sample_instance = X_test.iloc[sample_instance_index]

print(f"Explaining prediction for patient with features:\n{sample_instance}\n")

explanation = explainer.explain_instance(
    data_row=sample_instance.values,
    predict_fn=model.predict_proba,
    num_features=len(feature_names)
)

print("LIME Explanation for individual prediction:")
for feature, weight in explanation.as_list():
    print(f"  {feature}: {weight:.4f}")

# 4. Global Model Interpretability (Partial Dependence Plots)
print("\nGenerating Partial Dependence Plots...")

fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(model, X_train, features=feature_names, 
                                         feature_names=feature_names, target=1, ax=ax)
fig.suptitle('Partial Dependence Plots (Global Feature Importance)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\nClinical Decision Support System with Explainable AI - Demonstration Complete.")