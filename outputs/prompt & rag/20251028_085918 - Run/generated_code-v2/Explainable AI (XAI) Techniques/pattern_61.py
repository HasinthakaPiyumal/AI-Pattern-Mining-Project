import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt

# 1. Data Simulation
np.random.seed(42)
num_patients = 1000

data = {
    'age': np.random.randint(18, 90, num_patients),
    'diagnosis_code': np.random.choice(['D01', 'D02', 'D03', 'D04', 'D05'], num_patients),
    'length_of_stay': np.random.randint(1, 30, num_patients),
    'num_previous_admissions': np.random.randint(0, 5, num_patients),
    'medication_count': np.random.randint(1, 15, num_patients),
    'comorbidity_score': np.random.randint(0, 10, num_patients),
}
df = pd.DataFrame(data)

# Create a target variable 'readmitted' with some correlation to features
df['readmitted'] = ((df['age'] > 60) * 0.3 + 
                    (df['num_previous_admissions'] > 1) * 0.4 + 
                    (df['length_of_stay'] > 15) * 0.2 + 
                    (df['comorbidity_score'] > 5) * 0.5 + 
                    np.random.rand(num_patients) * 0.5) > 0.7
df['readmitted'] = df['readmitted'].astype(int)

# 2. Preprocessing and Splitting Data

# Define categorical and numerical features
categorical_features = ['diagnosis_code']
numerical_features = ['age', 'length_of_stay', 'num_previous_admissions', 'medication_count', 'comorbidity_score']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

X = df.drop('readmitted', axis=1)
y = df['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Machine Learning Model & Pipeline

# Create a pipeline with preprocessing and classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# 4. Model Training and Evaluation
y_pred = model_pipeline.predict(X_test)
print(f"Model Accuracy on Test Set: {accuracy_score(y_test, y_pred):.4f}")
print(f"Model F1-Score on Test Set: {f1_score(y_test, y_pred):.4f}")
print("\n" + "="*30 + "\n")

# 5. Permutation Feature Importance (PFI)

# Get feature names after one-hot encoding
encoded_feature_names = model_pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(categorical_features).tolist()
all_feature_names = numerical_features + encoded_feature_names

perm = PermutationImportance(model_pipeline, random_state=42, scoring='f1')
perm.fit(X_test, y_test)

print("Permutation Feature Importance:\n")
eli5.show_weights(perm, feature_names=all_feature_names, top=len(all_feature_names), show_feature_values=True)

# 6. Output and Visualization (Matplotlib)

# Extract scores and feature names for plotting
feature_importance_scores = perm.feature_importances_
feature_names_sorted_indices = np.argsort(feature_importance_scores)[::-1]

sorted_feature_names = [all_feature_names[i] for i in feature_names_sorted_indices]
sorted_importance_scores = feature_importance_scores[feature_names_sorted_indices]

plt.figure(figsize=(12, 8))
plt.barh(sorted_feature_names, sorted_importance_scores)
plt.xlabel("Permutation Importance (F1-score drop)")
plt.ylabel("Feature")
plt.title("Global Feature Importance for Patient Readmission Prediction")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()