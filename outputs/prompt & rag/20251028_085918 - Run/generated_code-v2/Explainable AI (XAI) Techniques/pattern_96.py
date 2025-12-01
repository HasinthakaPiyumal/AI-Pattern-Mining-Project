import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay

# 1. Data Handling: Generate a synthetic heart disease dataset
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(25, 80, n_samples),
        'cholesterol': np.random.randint(150, 300, n_samples),
        'blood_pressure': np.random.randint(90, 180, n_samples),
        'max_heart_rate': np.random.randint(120, 200, n_samples),
        'chest_pain_type': np.random.randint(0, 4, n_samples),
        'exercise_angina': np.random.randint(0, 2, n_samples),
    })

    # Create a synthetic 'heart_disease' target variable with some correlation
    heart_disease_prob = (
        0.02 * data['age'] 
        + 0.01 * data['cholesterol'] 
        + 0.005 * data['blood_pressure'] 
        - 0.01 * data['max_heart_rate'] 
        + 0.1 * data['chest_pain_type'] 
        + 0.2 * data['exercise_angina'] 
        + np.random.normal(0, 1, n_samples) * 5 # Add noise
    )
    data['heart_disease'] = (heart_disease_prob > np.percentile(heart_disease_prob, 70)).astype(int)
    return data

heart_data = generate_synthetic_data()

X = heart_data.drop('heart_disease', axis=1)
y = heart_data['heart_disease']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Machine Learning Model: Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model accuracy on test set: {model.score(X_test, y_test):.2f}")

# 3. Interpretability (Partial Dependence Plots - PDPs)
# 4. Visualization

# One-way Partial Dependence Plots
features_for_one_way_pdp = ['age', 'cholesterol', 'max_heart_rate']
fig, ax = plt.subplots(figsize=(15, 5), ncols=len(features_for_one_way_pdp))

for i, feature in enumerate(features_for_one_way_pdp):
    display = PartialDependenceDisplay.from_estimator(
        model, X_train, [feature], 
        kind='average', ax=ax[i], 
        feature_names=X.columns.tolist(), 
        response_method='predict_proba',
        random_state=42
    )
    ax[i].set_title(f'PDP for {feature}')
    ax[i].set_ylabel('Predicted Probability of Heart Disease')

plt.tight_layout()
plt.show()

print("\n--- Explanations for One-way PDPs ---")
print("Each plot shows the average predicted probability of heart disease as a single feature changes, holding other features constant at their average values. It helps understand the global trend of a feature's influence.")

# Two-way Partial Dependence Plot
features_for_two_way_pdp = [('age', 'cholesterol')]
fig, ax = plt.subplots(figsize=(8, 6))

display = PartialDependenceDisplay.from_estimator(
    model, X_train, features_for_two_way_pdp, 
    kind='contour', ax=ax, 
    feature_names=X.columns.tolist(), 
    response_method='predict_proba',
    random_state=42
)
ax.set_title('2D PDP for Age and Cholesterol')
ax.set_ylabel('Cholesterol')
ax.set_xlabel('Age')
plt.tight_layout()
plt.show()

print("\n--- Explanation for Two-way PDP ---")
print("This 2D contour plot shows how the average predicted probability of heart disease changes as 'age' and 'cholesterol' vary simultaneously. It can reveal interactions or combined effects of these two features on the prediction.")