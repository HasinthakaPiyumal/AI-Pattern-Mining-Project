import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence

# 1. Data Layer: Simulate Credit Application Data
np.random.seed(42)
num_samples = 1000

data = {
    'credit_score': np.random.randint(300, 850, num_samples),
    'income': np.random.randint(30000, 150000, num_samples),
    'loan_amount': np.random.randint(5000, 100000, num_samples),
    'debt_to_income_ratio': np.random.uniform(0.1, 0.6, num_samples),
    'employment_years': np.random.randint(0, 20, num_samples),
    'has_dependents': np.random.randint(0, 2, num_samples),
    'default': np.random.randint(0, 2, num_samples) # Target variable
}
df = pd.DataFrame(data)

# Introduce some correlation for 'default'
df['default'] = df.apply(lambda row: 1 if (row['credit_score'] < 600 and row['debt_to_income_ratio'] > 0.4) or \
                                          (row['loan_amount'] > 70000 and row['income'] < 50000) else row['default'], axis=1)
df['default'] = df.apply(lambda row: 0 if (row['credit_score'] > 750 and row['income'] > 100000) else row['default'], axis=1)

X = df.drop('default', axis=1)
y = df['default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Machine Learning Model Layer: Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print(f"Model accuracy on test set: {model.score(X_test, y_test):.4f}")

# 3. Model Interpretability Layer (PDPs) & 4. Visualization Layer
# Features for which to plot PDPs
features_to_plot = ['credit_score', 'income', 'loan_amount', 'debt_to_income_ratio', ('credit_score', 'debt_to_income_ratio')]

# Create a figure and a set of subplots for the PDPs
fig, ax = plt.subplots(figsize=(15, 10))

# Plot Partial Dependence Plots
plot_partial_dependence(
    estimator=model,
    X=X_train,
    features=features_to_plot,
    target=1, # Probability of default (class 1)
    feature_names=X.columns.tolist(),
    grid_resolution=50,
    ax=ax,
    kind='both' # 'average' for 1D, 'individual' for ICE, 'both' for both
)

fig.suptitle('Partial Dependence Plots for Credit Risk Model', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
plt.show()

# Interpretation (Console Output)
print("\n--- Interpretation of Partial Dependence Plots ---")
print("These plots show the average marginal effect of one or two features on the predicted probability of loan default, holding all other features constant.")
print("\n- **Credit Score:** As credit score increases, the probability of default tends to decrease.")
print("- **Income:** Higher income generally leads to a lower probability of default.")
print("- **Loan Amount:** Larger loan amounts might correlate with a slightly higher default probability, especially for lower income ranges (not explicitly shown in 1D but can be inferred from context or 2D PDPs).")
print("- **Debt-to-Income Ratio:** A higher debt-to-income ratio is associated with an increased probability of default.")
print("- **Credit Score vs. Debt-to-Income Ratio (2D PDP):** This plot reveals interaction effects. For instance, individuals with low credit scores AND high debt-to-income ratios have a significantly higher default probability, while those with high credit scores and low debt-to-income ratios have a very low default probability. The combination of these two factors can be more indicative than each alone.")