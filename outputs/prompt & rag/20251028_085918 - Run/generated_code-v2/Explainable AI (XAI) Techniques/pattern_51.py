import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay

# 1. Data Module: Simulate Credit Risk Dataset
def generate_credit_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'credit_score': np.random.randint(300, 850, num_samples),
        'income_level': np.random.randint(30000, 150000, num_samples),
        'loan_amount': np.random.randint(5000, 500000, num_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.6, num_samples),
        'employment_length': np.random.randint(0, 20, num_samples)
    }
    df = pd.DataFrame(data)

    # Simulate 'default' target variable
    # Higher debt-to-income, lower credit score, higher loan amount increase default risk
    default_probability = (0.6 * (1 - df['credit_score'] / 850)) + \
                            (0.3 * (df['debt_to_income_ratio'] / 0.6)) + \
                            (0.1 * (df['loan_amount'] / 500000))
    df['default'] = (default_probability + np.random.normal(0, 0.1, num_samples) > 0.7).astype(int)

    return df

# Generate the dataset
df_credit = generate_credit_data(num_samples=2000)

# Define features (X) and target (y)
X = df_credit.drop('default', axis=1)
y = df_credit['default']

# 2. Black-box ML Model Module: Train RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print(f"Model accuracy on test set: {model.score(X_test, y_test):.4f}")

# 3. PDP Generation and Visualization Module
# Features for which to plot PDPs
features_to_plot = ['credit_score', 'income_level', 'loan_amount', ('credit_score', 'debt_to_income_ratio')]

fig, ax = plt.subplots(figsize=(16, 8))

# Create PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(
    estimator=model,
    X=X_train,
    features=features_to_plot,
    feature_names=X.columns.tolist(),
    target=1,  # Plot PDP for the positive class (default = 1)
    grid_resolution=50,
    ax=ax
)

_ = fig.suptitle('Partial Dependence Plots for Credit Default Prediction')
_ = fig.tight_layout()

plt.show()
