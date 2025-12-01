import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay

# 1. Data Simulation and Preprocessing
np.random.seed(42)

num_samples = 1000

income = np.random.normal(loc=60000, scale=20000, size=num_samples)
credit_score = np.random.normal(loc=700, scale=50, size=num_samples)
debt_to_income_ratio = np.random.normal(loc=0.3, scale=0.1, size=num_samples)
loan_amount = np.random.normal(loc=15000, scale=7000, size=num_samples)

# Introduce some correlation for loan default
# Higher income, higher credit score -> lower default probability
# Higher debt_to_income_ratio, higher loan_amount -> higher default probability
loan_default_prob = (
    0.4
    - (income / 200000)
    + (credit_score / 1000)
    + (debt_to_income_ratio * 0.5)
    + (loan_amount / 50000)
    + np.random.normal(loc=0, scale=0.1, size=num_samples)
)
loan_default = (loan_default_prob > 0.5).astype(int)

df = pd.DataFrame({
    "income": income,
    "credit_score": credit_score,
    "debt_to_income_ratio": debt_to_income_ratio,
    "loan_amount": loan_amount,
    "loan_default": loan_default,
})

# Ensure features are non-negative and within reasonable bounds
df["income"] = df["income"].apply(lambda x: max(10000, x))
df["credit_score"] = df["credit_score"].apply(lambda x: min(max(300, x), 850))
df["debt_to_income_ratio"] = df["debt_to_income_ratio"].apply(lambda x: min(max(0.05, x), 0.7))
df["loan_amount"] = df["loan_amount"].apply(lambda x: max(1000, x))

X = df.drop("loan_default", axis=1)
y = df["loan_default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Black-box Machine Learning Model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 3. Partial Dependence Plot (PDP) Computation and Visualization
features_to_plot_1d = ["income", "credit_score", "debt_to_income_ratio"]
features_to_plot_2d = [("income", "credit_score")]

# 1D PDPs
fig, axes = plt.subplots(ncols=len(features_to_plot_1d), figsize=(18, 5))
fig.suptitle('Partial Dependence Plots (1D) for Loan Default Prediction', fontsize=16)
for i, feature in enumerate(features_to_plot_1d):
    display = PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=X_train,
        features=[feature],
        target=1,  # Probability of loan_default=1
        ax=axes[i],
        line_kw={'color': 'blue', 'alpha': 0.8, 'lw': 2},
        pd_line_kw={'color': 'green', 'linestyle': '--'}
    )
    axes[i].set_title(f'PDP for {feature}')
    axes[i].set_ylabel('Partial Dependence' if i == 0 else '')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('pdp_1d_loan_default.png')
plt.close()

# 2D PDPs
fig_2d, ax_2d = plt.subplots(figsize=(10, 8))
fig_2d.suptitle('Partial Dependence Plot (2D) for Loan Default Prediction', fontsize=16)

display_2d = PartialDependenceDisplay.from_estimator(
    estimator=model,
    X=X_train,
    features=features_to_plot_2d,
    target=1,
    ax=ax_2d,
    grid_resolution=50
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('pdp_2d_income_credit_score.png')
plt.close()