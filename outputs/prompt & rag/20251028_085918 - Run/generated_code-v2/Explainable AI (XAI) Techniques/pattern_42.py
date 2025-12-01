import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import plot_partial_dependence

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    credit_score = np.random.randint(300, 850, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    debt_to_income_ratio = np.random.normal(0.3, 0.1, n_samples)
    loan_amount = np.random.randint(5000, 100000, n_samples)

    # Introduce some correlation for credit risk
    credit_risk = (
        (credit_score < 600) * 0.4  # Low credit score -> higher risk
        + (income < 30000) * 0.3  # Low income -> higher risk
        + (debt_to_income_ratio > 0.4) * 0.2  # High DTI -> higher risk
        + (loan_amount > 50000) * 0.1 # High loan amount -> higher risk
        + np.random.rand(n_samples) * 0.2 # Random noise
    ) > 0.5
    credit_risk = credit_risk.astype(int)

    data = pd.DataFrame({
        'CreditScore': credit_score,
        'Income': income,
        'DebtToIncomeRatio': debt_to_income_ratio,
        'LoanAmount': loan_amount,
        'CreditRisk': credit_risk
    })
    return data


def train_black_box_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    return model


def main():
    # 1. Data Simulation Module
    print("Generating synthetic loan application data...")
    data = generate_synthetic_data()
    print("Data generated successfully. Sample data head:")
    print(data.head())

    X = data.drop('CreditRisk', axis=1)
    y = data['CreditRisk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Black-box Model Training Module
    print("\nTraining black-box RandomForestClassifier model...")
    model = train_black_box_model(X_train, y_train)
    print(f"Model trained. Accuracy on test set: {model.score(X_test, y_test):.4f}")

    # 3. Partial Dependence Plot (PDP) Generation Module
    print("\nGenerating Partial Dependence Plots for key features...")
    features_to_plot = ['CreditScore', 'Income', ('CreditScore', 'LoanAmount')]

    fig, ax = plt.subplots(figsize=(15, 7))
    plot_partial_dependence(
        estimator=model,
        X=X_test, # Using test data for plotting for robustness
        features=features_to_plot,
        feature_names=X.columns.tolist(),
        target=1, # Plotting dependence for the 'high risk' class (1)
        grid_resolution=50,
        ax=ax
    )
    fig.suptitle('Partial Dependence Plots for Credit Risk Model', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # 4. Explanation and Visualization Module
    print("\nDisplaying PDPs and providing explanations:")
    print("  - 'CreditScore': Shows how the model's predicted probability of high credit risk changes as credit score varies, averaging out other features. Expect to see a decrease in risk probability as credit score increases.")
    print("  - 'Income': Illustrates the model's average predicted risk across different income levels. Expect lower risk probability with higher income.")
    print("  - ('CreditScore', 'LoanAmount'): A 2D PDP showing the interaction effect of credit score and loan amount on the predicted credit risk.")
    print("These plots help understand the global average effect of features on the credit risk prediction, aiding in model interpretability and bias detection.")
    plt.show()

if __name__ == '__main__':
    main()