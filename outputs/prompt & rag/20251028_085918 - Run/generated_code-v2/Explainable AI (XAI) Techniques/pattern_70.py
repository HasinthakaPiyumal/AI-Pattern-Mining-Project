import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

def generate_credit_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'loan_amount': np.random.randint(1000, 50000, n_samples),
        'interest_rate': np.random.uniform(5.0, 20.0, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'employment_years': np.random.randint(0, 30, n_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.6, n_samples),
        'num_credit_lines': np.random.randint(1, 15, n_samples),
        'loan_term': np.random.choice([36, 60], n_samples)
    }
    df = pd.DataFrame(data)
    
    # Simulate target variable (loan default)
    # Simulating default based on a combination of features
    df['default'] = 0
    df.loc[(df['credit_score'] < 600) & (df['debt_to_income_ratio'] > 0.4), 'default'] = 1
    df.loc[(df['interest_rate'] > 15) & (df['income'] < 50000), 'default'] = 1
    df.loc[(df['loan_amount'] > 30000) & (df['employment_years'] < 2) & (df['credit_score'] < 650), 'default'] = 1
    df.loc[np.random.rand(n_samples) < 0.05, 'default'] = 1 # Introduce some random defaults

    return df

if __name__ == "__main__":
    # 1. Data Layer: Generate synthetic credit risk data
    print("Generating synthetic credit risk data...")
    df = generate_credit_data(n_samples=2000)
    X = df.drop('default', axis=1)
    y = df['default']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Dataset shape: {df.shape}")
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # 2. Model Training Layer: Train a Black-Box Model (RandomForestClassifier)
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 3. Permutation Feature Importance Layer:
    # Calculate baseline performance
    baseline_predictions = model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    print(f"\nBaseline Model Accuracy on Test Set: {baseline_accuracy:.4f}")

    print("Calculating Permutation Feature Importance...")
    # Use sklearn's permutation_importance
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='accuracy')

    # 4. Reporting and Visualization Layer:
    # Create a DataFrame for better display
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance_Mean': result.importances_mean,
        'Importance_Std': result.importances_std
    })

    # Sort by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance_Mean', ascending=False)

    print("\nPermutation Feature Importance Results (ranked by mean importance):")
    print(feature_importance_df.to_string(index=False))

    print("\nInterpretation: A higher 'Importance_Mean' indicates that shuffling the feature's values led to a greater drop in model accuracy, suggesting that the model relies heavily on that feature for its predictions.")
