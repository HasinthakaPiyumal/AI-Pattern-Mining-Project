import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'credit_score': np.random.randint(300, 850, num_samples),
        'income': np.random.randint(30000, 150000, num_samples),
        'loan_amount': np.random.randint(5000, 100000, num_samples),
        'loan_term_years': np.random.randint(1, 30, num_samples),
        'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Retired'], num_samples, p=[0.6, 0.2, 0.1, 0.1]),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.6, num_samples),
        'has_collateral': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'previous_loan_default': np.random.choice([0, 1], num_samples, p=[0.95, 0.05])
    }
    df = pd.DataFrame(data)

    # Simple logic for loan approval (can be complex in real-world)
    # Higher credit score, higher income, lower debt-to-income, collateral -> higher chance of approval
    df['loan_approved'] = (df['credit_score'] > 650).astype(int) + \
                          (df['income'] > 50000).astype(int) + \
                          (df['debt_to_income_ratio'] < 0.4).astype(int) + \
                          (df['has_collateral'] == 1).astype(int) + \
                          (df['previous_loan_default'] == 0).astype(int)
    
    # Scale down probabilities for some cases to create denials
    df['loan_approved'] = np.where(df['loan_approved'] >= 3, 1, 0)
    df.loc[df['credit_score'] < 550, 'loan_approved'] = 0 # Definitely deny low credit scores
    df.loc[df['debt_to_income_ratio'] > 0.55, 'loan_approved'] = 0 # Definitely deny high DTI

    # Introduce some noise/randomness to make it less deterministic
    df['loan_approved'] = df.apply(lambda row: 1 if np.random.rand() < 0.1 else row['loan_approved'], axis=1) # Occasionally approve randomly
    df['loan_approved'] = df.apply(lambda row: 0 if np.random.rand() < 0.05 else row['loan_approved'], axis=1) # Occasionally deny randomly
    
    return df

if __name__ == "__main__":
    print("Generating synthetic loan application data...")
    df = generate_synthetic_data(num_samples=2000)
    print(f"Generated {len(df)} samples.")

    # Preprocessing
    le = LabelEncoder()
    df['employment_status_encoded'] = le.fit_transform(df['employment_status'])
    df_processed = df.drop(columns=['employment_status'])

    X = df_processed.drop('loan_approved', axis=1)
    y = df_processed['loan_approved']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train a RandomForestClassifier as the black-box model
    print("Training the black-box loan approval model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate (optional, for verification)
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.2f}")

    # Save the model and feature names
    joblib.dump(model, 'loan_approval_model.joblib')
    joblib.dump(X.columns.tolist(), 'feature_names.joblib')
    joblib.dump(le, 'employment_status_encoder.joblib')
    joblib.dump(df.columns.tolist(), 'full_data_columns.joblib') # Save all columns including original employment status
    
    print("Model, feature names, and encoder saved successfully.")
    print("Sample data head:\n", df.head())
    print("Sample data statistics:\n", df.describe())
    print("Value counts for loan_approved:\n", df['loan_approved'].value_counts())
