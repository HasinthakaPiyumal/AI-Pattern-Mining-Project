import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from alibi.explainers import Counterfactual

def generate_synthetic_loan_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "credit_score": np.random.randint(300, 850, num_samples),
        "debt_to_income_ratio": np.random.uniform(0.1, 0.6, num_samples),
        "income": np.random.randint(30000, 150000, num_samples),
        "loan_amount": np.random.randint(5000, 100000, num_samples),
        "employment_years": np.random.randint(0, 20, num_samples),
    }
    df = pd.DataFrame(data)
    
    # Simple rule for loan approval (can be more complex)
    df["approved"] = ((df["credit_score"] > 650) & 
                      (df["debt_to_income_ratio"] < 0.4) & 
                      (df["income"] > 50000) & 
                      (df["loan_amount"] < 0.7 * df["income"]) & 
                      (df["employment_years"] >= 2)).astype(int)
    
    return df

def train_black_box_model(df):
    X = df.drop("approved", axis=1)
    y = df["approved"]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

def predict_loan_decision(model, features):
    return model.predict(np.array(features).reshape(1, -1))[0]

def explain_rejection(model, original_instance, feature_names):
    # Define ranges for mutable features
    # For simplicity, assuming all features are mutable and within a reasonable range
    feature_range = {
        "credit_score": (300, 850),
        "debt_to_income_ratio": (0.1, 0.6),
        "income": (30000, 150000),
        "loan_amount": (5000, 100000),
        "employment_years": (0, 20),
    }

    # Create feature ranges for Alibi explainer
    ranges = [(feature_range[f][0], feature_range[f][1]) for f in feature_names]
    
    # Define the target class for counterfactual: 1 (approved)
    target_label = 1
    
    explainer = Counterfactual(predictor=model.predict_proba, 
                               shape=(1, len(feature_names)), 
                               feature_range=ranges,
                               target_proba=0.75, # Aim for at least 75% probability of approval
                               c_steps=50, max_iter=1000)
    
    explanation = explainer.explain(original_instance.reshape(1, -1), 
                                    target_label=target_label)
                                    
    return explanation

if __name__ == "__main__":
    # 1. Data Generation and Model Training
    print("Generating synthetic loan data...")
    loan_data = generate_synthetic_loan_data(num_samples=2000)
    
    print("Training black-box loan prediction model...")
    model, feature_names = train_black_box_model(loan_data)
    
    # 2. Simulate a rejected loan application
    # Find a rejected instance for explanation
    rejected_applications = loan_data[loan_data["approved"] == 0].drop("approved", axis=1)
    if not rejected_applications.empty:
        original_rejected_instance_df = rejected_applications.sample(1, random_state=1)
        original_rejected_instance = original_rejected_instance_df.values[0]
        
        print("\n--- Original Loan Application --- ")
        print(original_rejected_instance_df.to_string(index=False))
        initial_prediction = predict_loan_decision(model, original_rejected_instance)
        print(f"Initial Loan Decision: {'Approved' if initial_prediction == 1 else 'Rejected'}")
        
        if initial_prediction == 0:
            print("\n--- Generating Counterfactual Explanation --- ")
            explanation = explain_rejection(model, original_rejected_instance, feature_names)
            
            if explanation.cf is not None:
                cf_instance = explanation.cf["X"][0]
                cf_prediction = predict_loan_decision(model, cf_instance)
                
                print("\n--- Counterfactual Explanation Found --- ")
                print("Minimal changes for loan approval:")
                cf_df = pd.DataFrame([cf_instance], columns=feature_names)
                
                changes = {}
                for i, col in enumerate(feature_names):
                    original_val = original_rejected_instance[i]
                    cf_val = cf_instance[i]
                    if not np.isclose(original_val, cf_val, atol=1e-2): # Use isclose for float comparison
                        changes[col] = f"{original_val:.2f} -> {cf_val:.2f}"
                
                if changes:
                    for k, v in changes.items():
                        print(f"  - {k}: {v}")
                else:
                    print("No significant changes found (this is unexpected if a counterfactual was found).")

                print("\n--- Counterfactual Loan Application --- ")
                print(cf_df.to_string(index=False))
                print(f"Counterfactual Loan Decision: {'Approved' if cf_prediction == 1 else 'Rejected'}")
                
            else:
                print("Could not find a counterfactual explanation within the given parameters.")
        else:
            print("The selected instance was not rejected. Please ensure the instance is rejected to get an explanation.")
    else:
        print("No rejected applications found in the synthetic data to explain.")