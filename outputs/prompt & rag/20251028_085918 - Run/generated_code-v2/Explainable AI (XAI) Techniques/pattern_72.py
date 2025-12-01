import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import dice_ml
from dice_ml.utils import helpers

MODEL_FILENAME = "credit_score_model.joblib"
FEATURE_NAMES_FILENAME = "credit_score_features.joblib"

def generate_and_train_model():
    np.random.seed(42)
    num_samples = 1000

    credit_utilization = np.random.rand(num_samples) * 0.9 # 0 to 0.9
    payment_history = np.random.randint(0, 10, num_samples) # Number of on-time payments
    income = np.random.randint(30000, 150000, num_samples)
    debt_to_income_ratio = np.random.rand(num_samples) * 0.7 # 0 to 0.7
    num_credit_accounts = np.random.randint(1, 15, num_samples)
    loan_amount_requested = np.random.randint(5000, 50000, num_samples)

    data = pd.DataFrame({
        'credit_utilization': credit_utilization,
        'payment_history': payment_history,
        'income': income,
        'debt_to_income_ratio': debt_to_income_ratio,
        'num_credit_accounts': num_credit_accounts,
        'loan_amount_requested': loan_amount_requested
    })

    # Simple logic for credit approval: higher income, better payment history, lower utilization/debt-to-income leads to approval
    credit_approved = ((data['income'] > 60000) & 
                       (data['payment_history'] >= 5) & 
                       (data['credit_utilization'] < 0.5) & 
                       (data['debt_to_income_ratio'] < 0.4)).astype(int)

    data['credit_approved'] = credit_approved

    X = data.drop('credit_approved', axis=1)
    y = data['credit_approved']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(X.columns.tolist(), FEATURE_NAMES_FILENAME)

    # Save a small sample of original data for Dice Data interface
    data.to_csv("credit_data_sample.csv", index=False)

def explain_credit_decision(denied_instance_df, desired_outcome=1):
    model = joblib.load(MODEL_FILENAME)
    feature_names = joblib.load(FEATURE_NAMES_FILENAME)

    # Load the full data to correctly define feature ranges for Dice
    full_data = pd.read_csv("credit_data_sample.csv")
    d = dice_ml.Data(dataframe=full_data, 
                     continuous_features=feature_names, 
                     outcome_name='credit_approved')
    
    m = dice_ml.Model(model=model, backend='sklearn', model_type='classifier')

    exp = dice_ml.Dice(d, m)

    query_instance = denied_instance_df

    # Ensure query instance columns match training data order
    query_instance = query_instance[feature_names]

    desired_class = [desired_outcome]

    cf_explanations = exp.generate_counterfactuals(
        query_instance, 
        total_CFs=3, 
        desired_class=desired_class,
        proximity_weight=0.75, # Emphasize proximity to original instance
        sparsity_weight=0.25, # Emphasize minimal changes
        verbose=False
    )

    explanations = []
    for i, cf in enumerate(cf_explanations.cf_examples_list[0].final_cfs_df.iterrows()):
        changes = []
        original_values = denied_instance_df.iloc[0]
        counterfactual_values = cf[1]

        for feature in feature_names:
            if original_values[feature] != counterfactual_values[feature]:
                if feature == 'credit_utilization':
                    change_percent = (original_values[feature] - counterfactual_values[feature]) / original_values[feature] * 100
                    changes.append(f"Reduce your credit utilization by {change_percent:.1f}% (from {original_values[feature]:.2f} to {counterfactual_values[feature]:.2f})")
                elif feature == 'payment_history':
                    changes.append(f"Increase your payment history to {int(counterfactual_values[feature])} on-time payments (from {int(original_values[feature])})")
                elif feature == 'income':
                    changes.append(f"Increase your annual income to ${counterfactual_values[feature]:,.0f} (from ${original_values[feature]:,.0f})")
                elif feature == 'debt_to_income_ratio':
                    change_percent = (original_values[feature] - counterfactual_values[feature]) / original_values[feature] * 100 if original_values[feature] != 0 else 0
                    changes.append(f"Reduce your debt-to-income ratio by {change_percent:.1f}% (from {original_values[feature]:.2f} to {counterfactual_values[feature]:.2f})")
                elif feature == 'num_credit_accounts':
                    changes.append(f"Adjust number of credit accounts to {int(counterfactual_values[feature])} (from {int(original_values[feature])})")
                elif feature == 'loan_amount_requested':
                    changes.append(f"Consider requesting a loan amount of ${counterfactual_values[feature]:,.0f} (instead of ${original_values[feature]:,.0f})")

        if changes:
            explanations.append(f"Option {i+1}: To get approved, you could: " + "; ".join(changes) + ".")
        else:
            explanations.append(f"Option {i+1}: No significant changes were found to reach the desired outcome with current parameters.")

    return explanations

if __name__ == "__main__":
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(FEATURE_NAMES_FILENAME) or not os.path.exists("credit_data_sample.csv"):
        print("Model, feature names, or data sample not found. Generating data and training model...")
        generate_and_train_model()
        print("Model training complete.")

    model = joblib.load(MODEL_FILENAME)
    feature_names = joblib.load(FEATURE_NAMES_FILENAME)

    # Create a sample denied instance (ensure it matches features and has values that would lead to denial)
    # Example: high credit_utilization, low payment_history, moderate income, high debt_to_income
    sample_denied_instance = pd.DataFrame([[0.85, 2, 50000, 0.55, 4, 25000]],
                                         columns=feature_names)
    print("\nSample Denied Instance:")
    print(sample_denied_instance)

    original_prediction = model.predict(sample_denied_instance)[0]
    print(f"\nOriginal Credit Decision: {'Approved' if original_prediction == 1 else 'Denied'}")

    if original_prediction == 0: 
        print("\nGenerating counterfactual explanations...")
        explanations = explain_credit_decision(sample_denied_instance, desired_outcome=1)
        print("\nActionable Advice to get Credit Approved:")
        if explanations:
            for exp in explanations:
                print(exp)
        else:
            print("Could not find clear actionable advice to achieve approval with the given parameters.")
    else:
        print("The sample instance is already approved. No counterfactuals needed for approval.")

