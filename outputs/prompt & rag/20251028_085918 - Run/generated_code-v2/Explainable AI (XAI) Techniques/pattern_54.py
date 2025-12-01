import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

class CounterfactualExplainer:
    def __init__(self, model, feature_names, feature_bounds):
        self.model = model
        self.feature_names = feature_names
        self.feature_bounds = feature_bounds

    def find_counterfactual(self, denied_instance, desired_outcome=1, max_iterations=100, step_factor=0.05):
        original_prediction = self.model.predict(denied_instance.reshape(1, -1))[0]
        if original_prediction == desired_outcome:
            return denied_instance, {}, "Original instance already matches desired outcome."

        counterfactual_instance = denied_instance.copy()
        explanation_changes = {}
        
        current_prediction = original_prediction

        for i in range(max_iterations):
            if current_prediction == desired_outcome:
                break

            best_feature_to_change_idx = -1
            min_magnitude_change = float('inf')
            best_new_value = None
            
            # Iterate through features to find the best single change
            for f_idx, feature_name in enumerate(self.feature_names):
                original_val = denied_instance[f_idx]
                current_val = counterfactual_instance[f_idx]
                min_bound, max_bound = self.feature_bounds[feature_name]
                
                # Determine step size based on feature range
                step = (max_bound - min_bound) * step_factor
                if step == 0: # Handle cases with zero range (e.g., categorical or fixed value) - though not in this example
                    continue

                # Try increasing the feature value
                temp_cf_inc = counterfactual_instance.copy()
                new_val_inc = min(max_bound, current_val + step)
                if new_val_inc > current_val:
                    temp_cf_inc[f_idx] = new_val_inc
                    if self.model.predict(temp_cf_inc.reshape(1, -1))[0] == desired_outcome:
                        magnitude = abs(new_val_inc - original_val) # Distance from original value
                        if magnitude < min_magnitude_change:
                            min_magnitude_change = magnitude
                            best_feature_to_change_idx = f_idx
                            best_new_value = new_val_inc
                
                # Try decreasing the feature value (e.g., for debt-to-income ratio)
                temp_cf_dec = counterfactual_instance.copy()
                new_val_dec = max(min_bound, current_val - step)
                if new_val_dec < current_val:
                    temp_cf_dec[f_idx] = new_val_dec
                    if self.model.predict(temp_cf_dec.reshape(1, -1))[0] == desired_outcome:
                        magnitude = abs(new_val_dec - original_val)
                        if magnitude < min_magnitude_change:
                            min_magnitude_change = magnitude
                            best_feature_to_change_idx = f_idx
                            best_new_value = new_val_dec

            if best_feature_to_change_idx != -1:
                feature_name_to_change = self.feature_names[best_feature_to_change_idx]
                original_value_of_feature = counterfactual_instance[best_feature_to_change_idx]

                counterfactual_instance[best_feature_to_change_idx] = best_new_value
                
                if feature_name_to_change not in explanation_changes:
                    explanation_changes[feature_name_to_change] = {
                        "original": original_value_of_feature,
                        "new": best_new_value
                    }
                else:
                    explanation_changes[feature_name_to_change]["new"] = best_new_value

                current_prediction = self.model.predict(counterfactual_instance.reshape(1, -1))[0]
            else:
                # No single step change flipped the prediction in this iteration
                # This means we might be stuck or need larger steps or a different search strategy
                break 

        if current_prediction == desired_outcome:
            explanation_str = "To get approved, consider making the following minimal changes:\n"
            for feature, vals in explanation_changes.items():
                if vals["new"] != original_val: # Only show features that actually changed from the *initial* denied_instance
                    # Check if the feature was actually part of the original denied instance for comparison
                    original_denied_val = denied_instance[self.feature_names.index(feature)]
                    explanation_str += f"  - Change {feature} from {original_denied_val:.2f} to {vals['new']:.2f}\n"
            return counterfactual_instance, explanation_changes, explanation_str
        else:
            return denied_instance, {}, "Could not find a counterfactual explanation within the given iterations/steps."

if __name__ == "__main__":
    # 1. Generate Synthetic Data
    np.random.seed(42)
    num_samples = 1000

    # Feature distributions
    income = np.random.normal(loc=60000, scale=20000, size=num_samples)
    credit_score = np.random.normal(loc=680, scale=50, size=num_samples)
    debt_to_income_ratio = np.random.uniform(low=0.1, high=0.6, size=num_samples)
    employment_years = np.random.uniform(low=0.5, high=15, size=num_samples)

    # Define features and their bounds for the explainer
    feature_names = ['income', 'credit_score', 'debt_to_income_ratio', 'employment_years']
    feature_bounds = {
        'income': (20000, 150000),
        'credit_score': (300, 850),
        'debt_to_income_ratio': (0.05, 0.7),
        'employment_years': (0, 20)
    }

    # Create a simple rule for loan approval (simulating a black-box model logic)
    # Approved if: high income, good credit, low DTI, decent employment history
    loan_approved = (
        (income > 55000) &
        (credit_score > 660) &
        (debt_to_income_ratio < 0.4) &
        (employment_years > 1.5)
    ).astype(int)

    data = pd.DataFrame({
        'income': income,
        'credit_score': credit_score,
        'debt_to_income_ratio': debt_to_income_ratio,
        'employment_years': employment_years,
        'loan_approved': loan_approved
    })

    X = data[feature_names].values
    y = data['loan_approved'].values

    # 2. Train a Black-Box Loan Approval Model
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X, y)

    print(f"Model accuracy on synthetic data: {model.score(X, y):.2f}")

    # 3. Find a Denied Loan Application to Explain
    denied_applications = data[data['loan_approved'] == 0]
    if not denied_applications.empty:
        denied_instance_df = denied_applications.sample(1, random_state=1).iloc[0]
        denied_instance = denied_instance_df[feature_names].values

        print(f"\n--- Denied Loan Application ---\n")
        for feature, value in zip(feature_names, denied_instance):
            print(f"{feature}: {value:.2f}")
        print(f"Predicted Outcome: {'Denied' if model.predict(denied_instance.reshape(1, -1))[0] == 0 else 'Approved'}")

        # 4. Generate Counterfactual Explanation
        explainer = CounterfactualExplainer(model, feature_names, feature_bounds)
        counterfactual_instance, changes_dict, explanation_string = explainer.find_counterfactual(denied_instance)

        print(f"\n--- Counterfactual Explanation ---")
        print(explanation_string)

        if counterfactual_instance is not None and model.predict(counterfactual_instance.reshape(1, -1))[0] == 1:
            print(f"\nCounterfactual Instance (if approved):")
            for feature, value in zip(feature_names, counterfactual_instance):
                print(f"{feature}: {value:.2f}")
            print(f"Predicted Outcome for Counterfactual: {'Approved' if model.predict(counterfactual_instance.reshape(1, -1))[0] == 1 else 'Denied'}")

    else:
        print("No denied applications found in the synthetic dataset. Adjust data generation rules.")