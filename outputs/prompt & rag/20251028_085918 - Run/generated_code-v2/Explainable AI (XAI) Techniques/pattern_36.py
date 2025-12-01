import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "credit_score": np.random.randint(300, 850, num_samples),
        "debt_to_income_ratio": np.random.uniform(0.1, 0.6, num_samples),
        "income": np.random.randint(30000, 150000, num_samples),
        "loan_amount": np.random.randint(5000, 100000, num_samples),
        "employment_years": np.random.randint(0, 20, num_samples),
    }
    df = pd.DataFrame(data)

    # Simple logic to determine loan approval (can be more complex)
    # Higher credit score, lower DTI, higher income, more employment years lead to approval
    df["approved"] = ((df["credit_score"] > 650) & 
                     (df["debt_to_income_ratio"] < 0.4) & 
                     (df["income"] > 50000) & 
                     (df["employment_years"] > 2)).astype(int)

    return df

class CounterfactualExplainer:
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.target_prediction = 1  # Assuming 1 means approved

    def explain(self, instance, max_iterations=50, step_size_multiplier=0.05):
        original_prediction = self.model.predict(self.scaler.transform(instance.to_frame().T))[0]

        if original_prediction == self.target_prediction:
            return {
                "status": "approved",
                "original_instance": instance,
                "explanation": "Loan was already approved. No counterfactual needed."
            }

        best_counterfactual = None
        min_changes_magnitude = float('inf')
        best_changes_description = ""

        # Iterate through each feature to find a minimal change
        for i, feature in enumerate(self.feature_names):
            temp_instance = instance.copy()
            original_value = instance[feature]

            # Define a dynamic step size based on the feature's range or standard deviation
            feature_std = np.std(self.scaler.inverse_transform(self.scaler.transform(pd.DataFrame([instance.values], columns=self.feature_names)))[:, i])
            step = feature_std * step_size_multiplier if feature_std > 0 else 1.0 # Fallback for zero std

            # Try increasing the feature
            for iter_count in range(max_iterations):
                temp_instance[feature] = original_value + (iter_count + 1) * step
                if temp_instance[feature] < 0: # Prevent negative values for typically non-negative features
                    temp_instance[feature] = 0
                
                scaled_temp_instance = self.scaler.transform(temp_instance.to_frame().T)
                if self.model.predict(scaled_temp_instance)[0] == self.target_prediction:
                    change_magnitude = abs(temp_instance[feature] - original_value)
                    if change_magnitude < min_changes_magnitude:
                        min_changes_magnitude = change_magnitude
                        best_counterfactual = temp_instance.copy()
                        best_changes_description = f