class LoanApprovalModel:
    def __init__(self):
        self.weights = {
            "income": 0.05,
            "credit_score": 0.1,
            "debt_to_income_ratio": -20.0,
            "employment_years": 0.5
        }
        self.bias = -25

    def predict_score(self, application):
        score = 0
        for feature, weight in self.weights.items():
            if feature in application:
                score += application[feature] * weight
        score += self.bias
        return score

    def predict(self, application):
        score = self.predict_score(application)
        return "approved" if score > 0 else "denied"


class CounterfactualExplainer:
    def __init__(self, model):
        self.model = model
        self.feature_ranges = {
            "income": (20000, 200000),
            "credit_score": (300, 850),
            "debt_to_income_ratio": (0.01, 0.6),
            "employment_years": (0, 30)
        }
        self.step_sizes = {
            "income": 1000,
            "credit_score": 10,
            "debt_to_income_ratio": 0.005,
            "employment_years": 1
        }
        self.positive_impact_features = ["income", "credit_score", "employment_years"]
        self.negative_impact_features = ["debt_to_income_ratio"]

    def explain_denial(self, denied_application, target_prediction="approved", max_iterations=500):
        if self.model.predict(denied_application) == target_prediction:
            return None, "Application is already in the target state (approved)."

        candidate_application = denied_application.copy()
        minimal_changes = {}

        for _ in range(max_iterations):
            current_prediction = self.model.predict(candidate_application)
            if current_prediction == target_prediction:
                return candidate_application, minimal_changes

            found_beneficial_change_in_iteration = False

            features_to_try = []
            for feature in self.positive_impact_features:
                features_to_try.append((feature, 1)) # 1 for increasing
            for feature in self.negative_impact_features:
                features_to_try.append((feature, -1)) # -1 for decreasing
            
            # Sort features to prioritize those with higher impact or more common changes
            # For this simple model, we can just iterate, but in a real scenario, you might sort by weight magnitude

            for feature, direction in features_to_try:
                original_value = denied_application.get(feature, 0)
                current_value = candidate_application.get(feature, original_value)
                step = self.step_sizes.get(feature, 1 if direction == 1 else 0.01)
                feature_min, feature_max = self.feature_ranges.get(feature, (float("-inf"), float("inf")))

                if direction == 1: # Try increasing
                    if current_value + step <= feature_max:
                        temp_app = candidate_application.copy()
                        temp_app[feature] = current_value + step
                        if self.model.predict(temp_app) == target_prediction:
                            candidate_application[feature] = current_value + step
                            minimal_changes[feature] = candidate_application[feature] - original_value
                            return candidate_application, minimal_changes
                        elif self.model.predict_score(temp_app) > self.model.predict_score(candidate_application):
                            candidate_application[feature] = current_value + step
                            minimal_changes[feature] = candidate_application[feature] - original_value
                            found_beneficial_change_in_iteration = True
                            break
                else: # Try decreasing
                    if current_value - step >= feature_min:
                        temp_app = candidate_application.copy()
                        temp_app[feature] = current_value - step
                        if self.model.predict(temp_app) == target_prediction:
                            candidate_application[feature] = current_value - step
                            minimal_changes[feature] = candidate_application[feature] - original_value
                            return candidate_application, minimal_changes
                        elif self.model.predict_score(temp_app) > self.model.predict_score(candidate_application):
                            candidate_application[feature] = current_value - step
                            minimal_changes[feature] = candidate_application[feature] - original_value
                            found_beneficial_change_in_iteration = True
                            break
            
            if not found_beneficial_change_in_iteration:
                break

        return None, "Could not find a counterfactual explanation within the given constraints (max iterations or feature limits)."


def run_loan_explainer_demo():
    print("--- Counterfactual Loan Explainer Demo ---")

    loan_model = LoanApprovalModel()
    print("Loan Approval Model initialized.")

    explainer = CounterfactualExplainer(loan_model)
    print("Counterfactual Explainer initialized.")

    denied_application = {
        "income": 35000,
        "credit_score": 580,
        "debt_to_income_ratio": 0.55,
        "employment_years": 2
    }

    print("\n--- Original Application Details ---")
    print(f"Application: {denied_application}")

    original_prediction = loan_model.predict(denied_application)
    print(f"Original Model Prediction: {original_prediction}")

    if original_prediction == "denied":
        print("\n--- Generating Counterfactual Explanation ---")
        counterfactual_app, changes = explainer.explain_denial(denied_application)

        if counterfactual_app:
            print("\n--- Counterfactual Explanation Found! ---")
            print("To change the prediction from \'denied\' to \'approved\', you would need the following minimal changes:")
            for feature, change_amount in changes.items():
                if change_amount > 0:
                    print(f" - Increase {feature} by {change_amount:,.2f}")
                else:
                    print(f" - Decrease {feature} by {abs(change_amount):,.2f}")

            print("\n--- Resulting Counterfactual Application ---")
            print(f"Application: {counterfactual_app}")
            print(f"Counterfactual Model Prediction: {loan_model.predict(counterfactual_app)}")
            print("This shows the specific, actionable advice to get an approval.")
        else:
            print("\n--- Could not find a counterfactual explanation ---")
            print(f"Reason: {changes}")
    else:
        print("\n--- Application was already approved. No denial explanation needed. ---")

if __name__ == "__main__":
    run_loan_explainer_demo()
