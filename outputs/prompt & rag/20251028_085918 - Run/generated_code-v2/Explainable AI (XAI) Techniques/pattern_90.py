import copy

class CreditModel:
    """
    A simplified, rule-based black-box model to predict credit risk category.
    """
    def predict(self, features):
        income = features.get("income", 0)
        credit_utilization = features.get("credit_utilization", 1.0) # as a ratio, e.g., 0.3 for 30%
        payment_history_score = features.get("payment_history_score", 0) # e.g., 0-100

        if income >= 60000 and credit_utilization <= 0.3 and payment_history_score >= 80:
            return "low_risk"
        elif income >= 30000 and credit_utilization <= 0.6 and payment_history_score >= 60:
            return "medium_risk"
        else:
            return "high_risk"

class CounterfactualGenerator:
    """
    Generates counterfactual explanations for the CreditModel.
    Identifies minimal changes to features to achieve a desired credit risk outcome.
    """
    def __init__(self, model):
        self.model = model
        self.feature_info = {
            "income": {"step": 5000, "min": 10000, "max": 100000, "direction": "increase"},
            "credit_utilization": {"step": 0.05, "min": 0.0, "max": 1.0, "direction": "decrease"},
            "payment_history_score": {"step": 5, "min": 0, "max": 100, "direction": "increase"}
        }

    def generate_counterfactual(self, original_instance, desired_outcome, max_iterations=200):
        current_instance = copy.deepcopy(original_instance)
        initial_prediction = self.model.predict(current_instance)

        if initial_prediction == desired_outcome:
            return original_instance, {}, f"Already {desired_outcome}. No changes needed."

        changes_made = {}
        explanation_steps = []

        for _ in range(max_iterations):
            made_change_in_iteration = False
            sorted_features = sorted(self.feature_info.keys())

            for feature in sorted_features:
                original_value = current_instance[feature]
                info = self.feature_info[feature]
                temp_instance = copy.deepcopy(current_instance)

                if info["direction"] == "increase":
                    new_value = original_value + info["step"]
                    if new_value <= info["max"]:
                        temp_instance[feature] = new_value
                    else:
                        continue
                elif info["direction"] == "decrease":
                    new_value = original_value - info["step"]
                    if new_value >= info["min"]:
                        temp_instance[feature] = new_value
                    else:
                        continue
                else:
                    continue # Should not happen with defined directions
                
                temp_prediction = self.model.predict(temp_instance)

                if temp_prediction == desired_outcome:
                    # Found a counterfactual, record the changes and return
                    if feature not in changes_made:
                        changes_made[feature] = new_value - original_instance[feature]
                    else:
                        changes_made[feature] += (new_value - current_instance[feature]) # Accumulate changes from original
                    
                    current_instance[feature] = new_value
                    
                    explanation_steps.append(f"Change {feature} from {original_value} to {new_value}")
                    final_explanation = "To achieve a " + desired_outcome + " credit risk:\n"
                    for f, change_val in changes_made.items():
                        if f == "credit_utilization":
                             final_explanation += f"  - Decrease {f} by {abs(change_val * 100):.0f} percentage points (from {original_instance[f]:.2f} to {current_instance[f]:.2f}).\n"
                        elif f == "income":
                            final_explanation += f"  - Increase {f} by ${change_val:.0f} (from ${original_instance[f]:.0f} to ${current_instance[f]:.0f}).\n"
                        elif f == "payment_history_score":
                            final_explanation += f"  - Increase {f} by {change_val:.0f} points (from {original_instance[f]:.0f} to {current_instance[f]:.0f}).\n"

                    return current_instance, changes_made, final_explanation
                elif temp_prediction != self.model.predict(current_instance): # If changing this feature improves the situation, keep it
                    current_instance[feature] = new_value
                    if feature not in changes_made:
                        changes_made[feature] = new_value - original_instance[feature]
                    else:
                        changes_made[feature] += (new_value - current_instance[feature]) # Accumulate changes from original
                    
                    made_change_in_iteration = True
                    break # Move to next iteration to try improving further
            
            if not made_change_in_iteration and self.model.predict(current_instance) != desired_outcome:
                # No individual feature change in this iteration led to the desired outcome
                # or an improvement. This simple greedy approach might get stuck.
                # For a more robust solution, one would explore combinations or use a more sophisticated search.
                pass # Continue if we can't make individual improvements, maybe another iteration will find a path

        return original_instance, {}, f"Could not find a counterfactual to reach {desired_outcome} within {max_iterations} iterations."

if __name__ == "__main__":
    credit_model = CreditModel()
    cf_generator = CounterfactualGenerator(credit_model)

    # Example 1: User with high risk wants to be low risk
    user_profile_1 = {
        "income": 25000,
        "credit_utilization": 0.8,
        "payment_history_score": 50
    }
    print(f"\nUser 1 Profile: {user_profile_1}")
    initial_risk_1 = credit_model.predict(user_profile_1)
    print(f"Initial Credit Risk: {initial_risk_1}")

    if initial_risk_1 != "low_risk":
        counterfactual_1, changes_1, explanation_1 = cf_generator.generate_counterfactual(
            user_profile_1, "low_risk"
        )
        print(f"Counterfactual Instance: {counterfactual_1}")
        print(f"Changes Needed: {changes_1}")
        print(explanation_1)
    else:
        print("Already low_risk, no counterfactual needed.")

    # Example 2: User with medium risk wants to be low risk
    user_profile_2 = {
        "income": 40000,
        "credit_utilization": 0.5,
        "payment_history_score": 70
    }
    print(f"\nUser 2 Profile: {user_profile_2}")
    initial_risk_2 = credit_model.predict(user_profile_2)
    print(f"Initial Credit Risk: {initial_risk_2}")

    if initial_risk_2 != "low_risk":
        counterfactual_2, changes_2, explanation_2 = cf_generator.generate_counterfactual(
            user_profile_2, "low_risk"
        )
        print(f"Counterfactual Instance: {counterfactual_2}")
        print(f"Changes Needed: {changes_2}")
        print(explanation_2)
    else:
        print("Already low_risk, no counterfactual needed.")

    # Example 3: User with high risk wants to be medium risk
    user_profile_3 = {
        "income": 20000,
        "credit_utilization": 0.9,
        "payment_history_score": 40
    }
    print(f"\nUser 3 Profile: {user_profile_3}")
    initial_risk_3 = credit_model.predict(user_profile_3)
    print(f"Initial Credit Risk: {initial_risk_3}")

    if initial_risk_3 != "medium_risk":
        counterfactual_3, changes_3, explanation_3 = cf_generator.generate_counterfactual(
            user_profile_3, "medium_risk"
        )
        print(f"Counterfactual Instance: {counterfactual_3}")
        print(f"Changes Needed: {changes_3}")
        print(explanation_3)
    else:
        print("Already medium_risk, no counterfactual needed.")

