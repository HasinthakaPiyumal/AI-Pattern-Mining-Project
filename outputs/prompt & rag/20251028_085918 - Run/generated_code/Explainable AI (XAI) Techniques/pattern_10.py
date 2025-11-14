def black_box_diagnosis_model(features):
    """
    Simulates a black-box medical diagnosis model.
    features: [Age, Symptom_A_Severity, Blood_Test_Y_Result]
    Returns 1 for positive diagnosis, 0 for negative.
    """
    age, symptom_a, blood_test_y = features
    if age > 50 and symptom_a > 0.7 and blood_test_y > 150:
        return 1 # High risk
    elif age < 30 and symptom_a < 0.3 and blood_test_y < 100:
        return 0 # Low risk
    elif symptom_a > 0.9 or blood_test_y > 200:
        return 1 # Very high risk
    return 0

def simulate_dataset():
    """Generates a small synthetic dataset."""
    data = [
        [60, 0.8, 160], # Patient 1: Positive
        [25, 0.2, 90],  # Patient 2: Negative
        [70, 0.95, 210],# Patient 3: Positive
        [45, 0.6, 130], # Patient 4: Negative
        [55, 0.75, 155],# Patient 5: Positive
        [30, 0.4, 110], # Patient 6: Negative
        [65, 0.9, 180], # Patient 7: Positive
    ]
    feature_names = ["Age", "Symptom_A_Severity", "Blood_Test_Y_Result"]
    return data, feature_names

def get_model_prediction(model, instance):
    return model(instance)

def local_interpretability_explanation(model, instance, feature_names):
    """
    Simplified LIME-like explanation for a single instance.
    Perturbs one feature at a time and observes prediction change.
    """
    original_prediction = get_model_prediction(model, instance)
    explanations = {}

    print(f"--- Local Explanation for Instance: {instance} (Predicted: {'Positive' if original_prediction == 1 else 'Negative'}) ---")

    for i, feature_name in enumerate(feature_names):
        perturbed_predictions = []
        original_feature_value = instance[i]
        
        # Simple perturbations: slightly increase and decrease
        perturb_values = []
        if isinstance(original_feature_value, (int, float)):
            # Add a range of values around the original
            perturb_values.append(original_feature_value * 0.8)
            perturb_values.append(original_feature_value * 1.2)
            if feature_name == "Age":
                perturb_values.append(original_feature_value - 10)
                perturb_values.append(original_feature_value + 10)
            elif feature_name == "Symptom_A_Severity":
                perturb_values.append(max(0, original_feature_value - 0.2))
                perturb_values.append(min(1, original_feature_value + 0.2))
            elif feature_name == "Blood_Test_Y_Result":
                perturb_values.append(original_feature_value - 20)
                perturb_values.append(original_feature_value + 20)
        else:
            print(f"Skipping perturbation for non-numeric feature '{feature_name}'.")
            continue

        # Ensure unique and non-negative perturbation values
        perturb_values = sorted(list(set([p for p in perturb_values if p >= 0])))

        for val in perturb_values:
            perturbed_instance = list(instance)
            perturbed_instance[i] = val
            perturbed_predictions.append(get_model_prediction(model, perturbed_instance))
        
        # Analyze prediction changes
        changed_predictions_count = sum(1 for p in perturbed_predictions if p != original_prediction)
        if changed_predictions_count > 0:
            explanations[feature_name] = f"Changing '{feature_name}' to {', '.join(map(str, perturb_values))} led to {changed_predictions_count} out of {len(perturb_values)} prediction changes, suggesting influence."
        else:
            explanations[feature_name] = f"Changing '{feature_name}' had no impact on prediction for the tested values."
    
    for feature, explanation in explanations.items():
        print(f"- {feature}: {explanation}")

    print("\n")


def global_partial_dependence_plot_explanation(model, data, feature_names, feature_index, num_points=10):
    """
    Simplified Partial Dependence Plot (PDP) explanation for a specific feature.
    Shows how the average prediction changes as a single feature varies.
    """
    target_feature_name = feature_names[feature_index]
    print(f"--- Global Explanation (Partial Dependence) for Feature: '{target_feature_name}' ---")

    if not data or not isinstance(data[0][feature_index], (int, float)):
        print(f"Cannot generate PDP for non-numeric or empty data for feature '{target_feature_name}'.")
        return

    feature_values = [row[feature_index] for row in data]
    min_val = min(feature_values)
    max_val = max(feature_values)
    step = (max_val - min_val) / (num_points - 1) if num_points > 1 else 1

    pdp_results = []
    
    for i in range(num_points):
        current_feature_val = min_val + i * step
        if target_feature_name == "Age":
            current_feature_val = int(current_feature_val) # Age should be int for this model

        temp_predictions = []
        for instance in data:
            temp_instance = list(instance)
            temp_instance[feature_index] = current_feature_val
            temp_predictions.append(get_model_prediction(model, temp_instance))
        
        average_prediction = sum(temp_predictions) / len(temp_predictions) if temp_predictions else 0
        pdp_results.append((current_feature_val, average_prediction))
    
    print(f"How average model prediction changes as '{target_feature_name}' varies (0=Negative, 1=Positive):")
    for val, avg_pred in pdp_results:
        print(f"  '{target_feature_name}' = {val:.2f} -> Average Prediction: {avg_pred:.2f}")
    print("\n")

def global_permutation_feature_importance_explanation(model, data, feature_names):
    """
    Simplified Permutation Feature Importance (PFI) explanation.
    Measures the drop in model performance when a feature's values are randomly shuffled.
    'Performance' is simplified to average prediction consistency here.
    """
    print(f"--- Global Explanation (Permutation Feature Importance) ---")

    if not data:
        print("No data available for PFI calculation.")
        return

    original_predictions = [get_model_prediction(model, instance) for instance in data]
    # A simple way to measure 'consistency' is how many predictions are 'positive'
    original_positive_predictions_count = sum(original_predictions)

    importance_scores = {}

    for i, feature_name in enumerate(feature_names):
        shuffled_data = []
        feature_column_values = [row[i] for row in data]
        
        # Basic in-place shuffle (not truly random without 'random' module)
        shuffled_feature_column = list(feature_column_values)
        if len(shuffled_feature_column) > 1:
            for j in range(len(shuffled_feature_column) - 1):
                shuffled_feature_column[j], shuffled_feature_column[j+1] = shuffled_feature_column[j+1], shuffled_feature_column[j]

        # Reconstruct the dataset with the shuffled feature
        for k in range(len(data)):
            temp_instance = list(data[k])
            temp_instance[i] = shuffled_feature_column[k]
            shuffled_data.append(temp_instance)
        
        # Get predictions on shuffled data
        shuffled_predictions = [get_model_prediction(model, instance) for instance in shuffled_data]
        shuffled_positive_predictions_count = sum(shuffled_predictions)

        # Importance is the absolute change in positive prediction count
        importance_scores[feature_name] = abs(original_positive_predictions_count - shuffled_positive_predictions_count)
    
    sorted_importance = sorted(importance_scores.items(), key=lambda item: item[1], reverse=True)
    
    print("Features ranked by their estimated importance (higher value means more impact on predictions when shuffled):")
    for feature, score in sorted_importance:
        print(f"- {feature}: {score:.4f}")
    print("\n")


# Main execution block
if __name__ == "__main__":
    # 1. Simulate Black-Box Model and Data
    diagnosis_model = black_box_diagnosis_model
    dataset, feature_names = simulate_dataset()

    print("--- Medical Diagnosis AI Interpretability Platform Demonstration ---")
    print("Simulated Dataset (first 3 instances for context):")
    for i in range(min(3, len(dataset))):
        print(f"  Patient {i+1} - {feature_names}: {dataset[i]} -> Prediction: {'Positive' if diagnosis_model(dataset[i]) == 1 else 'Negative'}")
    print("\n")

    # 2. Local Interpretability Example
    # Let's explain Patient 1 (index 0) with a positive diagnosis
    patient_to_explain_index = 0
    patient_instance = dataset[patient_to_explain_index]
    local_interpretability_explanation(diagnosis_model, patient_instance, feature_names)

    # Let's explain Patient 2 (index 1) with a negative diagnosis
    patient_to_explain_index_neg = 1
    patient_instance_neg = dataset[patient_to_explain_index_neg]
    local_interpretability_explanation(diagnosis_model, patient_instance_neg, feature_names)

    # 3. Global Interpretability Examples
    # Partial Dependence for 'Age'
    age_feature_index = feature_names.index("Age")
    global_partial_dependence_plot_explanation(diagnosis_model, dataset, feature_names, age_feature_index)

    # Partial Dependence for 'Symptom_A_Severity'
    symptom_feature_index = feature_names.index("Symptom_A_Severity")
    global_partial_dependence_plot_explanation(diagnosis_model, dataset, feature_names, symptom_feature_index)

    # Permutation Feature Importance
    global_permutation_feature_importance_explanation(diagnosis_model, dataset, feature_names)

    print("--- End of Demonstration ---")