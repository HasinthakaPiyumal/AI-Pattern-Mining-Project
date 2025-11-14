import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance, partial_dependence

# 1. Data Simulation
def simulate_patient_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 80, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples, p=[0.5, 0.5]),
        'ethnicity': np.random.choice(['Caucasian', 'African American', 'Asian', 'Other'], num_samples, p=[0.6, 0.2, 0.1, 0.1]),
        'symptom_A': np.random.randint(0, 2, num_samples), # Binary symptom
        'symptom_B': np.random.rand(num_samples),          # Continuous symptom
        'blood_test_X': np.random.normal(100, 15, num_samples),
        'diagnosis': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]) # 0: No Disease, 1: Disease
    }
    df = pd.DataFrame(data)

    # Introduce some correlation for diagnosis
    df['diagnosis'] = (df['symptom_A'] * 0.3 + df['symptom_B'] * 0.4 + (df['age'] > 60) * 0.2 + np.random.rand(num_samples) * 0.5 > 0.6).astype(int)

    return df

# Preprocessing function
def preprocess_data(df):
    df_encoded = pd.get_dummies(df, columns=['gender', 'ethnicity'], drop_first=True)
    X = df_encoded.drop('diagnosis', axis=1)
    y = df_encoded['diagnosis']
    return X, y, df_encoded.columns # Return columns for feature names

# 2. Model Training (Black-box model)
def train_diagnosis_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 3. Local Interpretability Functions (Conceptual implementations)

def generate_LACE_explanation(model, instance, feature_names):
    """
    Conceptual function for Local Interpretable Model-agnostic Explanations (LACE).
    In a real scenario, this would involve a more complex algorithm to find a local
    surrogate model (e.g., a decision tree) that approximates the black-box model's
    behavior around the specific instance. For this example, we'll return a placeholder.
    """
    print(f"\n--- LACE Explanation for instance ---")
    print(f"Instance values: {instance.values}")
    print("Conceptual LACE output: A local decision tree or rule set that explains this specific prediction.")
    print("Example: 'If symptom_A is 1 AND symptom_B > 0.5, then Diagnosis=1' with 90% fidelity locally.")
    return "Conceptual LACE explanation generated."

def generate_ICE_plot_data(model, X, instance_index, feature_name):
    """
    Conceptual function to generate data for Individual Conditional Expectation (ICE) plot.
    ICE plots show how the prediction for a single instance changes as a feature varies,
    while other features are held constant. This function simulates the data points
    needed to draw such a plot.
    """
    print(f"\n--- ICE Plot Data for instance {instance_index} and feature '{feature_name}' ---")
    original_instance = X.iloc[[instance_index]]
    feature_values = np.linspace(X[feature_name].min(), X[feature_name].max(), 50)
    predictions = []

    for val in feature_values:
        temp_instance = original_instance.copy()
        temp_instance[feature_name] = val
        pred_proba = model.predict_proba(temp_instance)[0, 1] # Probability of diagnosis=1
        predictions.append(pred_proba)
    
    print(f"Feature '{feature_name}' values: {feature_values[:5].round(2)}...{feature_values[-5:].round(2)}")
    print(f"Corresponding predicted probabilities (diagnosis=1): {np.array(predictions)[:5].round(2)}...{np.array(predictions)[-5:].round(2)}")
    print("Conceptual ICE plot data: A series of (feature_value, predicted_probability) pairs for a single instance.")
    return pd.DataFrame({'feature_value': feature_values, 'predicted_probability': predictions})

def generate_counterfactual_explanation(model, instance, feature_names, target_class=0):
    """
    Conceptual function for Counterfactual Explanations.
    A counterfactual explanation answers: 'What is the smallest change to the instance's
    features that would change its prediction to the target_class?'
    This is a complex optimization problem. Here, we'll provide a simplified example.
    """
    print(f"\n--- Counterfactual Explanation for instance ---")
    original_prediction = model.predict(instance)[0]
    if original_prediction == target_class:
        print(f"Instance already predicts {target_class}. No counterfactual needed.")
        return None

    print(f"Original instance predicts: {original_prediction}. Target: {target_class}.")
    print("Conceptual Counterfactual: Identify minimal feature changes to flip the prediction.")
    print("Example: If original prediction was 'Disease' (1), and target is 'No Disease' (0):\n")
    print("   'If symptom_A was 0 instead of 1, and blood_test_X was 90 instead of 110, then Diagnosis=0.'")
    return "Conceptual counterfactual explanation generated."

# 4. Global Interpretability Functions

def generate_partial_dependence_data(model, X, feature_names, feature_to_plot):
    """
    Generates data for Partial Dependence Plots (PDP).
    PDPs show the marginal effect of one or two features on the predicted outcome
    of a black-box model.
    """
    print(f"\n--- Partial Dependence Plot Data for feature '{feature_to_plot}' ---")
    # Ensure feature_to_plot is in a list for partial_dependence
    features_indices = [feature_names.get_loc(feature_to_plot)] if isinstance(feature_to_plot, str) else [feature_names.get_loc(f) for f in feature_to_plot]

    pdp_results = partial_dependence(model, X, features=features_indices, kind='average', grid_resolution=50)

    # Extract results for single feature
    if len(features_indices) == 1:
        feature_values = pdp_results.grid[0]
        average_predictions = pdp_results.average[0]
        pdp_df = pd.DataFrame({'feature_value': feature_values, 'average_prediction': average_predictions})
        print(f"Feature '{feature_to_plot}' values: {feature_values[:5].round(2)}...{feature_values[-5:].round(2)}")
        print(f"Corresponding average predicted probabilities (diagnosis=1): {average_predictions[:5].round(2)}...{average_predictions[-5:].round(2)}")
        print("Conceptual PDP data: Average predicted probability for diagnosis=1 as the feature varies.")
        return pdp_df
    else:
        # For two features, the output is a grid, which is harder to represent simply in text.
        # We'll just print a conceptual message.
        print(f"Conceptual PDP data for multiple features ({', '.join(feature_to_plot)}): A grid of average predicted probabilities.")
        return None # In a real visualization, this would be a 2D array or meshgrid

def calculate_permutation_feature_importance(model, X_test, y_test, feature_names):
    """
    Calculates Permutation Feature Importance (PFI).
    PFI measures the decrease in a model's score when a single feature is randomly
    permuted. This breaks the relationship between the feature and the target,
    thus the drop in the model score is indicative of how much the model depends on that feature.
    """
    print("\n--- Permutation Feature Importance ---")
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    
    importance_df = pd.DataFrame({
        'feature': feature_names[sorted_idx],
        'mean_importance': result.importances_mean[sorted_idx],
        'std_importance': result.importances_std[sorted_idx]
    })
    print(importance_df.to_string(index=False))
    return importance_df

# 5. Bias Detection Function

def detect_bias_in_subgroups(model, X_test, y_test, original_data_test, feature_names):
    """
    Detects potential biases by evaluating model performance across different demographic subgroups.
    """
    print("\n--- Bias Detection Across Subgroups ---")
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Map one-hot encoded features back to original for subgroup analysis
    original_test_with_predictions = original_data_test.copy()
    original_test_with_predictions['true_diagnosis'] = y_test.values
    original_test_with_predictions['predicted_diagnosis'] = predictions
    original_test_with_predictions['predicted_proba'] = probabilities

    subgroup_features = ['gender', 'ethnicity'] # Features to check for bias
    bias_reports = {}

    for feature in subgroup_features:
        print(f"\nAnalyzing bias for feature: '{feature}'")
        bias_reports[feature] = {}
        for subgroup in original_test_with_predictions[feature].unique():
            subgroup_df = original_test_with_predictions[original_test_with_predictions[feature] == subgroup]
            if len(subgroup_df) == 0: # Handle empty subgroups
                continue

            sub_y_true = subgroup_df['true_diagnosis']
            sub_y_pred = subgroup_df['predicted_diagnosis']

            accuracy = accuracy_score(sub_y_true, sub_y_pred)
            precision = precision_score(sub_y_true, sub_y_pred, zero_division=0)
            recall = recall_score(sub_y_true, sub_y_pred, zero_division=0)
            f1 = f1_score(sub_y_true, sub_y_pred, zero_division=0)

            bias_reports[feature][subgroup] = {
                'count': len(subgroup_df),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            print(f"  Subgroup '{subgroup}' (n={len(subgroup_df)}): ")
            print(f"    Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    print("\nInterpretation: Significant disparities in these metrics across subgroups may indicate bias.")
    print("For example, if 'Female' subgroup has much lower recall for 'Disease' compared to 'Male' subgroup, it indicates bias.")
    return bias_reports

# Main execution for demonstration
if __name__ == "__main__":
    print("--- Starting Medical Diagnosis Explainer System ---")

    # 1. Simulate and Preprocess Data
    print("\nSimulating patient data...")
    original_data = simulate_patient_data(num_samples=2000)
    X, y, all_feature_names = preprocess_data(original_data)

    # Split data while keeping original data for bias detection
    X_train, X_test, y_train, y_test, original_data_train, original_data_test = \
        train_test_split(X, y, original_data, test_size=0.2, random_state=42, stratify=y)
    
    feature_names = X_train.columns.tolist() # Features used by the model

    # 2. Train Model
    print("\nTraining black-box diagnosis model...")
    model = train_diagnosis_model(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model trained. Test Accuracy: {test_accuracy:.4f}")

    # Select an instance for local explanations
    instance_index_for_explanation = 0
    example_instance = X_test.iloc[[instance_index_for_explanation]]

    # 3. Local Interpretability
    generate_LACE_explanation(model, example_instance, feature_names)

    ice_df = generate_ICE_plot_data(model, X_test, instance_index_for_explanation, 'symptom_B')
    # In a real application, ice_df would be used to plot the ICE curve.

    generate_counterfactual_explanation(model, example_instance, feature_names, target_class=0) # Try to change to 'No Disease'

    # 4. Global Interpretability
    pdp_df = generate_partial_dependence_data(model, X_test, X.columns, 'age')
    # In a real application, pdp_df would be used to plot the PDP curve.

    pfi_df = calculate_permutation_feature_importance(model, X_test, y_test, np.array(feature_names))

    # 5. Bias Detection
    bias_reports = detect_bias_in_subgroups(model, X_test, y_test, original_data_test, feature_names)

    print("\n--- Medical Diagnosis Explainer System Finished ---")
