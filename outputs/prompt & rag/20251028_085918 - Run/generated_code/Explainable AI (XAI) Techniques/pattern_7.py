
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

import lime
import lime.lime_tabular
import shap

# --- 1. Data Generation --- 
print("Generating synthetic medical data...")
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, n_samples),
        'BloodPressure': np.random.randint(90, 180, n_samples),
        'Cholesterol': np.random.randint(150, 300, n_samples),
        'BMI': np.random.uniform(18.0, 35.0, n_samples),
        'GeneticMarker_A': np.random.randint(0, 2, n_samples), # Binary
        'GeneticMarker_B': np.random.randint(0, 2, n_samples), # Binary
        'Smoking': np.random.randint(0, 2, n_samples), # Binary
        'ExerciseFrequency': np.random.randint(0, 7, n_samples), # Days per week
    }
    df = pd.DataFrame(data)
    
    # Simulate a 'Disease' outcome based on some features
    # More complex interaction for a 'black-box' feel
    df['Disease'] = ((df['Age'] > 50).astype(int) * 0.3 + 
                     (df['BloodPressure'] > 140).astype(int) * 0.25 + 
                     (df['Cholesterol'] > 220).astype(int) * 0.2 + 
                     (df['BMI'] > 28).astype(int) * 0.15 + 
                     df['GeneticMarker_A'] * 0.1 - 
                     df['ExerciseFrequency'] * 0.05 + 
                     np.random.rand(n_samples) * 0.2 > 0.6).astype(int)
    
    return df

df = generate_synthetic_data()
features = [col for col in df.columns if col != 'Disease']
X = df[features]
y = df['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset shape: {df.shape}")
print(f"Features: {features}")

# --- 2. Black-box Predictive Model --- 
print("\nTraining a RandomForestClassifier (black-box model)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- 3. Local Interpretability Module ---
print("\n--- Local Interpretability ---")

# Choose an instance to explain (e.g., the first test instance)
instance_idx = 0
sample_instance = X_test.iloc[[instance_idx]]
sample_label = y_test.iloc[instance_idx]
sample_prediction = model.predict(sample_instance)[0]
print(f"\nExplaining instance {instance_idx}: True Label={sample_label}, Predicted Label={sample_prediction}")

# LIME Explanation
print("\nGenerating LIME explanation...")
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=['No Disease', 'Disease'],
    mode='classification'
)
explanation_lime = explainer_lime.explain_instance(
    data_row=sample_instance.values[0],
    predict_fn=model.predict_proba,
    num_features=5
)
print("LIME explanation for instance:")
for feature, weight in explanation_lime.as_list():
    print(f"  {feature}: {weight:.4f}")
# explanation_lime.show_in_notebook(show_all=False) # Uncomment to view in Jupyter

# SHAP Explanation
print("\nGenerating SHAP explanation...")
explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(sample_instance)

# For binary classification, shap_values is a list of two arrays (for class 0 and class 1)
# We typically look at the positive class (class 1 for 'Disease')
shap_values_class_1 = shap_values[1][0]

print("SHAP values for instance (contributions to 'Disease' prediction):")
for feature, shap_val in zip(X_test.columns, shap_values_class_1):
    print(f"  {feature}: {shap_val:.4f}")

# shap.initjs() # Uncomment for JS visualizations in Jupyter
# shap.force_plot(explainer_shap.expected_value[1], shap_values_class_1, sample_instance)
# shap.waterfall_plot(shap.Explanation(values=shap_values_class_1, base_values=explainer_shap.expected_value[1], data=sample_instance.values[0], feature_names=X_test.columns))

# ICE Plots (for a single feature and instance)
print("\nGenerating ICE Plot for 'Age'...")
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(
    estimator=model,
    X=X_test,
    features=['Age'],
    kind='individual',
    feature_names=features,
    ax=ax,
    pd_kwargs={'feature_names': features} # Pass feature names for plotting
)
ax.set_title(f"ICE Plot for Age (Instance {instance_idx})")
plt.tight_layout()
# plt.show() # Uncomment to display plot
plt.close(fig) # Close plot to avoid showing many plots if run in a loop

# Counterfactual Explanations (Simplified/Conceptual)
print("\nGenerating simplified Counterfactual Explanation...")
def generate_counterfactual_explanation(model, instance, features, target_label=0, threshold=0.5, step_size=5):
    original_prediction_proba = model.predict_proba(instance)[:, 1][0]
    original_prediction_label = 1 if original_prediction_proba >= threshold else 0

    if original_prediction_label == target_label:
        return "Instance already predicts the target label. No counterfactual needed."

    counterfactual_changes = {}
    temp_instance = instance.copy()

    print(f"Original prediction for target label ({target_label}): {original_prediction_proba:.4f}")
    print(f"Trying to change prediction from {original_prediction_label} to {target_label}")

    # Iterate through features to find minimal changes
    for feature in features:
        original_value = temp_instance[feature].iloc[0]
        
        # Try decreasing value
        if pd.api.types.is_numeric_dtype(temp_instance[feature]):
            candidate_value = original_value
            while True:
                candidate_value -= step_size
                if feature in ['GeneticMarker_A', 'GeneticMarker_B', 'Smoking']:
                    candidate_value = max(0, int(candidate_value))
                temp_instance[feature] = candidate_value
                new_prediction_proba = model.predict_proba(temp_instance)[:, 1][0]
                new_prediction_label = 1 if new_prediction_proba >= threshold else 0
                
                if new_prediction_label == target_label:
                    counterfactual_changes[feature] = f"Decrease {feature} from {original_value:.2f} to {candidate_value:.2f}"
                    print(f"  Found counterfactual by changing {feature}. New prediction: {new_prediction_proba:.4f}")
                    return counterfactual_changes
                if (original_value - candidate_value) > 50: # Avoid infinite loop for continuous features
                    break
                
            temp_instance[feature] = original_value # Reset for next attempt
            
            # Try increasing value
            candidate_value = original_value
            while True:
                candidate_value += step_size
                if feature in ['GeneticMarker_A', 'GeneticMarker_B', 'Smoking']:
                    candidate_value = min(1, int(candidate_value))
                temp_instance[feature] = candidate_value
                new_prediction_proba = model.predict_proba(temp_instance)[:, 1][0]
                new_prediction_label = 1 if new_prediction_proba >= threshold else 0
                
                if new_prediction_label == target_label:
                    counterfactual_changes[feature] = f"Increase {feature} from {original_value:.2f} to {candidate_value:.2f}"
                    print(f"  Found counterfactual by changing {feature}. New prediction: {new_prediction_proba:.4f}")
                    return counterfactual_changes
                if (candidate_value - original_value) > 50: # Avoid infinite loop
                    break
            temp_instance[feature] = original_value # Reset for next attempt
        
    return "Could not find a simple counterfactual by changing one feature."

# Example usage of counterfactual explanation
counterfactual = generate_counterfactual_explanation(model, sample_instance, features, target_label=0) # Try to change to 'No Disease'
print("Counterfactual Explanation:")
print(counterfactual)


# --- 4. Global Interpretability Module ---
print("\n--- Global Interpretability ---")

# Partial Dependence Plots (PDPs)
print("\nGenerating Partial Dependence Plots...")
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(
    estimator=model,
    X=X_train,
    features=['Age', 'Cholesterol', ('Age', 'Cholesterol')], # Single and interaction plots
    feature_names=features,
    target=1, # Plot for the 'Disease' class
    grid_resolution=20,
    ax=ax
)
ax.set_title("Partial Dependence Plots")
plt.tight_layout()
# plt.show() # Uncomment to display plot
plt.close(fig) # Close plot

# Permutation Feature Importance
print("\nCalculating Permutation Feature Importance...")
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(result.importances[sorted_idx].T, vert=False,
           labels=X_test.columns[sorted_idx])
ax.set_title("Permutation Feature Importance (Test Set)")
plt.tight_layout()
# plt.show() # Uncomment to display plot
plt.close(fig) # Close plot


# --- 5. Interactive Debugging and Bias Detection (Conceptual) ---
print("\n--- Interactive Debugging and Bias Detection (Conceptual) ---")

# Compare model behavior across different patient subgroups
print("\nAnalyzing model predictions across Age subgroups...")
def analyze_subgroup_performance(model, X, y, subgroup_feature, bins, labels):
    data = X.copy()
    data['TrueLabel'] = y
    data['Prediction'] = model.predict(X)
    data['PredictionProba'] = model.predict_proba(X)[:, 1]

    data['Subgroup'] = pd.cut(data[subgroup_feature], bins=bins, labels=labels, right=False)

    print(f"Performance by {subgroup_feature} Subgroup:")
    for subgroup in data['Subgroup'].unique():
        if pd.isna(subgroup): # Handle NaN if any
            continue
        subgroup_df = data[data['Subgroup'] == subgroup]
        if not subgroup_df.empty:
            accuracy = accuracy_score(subgroup_df['TrueLabel'], subgroup_df['Prediction'])
            positive_pred_rate = (subgroup_df['Prediction'] == 1).mean()
            true_positive_rate = subgroup_df[subgroup_df['TrueLabel'] == 1]['Prediction'].mean()
            false_positive_rate = subgroup_df[subgroup_df['TrueLabel'] == 0]['Prediction'].mean()

            print(f"  Subgroup '{subgroup}':")
            print(f"    N samples: {len(subgroup_df)}")
            print(f"    Accuracy: {accuracy:.4f}")
            print(f"    Predicted Disease Rate: {positive_pred_rate:.4f}")
            print(f"    True Positive Rate (Sensitivity): {true_positive_rate:.4f}")
            print(f"    False Positive Rate: {false_positive_rate:.4f}")

# Example subgroup analysis for 'Age'
age_bins = [20, 40, 60, 80]
age_labels = ['20-39', '40-59', '60-79']
analyze_subgroup_performance(model, X_test, y_test, 'Age', age_bins, age_labels)

# Identify divergent model behaviors (e.g., low confidence or disagreement)
print("\nIdentifying instances with low prediction confidence or high disagreement...")
def identify_divergent_behavior(model, X, y, confidence_threshold=0.6, top_n=5):
    predictions_proba = model.predict_proba(X)
    max_proba = np.max(predictions_proba, axis=1)
    predicted_labels = model.predict(X)

    # Cases with low confidence
    low_confidence_indices = np.where(max_proba < confidence_threshold)[0]
    print(f"\nTop {top_n} instances with low prediction confidence (< {confidence_threshold:.2f}):")
    if len(low_confidence_indices) > 0:
        for idx in low_confidence_indices[:top_n]:
            print(f"  Instance {idx}: Predicted {predicted_labels[idx]}, Confidence {max_proba[idx]:.4f}, True Label {y.iloc[idx]}")
            print(f"    Features: {X.iloc[idx].to_dict()}")
    else:
        print("  No instances found with confidence below threshold.")

    # Cases where prediction is different from a hypothetical 'simpler' model or expected outcome
    # For this example, we'll just show cases where prediction is wrong and confidence is high.
    wrong_predictions_high_confidence_indices = np.where((predicted_labels != y) & (max_proba >= confidence_threshold))[0]
    print(f"\nTop {top_n} instances with wrong prediction but high confidence:")
    if len(wrong_predictions_high_confidence_indices) > 0:
        for idx in wrong_predictions_high_confidence_indices[:top_n]:
            print(f"  Instance {idx}: Predicted {predicted_labels[idx]}, Confidence {max_proba[idx]:.4f}, True Label {y.iloc[idx]}")
            print(f"    Features: {X.iloc[idx].to_dict()}")
    else:
        print("  No instances found with wrong prediction and high confidence.")

identify_divergent_behavior(model, X_test, y_test, confidence_threshold=0.7, top_n=3)

print("\nInterpretable AI Framework demonstration complete.")
