import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def create_synthetic_medical_data(num_samples=1000):
    """Generates synthetic medical patient data for a binary diagnosis."""
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 80, num_samples),
        'blood_pressure': np.random.randint(90, 180, num_samples),
        'cholesterol': np.random.randint(120, 300, num_samples),
        'symptom_A': np.random.randint(0, 2, num_samples), # binary symptom
        'symptom_B': np.random.randint(0, 2, num_samples), # binary symptom
        'gender': np.random.choice(['Male', 'Female'], num_samples),
    }
    df = pd.DataFrame(data)

    # Create a synthetic 'diagnosis' based on features (simplified logic)
    df['diagnosis'] = ((df['age'] > 50).astype(int) +
                       (df['blood_pressure'] > 140).astype(int) +
                       (df['cholesterol'] > 200).astype(int) +
                       df['symptom_A'] +
                       (df['symptom_B'] * 2) + # symptom B has higher weight
                       (df['gender'].apply(lambda x: 1 if x == 'Female' else 0) * 0.5) # slight gender bias
                      ).apply(lambda x: 1 if x >= 3 else 0) # Diagnosis if score >= 3

    # Add some noise to make it less perfectly linear
    noise = np.random.choice([0, 1], num_samples, p=[0.8, 0.2]) # 20% random flip
    df['diagnosis'] = np.abs(df['diagnosis'] - noise)

    return df

def train_black_box_model(X_train, y_train):
    """Trains a RandomForestClassifier as the black-box model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def get_local_explanation(model, instance, feature_names):
    """
    Simulates a local explanation for a single instance.
    In a real system, this would use libraries like SHAP or LIME.
    Here, we'll just show the instance's features and a conceptual 'feature contribution'.
    """
    prediction = model.predict(instance.reshape(1, -1))[0]
    prediction_proba = model.predict_proba(instance.reshape(1, -1))[0]

    # Conceptual feature importance for this instance (simplified, not actual SHAP values)
    feature_contributions = {}
    for i, feature_name in enumerate(feature_names):
        # Assign a random "contribution" for demonstration purposes, scaled by feature value
        base_contribution = np.random.uniform(-0.5, 0.5)
        feature_contributions[feature_name] = base_contribution * instance[i] if instance[i] != 0 else base_contribution * 0.1

    explanation_text = f"Diagnosis Prediction: {'Positive' if prediction == 1 else 'Negative'}\n"
    explanation_text += f"Confidence (Negative/Positive): {prediction_proba[0]:.2f}/{prediction_proba[1]:.2f}\n"
    explanation_text += "\nConceptual Feature Contributions (Higher absolute value means more influence):\n"
    for feature, contribution in feature_contributions.items():
        explanation_text += f"- {feature}: {contribution:.2f}\n"

    return explanation_text, prediction

def get_global_feature_importance(model, feature_names):
    """
    Calculates and returns global feature importance using the model's built-in attribute.
    For Random Forests, this is Gini importance, indicating average feature contribution.
    """
    importances = model.feature_importances_
    global_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    return global_importance.to_string()

def analyze_subgroup_bias(model, X_test_processed, y_test, raw_test_df_original_features, categorical_features_for_bias, all_feature_names_after_ohe):
    """
    Analyzes model performance across different subgroups defined by categorical features.
    `raw_test_df_original_features` is the test set *before* one-hot encoding, with indices matching `X_test_processed` and `y_test`.
    """
    bias_report = "--- Subgroup Bias Analysis ---\n"
    
    # Ensure X_test_processed is a DataFrame for easier slicing with loc/iloc
    if not isinstance(X_test_processed, pd.DataFrame):
        X_test_processed_df = pd.DataFrame(X_test_processed, columns=all_feature_names_after_ohe, index=raw_test_df_original_features.index)
    else:
        X_test_processed_df = X_test_processed

    # Ensure y_test is a Series for easier slicing
    if not isinstance(y_test, pd.Series):
        y_test_series = pd.Series(y_test, index=raw_test_df_original_features.index)
    else:
        y_test_series = y_test

    for col in categorical_features_for_bias:
        if col in raw_test_df_original_features.columns:
            unique_subgroups = raw_test_df_original_features[col].unique()
            bias_report += f"\nAnalysis by {col}:\n"
            for subgroup_value in unique_subgroups:
                # Get indices for the current subgroup from the original test data
                subgroup_indices = raw_test_df_original_features[raw_test_df_original_features[col] == subgroup_value].index
                
                # Filter the processed X_test and y_test using these indices
                subgroup_X_test = X_test_processed_df.loc[subgroup_indices]
                subgroup_y_test = y_test_series.loc[subgroup_indices]

                if len(subgroup_X_test) > 0:
                    subgroup_preds = model.predict(subgroup_X_test)
                    subgroup_accuracy = accuracy_score(subgroup_y_test, subgroup_preds)
                    bias_report += f"  - Subgroup '{subgroup_value}' (n={len(subgroup_y_test)}):\n"
                    bias_report += f"    Accuracy: {subgroup_accuracy:.2f}\n"
                    try:
                        report = classification_report(subgroup_y_test, subgroup_preds, output_dict=False, zero_division=0)
                        bias_report += f"    Classification Report:\n{report}\n"
                    except ValueError as e:
                        bias_report += f"    Could not generate full classification report: {e}\n"
                else:
                    bias_report += f"  - Subgroup '{subgroup_value}': Not enough data in test set for this subgroup.\n"
        else:
            bias_report += f"  - Categorical feature '{col}' not found in the raw test data for bias analysis.\n"
    return bias_report


def run_explainer_system():
    """Main function to orchestrate the medical diagnosis explanation system."""
    print("Initializing Medical Diagnosis Explanation System...")

    # 1. Create Synthetic Data
    df = create_synthetic_medical_data()
    print(f"Generated {len(df)} synthetic patient records.")
    print("Sample Data:\n", df.head())

    # 2. Split Data (before one-hot encoding for easier subgroup analysis)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['diagnosis'])
    
    print(f"\nTraining model on {len(train_df)} samples, testing on {len(test_df)} samples.")

    # 3. Preprocessing: Convert categorical features to numerical using One-Hot Encoding
    # To ensure consistent columns, we first get all possible column names after OHE
    temp_df_for_cols = pd.get_dummies(df.drop('diagnosis', axis=1), columns=['gender'], drop_first=True)
    all_feature_names_after_ohe = temp_df_for_cols.columns.tolist()

    X_train_processed = pd.get_dummies(train_df.drop('diagnosis', axis=1), columns=['gender'], drop_first=True)
    y_train = train_df['diagnosis']

    X_test_processed = pd.get_dummies(test_df.drop('diagnosis', axis=1), columns=['gender'], drop_first=True)
    y_test = test_df['diagnosis']

    # Align columns between train and test sets in case a category was missing in one split
    missing_cols_in_test = set(X_train_processed.columns) - set(X_test_processed.columns)
    for c in missing_cols_in_test:
        X_test_processed[c] = 0
    X_test_processed = X_test_processed[X_train_processed.columns] # Ensure column order matches train

    # 4. Train Black-box Model
    model = train_black_box_model(X_train_processed, y_train)
    
    # Evaluate model performance
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Test Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # --- Interpretability Section ---

    # 5. Local Explanations (for a specific patient)
    print("\n--- Local Explanation for a specific patient ---")
    # Choose a random patient from the test set
    random_patient_idx_in_test_df = np.random.randint(0, len(X_test_processed))
    
    # Get the processed instance for prediction
    patient_instance_processed = X_test_processed.iloc[random_patient_idx_in_test_df].values
    
    local_explanation_text, _ = get_local_explanation(model, patient_instance_processed, all_feature_names_after_ohe)
    print(f"\nExplaining prediction for Patient (test set index) {random_patient_idx_in_test_df}:\n{local_explanation_text}")

    # 6. Global Explanations (overall model behavior)
    print("\n--- Global Feature Importance ---")
    global_importance_report = get_global_feature_importance(model, all_feature_names_after_ohe)
    print(global_importance_report)

    # 7. Subgroup Bias Diagnostics
    print("\n--- Subgroup Bias Diagnostics ---")
    categorical_features_for_bias = ['gender'] # Extend this if more categorical features are added
    
    bias_report_output = analyze_subgroup_bias(
        model, 
        X_test_processed, 
        y_test, 
        test_df.drop('diagnosis', axis=1), # Pass the raw test dataframe without target for subgrouping
        categorical_features_for_bias,
        all_feature_names_after_ohe
    )
    print(bias_report_output)

    print("\n--- Interactive System Integration (Conceptual) ---")
    print("The above components would be integrated into an interactive web interface (e.g., using Gradio or Streamlit).")
    print("Clinicians could:")
    print(" - Input new patient data to get a diagnosis and its local explanation.")
    print(" - Select different patient subgroups to compare model performance and identify biases.")
    print(" - Visualize global feature importance plots to understand overall model drivers.")
    print(" - Compare different model versions or explanation techniques.")
    print("This fosters human-in-the-loop exploration and debugging, leading to more trustworthy and responsible AI in healthcare.")

if __name__ == "__main__":
    run_explainer_system()