import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.inspection import plot_partial_dependence, plot_ice, permutation_importance
import shap
import dice_ml

# Set random seed for reproducibility
np.random.seed(42)

# 1. Data Module: Synthetic Data Generation
def generate_synthetic_data(n_samples=1000):
    """Generates synthetic patient medical data."""
    data = {
        'age': np.random.randint(20, 80, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'symptom_fever': np.random.randint(0, 2, n_samples),
        'symptom_cough': np.random.randint(0, 2, n_samples),
        'symptom_fatigue': np.random.randint(0, 2, n_samples),
        'lab_result_crp': np.random.normal(5, 3, n_samples).clip(0, 20),
        'lab_result_wbc': np.random.normal(8, 2, n_samples).clip(2, 15),
        'medical_history_diabetes': np.random.randint(0, 2, n_samples),
        'medical_history_hypertension': np.random.randint(0, 2, n_samples),
        'diagnosis': np.random.randint(0, 2, n_samples) # 0: No Disease, 1: Disease
    }
    df = pd.DataFrame(data)

    # Introduce some correlations to make the data more realistic
    df.loc[df['symptom_fever'] == 1, 'lab_result_crp'] = np.random.normal(10, 5, df['symptom_fever'].sum()).clip(0, 20)
    df.loc[(df['symptom_fever'] == 1) & (df['lab_result_crp'] > 7), 'diagnosis'] = 1
    df.loc[df['age'] > 60, 'medical_history_hypertension'] = np.random.choice([0, 1], df['age'].gt(60).sum(), p=[0.3, 0.7])
    df.loc[(df['medical_history_diabetes'] == 1) | (df['medical_history_hypertension'] == 1), 'diagnosis'] = np.random.choice([0, 1], (df['medical_history_diabetes'] == 1) | (df['medical_history_hypertension'] == 1).sum(), p=[0.2, 0.8])
    df.loc[(df['diagnosis'] == 0) & (df['symptom_fever'] == 1) & (df['lab_result_crp'] < 5), 'diagnosis'] = 0

    # Ensure some balance in diagnosis for demonstration
    num_diseased = df['diagnosis'].sum()
    if num_diseased < n_samples * 0.3:
        mask_to_change = df['diagnosis'] == 0
        change_count = int(n_samples * 0.3) - num_diseased
        if change_count > 0:
            change_indices = df[mask_to_change].sample(n=min(change_count, len(df[mask_to_change])), random_state=42).index
            df.loc[change_indices, 'diagnosis'] = 1
    return df


# 2. Prediction Model Module: Preprocessing and Model Training
def create_and_train_model(X_train, y_train, X_test, y_test):
    """Creates a preprocessing pipeline, trains a RandomForestClassifier, and evaluates it."""
    # Define categorical and numerical features
    categorical_features = ['gender']
    numerical_features = [
        'age', 'symptom_fever', 'symptom_cough', 'symptom_fatigue',
        'lab_result_crp', 'lab_result_wbc', 'medical_history_diabetes',
        'medical_history_hypertension'
    ]

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine preprocessors using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the full pipeline with preprocessor and RandomForestClassifier
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_score(y_test, y_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return model_pipeline, numerical_features, categorical_features


# 3. Interpretability & Debugging Framework Module
class InterpretabilityFramework:
    def __init__(self, model_pipeline, X_train_raw, numerical_features, categorical_features):
        self.model_pipeline = model_pipeline
        self.X_train_raw = X_train_raw
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.feature_names = numerical_features + list(model_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))

        # Initialize SHAP Explainer (using TreeExplainer for RandomForest)
        # Note: SHAP needs the *trained* classifier and preprocessed data
        self.explainer = shap.TreeExplainer(model_pipeline.named_steps['classifier'])
        self.shap_values_raw = self.explainer.shap_values(model_pipeline.named_steps['preprocessor'].transform(X_train_raw))

        # Initialize DiCE for Counterfactual Explanations
        # Dice requires a data_interface and a model_interface
        self.d = dice_ml.Data(dataframe=self.X_train_raw, continuous_features=self.numerical_features, outcome_name='diagnosis')
        self.m = dice_ml.Model(model=self.model_pipeline, backend='sklearn')
        self.exp = dice_ml.Dice(self.d, self.m, method='random') # Using 'random' method for simplicity

    def get_feature_names(self):
        """Returns the list of all feature names, including one-hot encoded ones."""
        return self.feature_names

    def explain_local_shap(self, X_instance_raw, instance_id=None):
        """Provides instance-specific SHAP explanations."""
        print(f"\n--- Local SHAP Explanation for Instance {instance_id if instance_id is not None else ''} ---")
        # Transform the instance using the preprocessor
        X_instance_processed = self.model_pipeline.named_steps['preprocessor'].transform(X_instance_raw)

        # Calculate SHAP values for the processed instance
        # shap_values_instance = self.explainer.shap_values(X_instance_processed)
        # For binary classification, shap_values_instance will be a list of two arrays. We take the values for the 'positive' class (index 1)
        # Ensure we get the correct output for SHAP (often a list of arrays for multi-output models or binary classification)
        if isinstance(self.shap_values_raw, list):
            shap_values_instance = self.explainer.shap_values(X_instance_processed)[1] # For positive class
        else:
            shap_values_instance = self.explainer.shap_values(X_instance_processed)

        # Create a SHAP Explanation object for visualization
        # base_values_instance = self.explainer.expected_value
        if isinstance(self.explainer.expected_value, list):
            base_values_instance = self.explainer.expected_value[1]
        else:
            base_values_instance = self.explainer.expected_value

        # Original values of the instance in the preprocessed space
        # To get feature values for the plot, we need to map back to original space or use the transformed values
        # For SHAP force plot, it's often easier to use a shap.Explanation object
        
        # Create a dummy shap.Explanation object for the instance
        # This requires the base value, shap values for the instance, and original feature values
        # The feature values should be in the same order as self.feature_names

        # Create a DataFrame from the processed instance to match feature names
        X_instance_processed_df = pd.DataFrame(X_instance_processed, columns=self.feature_names)
        
        shap.initjs()
        shap.force_plot(
            base_values_instance, 
            shap_values_instance, 
            X_instance_processed_df,
            feature_names=self.feature_names,
            matplotlib=True, 
            show=False
        )
        plt.title(f'SHAP Force Plot for Instance {instance_id if instance_id is not None else ''}')
        plt.tight_layout()
        plt.show()

        # Also show a waterfall plot for more detail
        shap.plots.waterfall(shap.Explanation(
            values=shap_values_instance,
            base_values=base_values_instance,
            data=X_instance_processed_df.iloc[0].values,
            feature_names=self.feature_names
        ), show=False)
        plt.title(f'SHAP Waterfall Plot for Instance {instance_id if instance_id is not None else ''}')
        plt.tight_layout()
        plt.show()


    def plot_individual_conditional_expectation(self, X_instance_raw, feature, instance_id=None):
        """Generates and plots Individual Conditional Expectation (ICE) for a specific instance and feature."""
        print(f"\n--- ICE Plot for Feature '{feature}' on Instance {instance_id if instance_id is not None else ''} ---")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # plot_ice expects the full training data and an instance to highlight
        # It handles the preprocessing within the pipeline
        plot_ice(self.model_pipeline, self.X_train_raw, feature, 
                 target_names=['No Disease', 'Disease'], 
                 ax=ax, 
                 random_state=42,
                 line_kw={'color': 'grey', 'alpha': 0.3},
                 plot_lines_kw={'c': 'red', 'linewidth': 3, 'label': f'Highlighted Instance {instance_id if instance_id is not None else ''}'}
                )
        
        # Highlight the specific instance manually if plot_ice doesn't directly support highlighting a specific X_instance_raw
        # For plot_ice, we pass the original training data and it will select an instance from there
        # To highlight a specific arbitrary instance, we need to manually plot its prediction curve
        # This is complex as it requires inverse transforming feature values or re-running prediction for a range.
        # Instead, we will rely on plot_ice to show the overall trend and pick an instance *from X_train_raw* for highlight.

        # Let's find an index in X_train_raw that matches X_instance_raw for highlighting if needed
        # For simplicity, we assume X_instance_raw is part of X_train_raw or similar distribution.
        # plot_ice will plot all lines in X_train_raw and highlight one (randomly if not specified by index)
        # If we want to specifically highlight X_instance_raw, we might need a custom ICE plotting function
        # For demonstration, we'll let plot_ice handle highlighting an instance from X_train_raw.

        # To make it truly highlight X_instance_raw, we need to simulate. This is a simplification.
        # The plot_ice function from sklearn.inspection is designed to work on the dataset given, not an arbitrary single instance easily.
        # We will plot the ICE plot for *all* instances in X_train_raw and let it pick one to highlight.

        plt.title(f'Individual Conditional Expectation (ICE) for {feature}')
        plt.xlabel(feature)
        plt.ylabel('Predicted Probability of Disease')
        plt.legend()
        plt.grid(True)
        plt.show()


    def generate_counterfactual_explanations(self, X_instance_raw, desired_class=0, instance_id=None):
        """Generates counterfactual explanations for a given instance using DiCE."""
        print(f"\n--- Counterfactual Explanations for Instance {instance_id if instance_id is not None else ''} (Desired Class: {desired_class}) ---")
        # DiCE expects the input instance as a DataFrame
        query_instance = pd.DataFrame(X_instance_raw, columns=self.X_train_raw.columns)

        # Generate counterfactuals
        dice_exp = self.exp.generate_counterfactuals(
            query_instance,
            total_CFs=3, # Number of counterfactuals to generate
            desired_class=desired_class # Desired outcome (e.g., 0 for no disease)
        )

        # Print counterfactuals
        print("Original Instance:")
        print(query_instance.to_string())
        print(f"Original Prediction: {self.model_pipeline.predict(query_instance)[0]} (Probability: {self.model_pipeline.predict_proba(query_instance)[0][1]:.4f})")
        print("\nGenerated Counterfactuals (what needs to change for desired outcome):")
        dice_exp.visualize_as_dataframe(show_only_changes=True)


    def plot_global_partial_dependence(self, features_to_plot):
        """Generates and plots Global Partial Dependence Plots (PDPs)."""
        print("\n--- Global Partial Dependence Plots ---")
        fig, ax = plt.subplots(figsize=(15, 8))
        plot_partial_dependence(self.model_pipeline, self.X_train_raw, features_to_plot,
                                feature_names=list(self.X_train_raw.columns),
                                target=1, # Probability of 'Disease'
                                ax=ax, grid_resolution=50)
        fig.suptitle('Partial Dependence Plots (Probability of Disease)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()


    def plot_permutation_feature_importance(self, X_test_raw, y_test):
        """Calculates and plots Permutation Feature Importance."""
        print("\n--- Permutation Feature Importance ---")
        result = permutation_importance(
            self.model_pipeline, X_test_raw, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        sorted_idx = result.importances_mean.argsort()

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.boxplot(result.importances[sorted_idx].T,
                   vert=False, labels=self.X_train_raw.columns[sorted_idx])
        ax.set_title("Permutation Feature Importances")
        ax.set_xlabel("Importance Score (Decrease in Accuracy)")
        plt.tight_layout()
        plt.show()


# Main execution block
if __name__ == "__main__":
    print("Generating synthetic patient data...")
    df = generate_synthetic_data(n_samples=2000) # Increased samples for better interpretability results
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Creating and training the AI disease diagnosis model...")
    model_pipeline, numerical_features, categorical_features = create_and_train_model(X_train, y_train, X_test, y_test)

    print("\nInitializing Interpretability Framework...")
    interpret_framework = InterpretabilityFramework(model_pipeline, X_train, numerical_features, categorical_features)

    # --- Demonstrating Interpretability Methods ---

    # 1. Local Interpretability: SHAP Explanations
    print("\n--- Demonstrating Local Interpretability: SHAP Explanations ---")
    # Select an instance from the test set for explanation
    test_instance_id = 5
    X_single_instance_raw = X_test.iloc[test_instance_id:test_instance_id+1]
    print(f"Explaining prediction for patient instance:\n{X_single_instance_raw.to_string()}")
    predicted_class = model_pipeline.predict(X_single_instance_raw)[0]
    predicted_proba = model_pipeline.predict_proba(X_single_instance_raw)[0][1]
    print(f"Predicted Diagnosis: {'Disease' if predicted_class == 1 else 'No Disease'} (Probability: {predicted_proba:.4f})")
    interpret_framework.explain_local_shap(X_single_instance_raw, instance_id=test_instance_id)

    # 2. Local Interpretability: ICE Plots
    print("\n--- Demonstrating Local Interpretability: ICE Plots ---")
    # Plot ICE for 'age' for the same instance
    interpret_framework.plot_individual_conditional_expectation(X_single_instance_raw, feature='age', instance_id=test_instance_id)
    # Plot ICE for 'lab_result_crp'
    interpret_framework.plot_individual_conditional_expectation(X_single_instance_raw, feature='lab_result_crp', instance_id=test_instance_id)


    # 3. Local Interpretability: Counterfactual Explanations (DiCE)
    print("\n--- Demonstrating Local Interpretability: Counterfactual Explanations ---")
    # Try to change the prediction of the selected instance to 'No Disease' (class 0)
    interpret_framework.generate_counterfactual_explanations(X_single_instance_raw, desired_class=0, instance_id=test_instance_id)


    # 4. Global Interpretability: Partial Dependence Plots (PDPs)
    print("\n--- Demonstrating Global Interpretability: Partial Dependence Plots ---")
    # Plot PDPs for a few key features
    interpret_framework.plot_global_partial_dependence(
        features_to_plot=['age', 'lab_result_crp', 'symptom_fever', 'gender', 
                          ('age', 'lab_result_crp')] # Example of 2-way interaction
    )

    # 5. Global Interpretability: Permutation Feature Importance
    print("\n--- Demonstrating Global Interpretability: Permutation Feature Importance ---")
    interpret_framework.plot_permutation_feature_importance(X_test, y_test)

    print("\nAI Interpretability & Debugging Framework Demonstration Complete.")
