import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class HealthcareFraudDetector:
    """A dummy black-box classifier for healthcare fraud detection."""
    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

class LACEExplainer:
    """Implements the LACE (Local Agnostic attribute Contribution Explanation) pattern.
    Provides local, interpretable explanations for black-box model predictions.
    """
    def __init__(self, black_box_model, training_data, feature_names):
        self.black_box_model = black_box_model
        self.training_data = training_data # Full training data for default values
        self.feature_names = feature_names
        self.default_values = self._get_default_values(training_data)

    def _get_default_values(self, data):
        """Calculates default (median/mode) values for perturbation."""
        default_vals = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                default_vals[col] = data[col].median()
            else:
                default_vals[col] = data[col].mode()[0]
        return default_vals

    def find_k_nearest_neighbors(self, instance, local_data, k):
        """Finds K-nearest neighbors for a given instance in the local_data.
        local_data typically consists of the original training data or a perturbed sample.
        """
        if k > len(local_data):
            k = len(local_data) # Adjust k if it's larger than available data
            
        # Use NearestNeighbors from sklearn to find closest instances
        # Assuming numerical features for distance calculation
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(local_data)
        distances, indices = nn.kneighbors(instance.reshape(1, -1))
        return local_data.iloc[indices[0]]

    def train_local_surrogate(self, local_neighbors_X, black_box_predictions_proba):
        """Trains a rule-based local surrogate model (Decision Tree) on the neighbors.
        The surrogate model aims to mimic the black-box model's behavior locally.
        """
        # We train the Decision Tree to predict the probability of the positive class
        local_surrogate = DecisionTreeClassifier(max_depth=5, random_state=42) # Limit depth for interpretability
        local_surrogate.fit(local_neighbors_X, black_box_predictions_proba)
        return local_surrogate

    def _calculate_individual_contribution(self, instance_df, feature_to_perturb):
        """Calculates the prediction difference for a single feature.
        Compares original prediction with prediction when feature is set to default.
        """
        original_prediction_proba = self.black_box_model.predict_proba(instance_df)[:, 1][0]

        perturbed_instance_df = instance_df.copy()
        if feature_to_perturb in self.default_values:
            perturbed_instance_df[feature_to_perturb] = self.default_values[feature_to_perturb]
        else:
            # Handle cases where feature_to_perturb might not be in default_values (e.g., if it's the target)
            return 0 # Or raise an error, depending on desired behavior
        
        perturbed_prediction_proba = self.black_box_model.predict_proba(perturbed_instance_df)[:, 1][0]
        
        # Contribution: how much the feature *contributes* to the original prediction
        # If original is high and perturbed is low, the feature increased the prob
        return original_prediction_proba - perturbed_prediction_proba

    def explain_instance(self, instance, k=10):
        """Generates a LACE explanation for a single instance.
        Returns qualitative rules and quantitative contributions.
        """
        instance_df = pd.DataFrame(instance.reshape(1, -1), columns=self.feature_names)

        # Step 1: Find K-nearest neighbors
        # Use the training_data as the reference for neighbors
        k_neighbors_df = self.find_k_nearest_neighbors(instance, self.training_data, k)
        
        # Append the instance itself to the neighbors to ensure local model has context
        local_data_for_surrogate = pd.concat([k_neighbors_df, instance_df])
        
        # Step 2: Get black-box predictions for neighbors
        black_box_local_predictions_proba = self.black_box_model.predict_proba(local_data_for_surrogate)[:, 1]

        # Step 3: Train local rule-based surrogate model
        local_surrogate_model = self.train_local_surrogate(local_data_for_surrogate, black_box_local_predictions_proba)

        # Step 4: Extract qualitative rules from the local surrogate
        # We want rules specific to the instance's prediction path
        instance_prediction_class = self.black_box_model.predict(instance_df)[0]
        
        # A more robust way to get rules for an instance from Decision Tree
        # Traverse the tree to find the leaf node for the instance
        # For simplicity, we'll extract the full tree text and highlight relevant parts conceptually.
        tree_rules_text = export_text(local_surrogate_model, feature_names=list(self.feature_names))
        qualitative_explanation = f"Local rules from surrogate model (predicting black-box probability of class 1):\n{tree_rules_text}"
        
        # Step 5: Quantify influence of individual attributes via prediction difference
        quantitative_contributions = {}
        for feature in self.feature_names:
            contribution = self._calculate_individual_contribution(instance_df, feature)
            quantitative_contributions[feature] = contribution
            
        # Sort contributions for better visualization
        sorted_contributions = sorted(quantitative_contributions.items(), key=lambda item: abs(item[1]), reverse=True)

        return qualitative_explanation, sorted_contributions

    def plot_explanation(self, contributions, title="Feature Contributions to Fraud Prediction"):
        """Visualizes feature contributions with a bar plot."""
        features = [item[0] for item in contributions]
        values = [item[1] for item in contributions]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=values, y=features, palette="viridis")
        plt.title(title)
        plt.xlabel("Prediction Difference (Original - Perturbed)")
        plt.ylabel("Feature")
        plt.axvline(x=0, color='grey', linestyle='--')
        plt.show()

# --- Main Application Logic --- 
if __name__ == "__main__":
    # 1. Simulate Healthcare Claims Data
    print("1. Simulating Healthcare Claims Data...")
    np.random.seed(42)
    num_samples = 1000
    
    data = {
        'Age': np.random.randint(20, 80, num_samples),
        'ClaimAmount': np.random.uniform(100, 5000, num_samples),
        'NumProcedures': np.random.randint(1, 10, num_samples),
        'ProviderRating': np.random.uniform(1, 5, num_samples),
        'DiagnosisCodeCategory': np.random.choice(['A', 'B', 'C', 'D'], num_samples),
        'ClaimDurationDays': np.random.randint(1, 90, num_samples),
        'IsFraud': np.zeros(num_samples, dtype=int) # Default to not fraud
    }
    df = pd.DataFrame(data)

    # Introduce some synthetic fraud patterns
    # Pattern 1: High claim amount, low provider rating, many procedures
    fraud_idx1 = df[(df['ClaimAmount'] > 4000) & (df['ProviderRating'] < 2) & (df['NumProcedures'] > 7)].sample(frac=0.6, random_state=42).index
    df.loc[fraud_idx1, 'IsFraud'] = 1
    # Pattern 2: Older age, specific diagnosis, very long claim duration
    fraud_idx2 = df[(df['Age'] > 65) & (df['DiagnosisCodeCategory'] == 'D') & (df['ClaimDurationDays'] > 70)].sample(frac=0.7, random_state=42).index
    df.loc[fraud_idx2, 'IsFraud'] = 1

    print(f"Generated {df['IsFraud'].sum()} fraud cases out of {num_samples} total.")

    # Preprocessing: One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=['DiagnosisCodeCategory'], drop_first=True)
    
    X = df_encoded.drop('IsFraud', axis=1)
    y = df_encoded['IsFraud']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Train a Black-box Fraud Detection Model
    print("\n2. Training Black-box Fraud Detection Model...")
    fraud_detector = HealthcareFraudDetector()
    fraud_detector.fit(X_train, y_train)

    y_pred = fraud_detector.predict(X_test)
    print(f"Model Accuracy on test set: {accuracy_score(y_test, y_pred):.4f}")

    # 3. Select an instance to explain (e.g., a predicted fraud case)
    print("\n3. Selecting an instance for explanation...")
    fraud_cases_in_test = X_test[y_pred == 1]
    if not fraud_cases_in_test.empty:
        instance_to_explain_df = fraud_cases_in_test.sample(1, random_state=42)
        instance_to_explain_original = df.loc[instance_to_explain_df.index[0]] # Get original df row for display
        instance_to_explain = instance_to_explain_df.values[0]
        print("\nInstance to explain (original features):")
        print(instance_to_explain_original)
        print(f"Black-box model predicts fraud probability: {fraud_detector.predict_proba(instance_to_explain_df)[:, 1][0]:.4f}")

        # 4. Initialize and use LACE Explainer
        print("\n4. Generating LACE Explanation...")
        lace_explainer = LACEExplainer(fraud_detector, X_train, feature_names)
        qualitative_rules, quantitative_contributions = lace_explainer.explain_instance(instance_to_explain, k=20)

        print("\n--- Qualitative Explanation (Local Rules) ---")
        print(qualitative_rules)

        print("\n--- Quantitative Explanation (Feature Contributions) ---")
        for feature, contrib in quantitative_contributions:
            print(f"  {feature}: {contrib:.4f}")
            
        # 5. Visualize Contributions
        print("\n5. Visualizing Feature Contributions...")
        lace_explainer.plot_explanation(quantitative_contributions)

    else:
        print("No fraud cases predicted in the test set to explain.")
