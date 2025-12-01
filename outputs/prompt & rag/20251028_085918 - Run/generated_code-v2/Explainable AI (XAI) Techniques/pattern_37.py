import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # Example black-box model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class BlackBoxCreditModel:
    """
    A wrapper for the black-box credit scoring model.
    Assumes the underlying model has `predict_proba` and `predict` methods.
    """
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

class LACEExplainer:
    """
    Implements the LACE (Local Agnostic attribute Contribution Explanation) pattern
    to explain predictions of a black-box model for individual instances.
    """
    def __init__(self, black_box_model_wrapper, training_data, feature_names):
        """
        Initializes the LACEExplainer.

        Args:
            black_box_model_wrapper (BlackBoxCreditModel): Wrapped black-box model.
            training_data (pd.DataFrame): The dataset used for training the black-box model,
                                        or a representative sample, used to find neighbors.
            feature_names (pd.Index or list): Names of the features in the training_data.
        """
        self.black_box_model = black_box_model_wrapper
        self.training_data = training_data # Features (X_train) for neighborhood search
        self.feature_names = feature_names
        self.feature_means = training_data.mean(axis=0) # Used for marginalization (omission)

        # Initialize NearestNeighbors model. It will be fitted once.
        self.nn_model = NearestNeighbors(metric='euclidean')
        self.nn_model.fit(training_data)

    def _find_k_nearest_neighbors(self, instance, k):
        """
        Finds the K-nearest neighbors for a given instance in the training data.
        """
        # Reshape instance for NN query if it's a 1D array
        instance_reshaped = instance.reshape(1, -1) if instance.ndim == 1 else instance
        
        # Dynamically set n_neighbors to avoid fitting a new NN model every time
        self.nn_model.n_neighbors = min(k, len(self.training_data)) # Ensure k is not greater than data size
        
        distances, indices = self.nn_model.kneighbors(instance_reshaped)
        return self.training_data.iloc[indices[0]] # Return a DataFrame of neighbors

    def _train_local_surrogate(self, neighborhood_X, neighborhood_predictions_proba):
        """
        Trains a shallow Decision Tree as a local interpretable surrogate model.
        """
        # The target for the surrogate is the black-box model's predicted class label
        surrogate_y = np.argmax(neighborhood_predictions_proba, axis=1)

        local_surrogate = DecisionTreeClassifier(max_depth=3, random_state=42)
        local_surrogate.fit(neighborhood_X, surrogate_y)
        return local_surrogate

    def _calculate_prediction_difference(self, original_instance, target_class, features_to_perturb_indices):
        """
        Calculates the prediction difference by omitting (marginalizing) specified features.
        Omission is approximated by replacing feature values with their training data means.
        """
        original_proba = self.black_box_model.predict_proba(original_instance.reshape(1, -1))[0, target_class]

        perturbed_instance = original_instance.copy()
        for idx in features_to_perturb_indices:
            perturbed_instance[idx] = self.feature_means.iloc[idx] # Use .iloc for Series mean access

        perturbed_proba = self.black_box_model.predict_proba(perturbed_instance.reshape(1, -1))[0, target_class]

        return original_proba - perturbed_proba

    def explain_instance(self, instance_to_explain, k=10, target_class=None, top_k_interactions=3):
        """
        Generates a LACE explanation for a single instance.

        Args:
            instance_to_explain (np.array or pd.Series): The instance (features) to explain.
            k (int): Number of neighbors to consider for the local neighborhood.
            target_class (int, optional): The class label to explain the prediction for.
                                          If None, uses the black-box model's predicted class for the instance.
            top_k_interactions (int): Number of top individual features to consider for pairwise interactions.

        Returns:
            dict: A dictionary containing feature and interaction contributions (prediction differences).
        """
        # Ensure instance_to_explain is a numpy array for consistent indexing and reshaping
        if isinstance(instance_to_explain, pd.Series):
            instance_to_explain = instance_to_explain.values

        # 1. Get black-box prediction for the instance
        if target_class is None:
            target_class = self.black_box_model.predict(instance_to_explain.reshape(1, -1))[0]
        
        # 2. Find K-nearest neighbors
        local_neighborhood_X = self._find_k_nearest_neighbors(instance_to_explain, k)
        
        # 3. Get black-box predictions (probabilities) for the neighborhood
        local_neighborhood_predictions_proba = self.black_box_model.predict_proba(local_neighborhood_X)

        # 4. Train local surrogate model
        local_surrogate_model = self._train_local_surrogate(local_neighborhood_X, local_neighborhood_predictions_proba)

        contributions = {}
        
        # Calculate contributions for individual features
        feature_contributions_list = []
        for i, feature_name in enumerate(self.feature_names):
            pred_diff = self._calculate_prediction_difference(
                instance_to_explain, target_class, [i]
            )
            contributions[feature_name] = pred_diff
            feature_contributions_list.append((feature_name, pred_diff, i))
            
        # Consider pairwise interaction terms among top features identified by individual contributions
        # Sort features by absolute contribution to identify top ones
        sorted_features = sorted(feature_contributions_list, key=lambda item: abs(item[1]), reverse=True)
        
        # Select indices of the top features for interaction analysis
        top_feature_indices_for_interactions = [f[2] for f in sorted_features[:top_k_interactions]]

        # Consider pairwise interactions among these top features
        for i in range(len(top_feature_indices_for_interactions)):
            for j in range(i + 1, len(top_feature_indices_for_interactions)):
                idx1, idx2 = top_feature_indices_for_interactions[i], top_feature_indices_for_interactions[j]
                feature_combo_name = f"{self.feature_names[idx1]}_x_{self.feature_names[idx2]}"
                
                # Calculate prediction difference for the combined omission of both features
                pred_diff_interaction = self._calculate_prediction_difference(
                    instance_to_explain, target_class, [idx1, idx2]
                )
                
                # The LACE pattern describes 'prediction difference' upon omission.
                # For an interaction term, this can be interpreted as the combined effect of removing both.
                # A more precise interaction effect would subtract individual effects, but for direct 'contribution upon omission', this is sufficient.
                contributions[feature_combo_name] = pred_diff_interaction

        return contributions

# Example Usage (Demonstration):
# This part would typically be in a separate script or a notebook to run the explainer.
if __name__ == "__main__":
    # Generate some synthetic data for a credit scoring model
    np.random.seed(42)
    num_samples = 1000
    data = {
        'credit_score': np.random.randint(300, 850, num_samples),
        'debt_to_income': np.random.rand(num_samples) * 0.5 + 0.1, # 10-60%
        'loan_amount': np.random.randint(1000, 50000, num_samples),
        'employment_years': np.random.randint(0, 30, num_samples),
        'income': np.random.randint(20000, 200000, num_samples)
    }
    df = pd.DataFrame(data)

    # Create a synthetic target variable (loan approval)
    # Simple rule: High credit score, low DTI, stable employment -> approved
    df['approved'] = ((df['credit_score'] > 650) &
                      (df['debt_to_income'] < 0.3) &
                      (df['employment_years'] > 2) &
                      (df['loan_amount'] < df['income'] * 0.5)).astype(int)

    X = df[['credit_score', 'debt_to_income', 'loan_amount', 'employment_years', 'income']]
    y = df['approved']

    # Train a black-box model (Random Forest)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    black_box_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    black_box_rf.fit(X_train, y_train)

    print(f"Black-box model accuracy: {accuracy_score(y_test, black_box_rf.predict(X_test)):.4f}")

    # Wrap the black-box model
    credit_model_wrapper = BlackBoxCreditModel(black_box_rf)

    # Initialize LACE explainer
    lace_explainer = LACEExplainer(credit_model_wrapper, X_train, X.columns)

    # Choose an instance to explain (e.g., a challenging one from test set)
    instance_to_explain_idx = 5
    instance_to_explain_df = X_test.iloc[instance_to_explain_idx]
    instance_to_explain_array = instance_to_explain_df.values
    actual_label = y_test.iloc[instance_to_explain_idx]
    predicted_label = credit_model_wrapper.predict(instance_to_explain_array.reshape(1, -1))[0]

    print(f"\nExplaining instance: {instance_to_explain_df.to_dict()}")
    print(f"Actual loan approval: {actual_label}, Predicted loan approval: {predicted_label}")

    # Generate explanation
    explanation = lace_explainer.explain_instance(instance_to_explain_array, k=20, target_class=predicted_label, top_k_interactions=3)

    print("\nLACE Explanations (Prediction Difference for Predicted Class):")
    print("  (Positive value means feature/interaction contributes to predicted class probability)")
    for feature, contribution in sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True):
        print(f"  {feature}: {contribution:.4f}")

    # Example of an instance that was predicted differently from actual, if available
    # Find an incorrectly predicted instance
    misclassified_indices = np.where(black_box_rf.predict(X_test) != y_test)[0]
    if len(misclassified_indices) > 0:
        print("\n--- Explaining a Misclassified Instance ---")
        mis_idx = misclassified_indices[0]
        mis_instance_df = X_test.iloc[mis_idx]
        mis_instance_array = mis_instance_df.values
        mis_actual_label = y_test.iloc[mis_idx]
        mis_predicted_label = credit_model_wrapper.predict(mis_instance_array.reshape(1, -1))[0]

        print(f"\nExplaining misclassified instance: {mis_instance_df.to_dict()}")
        print(f"Actual loan approval: {mis_actual_label}, Predicted loan approval: {mis_predicted_label}")

        mis_explanation = lace_explainer.explain_instance(mis_instance_array, k=20, target_class=mis_predicted_label, top_k_interactions=3)
        print("\nLACE Explanations (Prediction Difference for Predicted Class):")
        for feature, contribution in sorted(mis_explanation.items(), key=lambda item: abs(item[1]), reverse=True):
            print(f"  {feature}: {contribution:.4f}")
    else:
        print("\nNo misclassified instances found in the test set for demonstration.")
