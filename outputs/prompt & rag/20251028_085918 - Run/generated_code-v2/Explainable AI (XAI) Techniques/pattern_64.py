import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors

class LACEExplainer:
    def __init__(self, black_box_model, X_train, feature_names):
        self.black_box_model = black_box_model
        self.X_train = X_train
        self.feature_names = feature_names
        self.knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean') # Fixed K for POC
        self.knn_model.fit(X_train)

    def _get_local_neighborhood(self, instance):
        distances, indices = self.knn_model.kneighbors(instance.reshape(1, -1))
        return self.X_train.iloc[indices[0]]

    def _train_surrogate_model(self, local_neighborhood, local_labels):
        surrogate_model = DecisionTreeClassifier(max_depth=3) # Simple tree for interpretability
        surrogate_model.fit(local_neighborhood, local_labels)
        return surrogate_model

    def _calculate_prediction_difference(self, instance, black_box_prediction_proba, surrogate_model, local_neighborhood):
        contributions = {}
        original_proba = black_box_prediction_proba[0]

        for i, feature in enumerate(self.feature_names):
            # Create a perturbed instance by marginalizing (averaging) the feature value
            # For simplicity, we'll replace the feature with its mean in the local neighborhood
            perturbed_instance = instance.copy()
            perturbed_instance[i] = local_neighborhood.iloc[:, i].mean()

            perturbed_proba = self.black_box_model.predict_proba(perturbed_instance.reshape(1, -1))[0]
            contributions[feature] = original_proba - perturbed_proba

        # For interaction patterns, a full LACE implementation would extract rules from the surrogate
        # and calculate their combined prediction difference. For this POC, we focus on individual features.
        return contributions

    def explain_instance(self, instance, instance_label):
        # Get black-box prediction for the instance
        black_box_prediction_proba = self.black_box_model.predict_proba(instance.reshape(1, -1))[0]
        black_box_prediction_class = self.black_box_model.predict(instance.reshape(1, -1))[0]

        # Get local neighborhood
        local_neighborhood_df = self._get_local_neighborhood(instance)
        local_labels = self.black_box_model.predict(local_neighborhood_df)

        # Train local surrogate model
        surrogate_model = self._train_surrogate_model(local_neighborhood_df, local_labels)

        # Calculate prediction differences (contributions)
        contributions = self._calculate_prediction_difference(instance, black_box_prediction_proba, surrogate_model, local_neighborhood_df)

        explanation_text = f"\nExplanation for Instance (Actual Label: {instance_label}, Black-Box Prediction: {black_box_prediction_class}, Probabilities: {black_box_prediction_proba}):\n"
        explanation_text += "Individual Feature Contributions (Prediction Difference):\n"
        for feature, diff in contributions.items():
            explanation_text += f"  - {feature}: {diff[int(black_box_prediction_class)]:.4f}\n"

        return contributions, explanation_text, black_box_prediction_class, black_box_prediction_proba

def visualize_contributions(contributions, instance_id, predicted_class, black_box_probas):
    features = list(contributions.keys())
    class_0_diffs = [diff[0] for diff in contributions.values()]
    class_1_diffs = [diff[1] for diff in contributions.values()]

    width = 0.35
    x = np.arange(len(features))

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, class_0_diffs, width, label='Contribution to Class 0')
    rects2 = ax.bar(x + width/2, class_1_diffs, width, label='Contribution to Class 1')

    ax.set_ylabel('Prediction Difference (Change in Probability)')
    ax.set_title(f'LACE Feature Contributions for Instance {instance_id} (Predicted Class: {predicted_class})')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Data Simulation/Preparation Layer
    print("1. Generating synthetic patient data...")
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, n_classes=2, random_state=42)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')

    # Split data (simplified for POC, usually train/test split)
    X_train_df = X_df.copy()
    y_train_series = y_series.copy()

    # 2. Black-Box Diagnosis Model Layer
    print("2. Training Black-Box RandomForestClassifier...")
    black_box_model = RandomForestClassifier(n_estimators=100, random_state=42)
    black_box_model.fit(X_train_df, y_train_series)

    # 3. LACE Explainer Layer
    print("3. Initializing LACE Explainer...")
    lace_explainer = LACEExplainer(black_box_model, X_train_df, feature_names)

    # 4. API/Interface Layer & Explanation Request
    print("4. Explaining a specific patient instance...")
    # Select an instance to explain (e.g., the 5th instance from the dataset)
    instance_to_explain_idx = 4
    instance_to_explain = X_df.iloc[instance_to_explain_idx].values
    actual_label = y_series.iloc[instance_to_explain_idx]

    contributions, explanation_text, predicted_class, black_box_probas = lace_explainer.explain_instance(instance_to_explain, actual_label)

    print(explanation_text)
    print(f"Black-Box Model Predicted Probabilities: Class 0: {black_box_probas[0]:.4f}, Class 1: {black_box_probas[1]:.4f}")

    # 5. Visualization Layer
    print("5. Visualizing feature contributions...")
    visualize_contributions(contributions, instance_to_explain_idx, predicted_class, black_box_probas)
