import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import itertools

def generate_disease_data(n_samples=1000, n_features=10, n_informative=5, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=random_state
    )

    feature_names = [f"feature_{i:02d}" for i in range(n_features - 2)]
    feature_names.insert(0, "Age")
    feature_names.insert(1, "BMI")
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="Disease_Risk")

    X_df["Age"] = (X_df["Age"] * 10 + 40).astype(int)
    X_df["BMI"] = (X_df["BMI"] * 5 + 25).round(1)
    
    for i in range(2, n_features):
        if i % 3 == 0:
            X_df[f"feature_{i:02d}"] = (X_df[f"feature_{i:02d}"] > 0).astype(int)

    return X_df, y_series

def train_black_box_model(X: pd.DataFrame, y: pd.Series, model_path="black_box_model.joblib", random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, model_path)

    return model

def load_black_box_model(model_path="black_box_model.joblib"):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        return None

class LaceExplainer:
    def __init__(self, black_box_model, training_data_X, feature_names):
        self.model = black_box_model
        self.training_data_X = training_data_X
        self.feature_names = feature_names
        self.nn_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
        self.nn_model.fit(training_data_X)

    def _get_local_data(self, instance, k):
        distances, indices = self.nn_model.kneighbors(instance.values.reshape(1, -1), n_neighbors=k + 1)
        local_indices = indices.flatten()[1:]
        local_X = self.training_data_X.iloc[local_indices]
        local_y_pred = self.model.predict(local_X)
        local_y_proba = self.model.predict_proba(local_X)
        return local_X, local_y_pred, local_y_proba

    def _train_local_surrogate(self, local_X, local_y_pred):
        surrogate_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        surrogate_model.fit(local_X, local_y_pred)
        return surrogate_model

    def _calculate_prediction_difference(self, instance, local_X, original_prediction_proba, target_class_idx, num_perturbations=100):
        contributions = {}

        for feature in self.feature_names:
            perturbed_samples = pd.DataFrame([instance.values.flatten()] * num_perturbations, columns=self.feature_names)
            perturbed_samples[feature] = local_X[feature].sample(n=num_perturbations, replace=True, random_state=42).values

            perturbed_probas = self.model.predict_proba(perturbed_samples)
            avg_perturbed_prob = np.mean(perturbed_probas[:, target_class_idx])
            contributions[feature] = original_prediction_proba[target_class_idx] - avg_perturbed_prob

        for f1, f2 in itertools.combinations(self.feature_names, 2):
            perturbed_samples = pd.DataFrame([instance.values.flatten()] * num_perturbations, columns=self.feature_names)
            perturbed_samples[f1] = local_X[f1].sample(n=num_perturbations, replace=True, random_state=42).values
            perturbed_samples[f2] = local_X[f2].sample(n=num_perturbations, replace=True, random_state=42).values

            perturbed_probas = self.model.predict_proba(perturbed_samples)
            avg_perturbed_prob = np.mean(perturbed_probas[:, target_class_idx])
            contributions[f"{f1} x {f2}"] = original_prediction_proba[target_class_idx] - avg_perturbed_prob

        return contributions

    def explain_instance(self, instance: pd.Series, k=10, num_perturbations=100):
        original_prediction_proba = self.model.predict_proba(instance.to_frame().T)[0]
        predicted_class = self.model.predict(instance.to_frame().T)[0]
        target_class_idx = np.argmax(original_prediction_proba)

        local_X, local_y_pred, _ = self._get_local_data(instance, k)
        
        surrogate_model = self._train_local_surrogate(local_X, local_y_pred)

        contributions = self._calculate_prediction_difference(
            instance, local_X, original_prediction_proba, target_class_idx, num_perturbations
        )

        return {
            "instance": instance.to_dict(),
            "original_prediction_proba": original_prediction_proba.tolist(),
            "predicted_class": predicted_class.item(),
            "contributions": contributions,
            "target_class_explained": target_class_idx.item()
        }

    def _plot_contributions(self, explanation_result, top_n=10):
        contributions = explanation_result["contributions"]
        sorted_contributions = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        top_contributions = sorted_contributions[:top_n]

        labels = [item[0] for item in top_contributions]
        values = [item[1] for item in top_contributions]

        y_pos = np.arange(len(labels))

        plt.figure(figsize=(10, 6))
        plt.barh(y_pos, values, align='center', color=['green' if v > 0 else 'red' for v in values])
        plt.yticks(y_pos, labels)
        plt.xlabel("Prediction Difference (Impact on P(Target Class))")
        plt.title(f"LACE Contributions for Instance (Class {explanation_result['predicted_class']})")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    X_data, y_data = generate_disease_data(n_samples=1000, n_features=12)

    black_box_model = train_black_box_model(X_data, y_data, model_path="black_box_disease_predictor.joblib")

    if black_box_model:
        explainer = LaceExplainer(black_box_model, X_data, X_data.columns.tolist())

        instance_to_explain_idx = 5
        instance_to_explain = X_data.iloc[instance_to_explain_idx]
        actual_label = y_data.iloc[instance_to_explain_idx]

        print(f"Explaining instance at index {instance_to_explain_idx} (Actual Label: {actual_label}):")
        print(instance_to_explain.to_string())

        explanation = explainer.explain_instance(instance_to_explain, k=15)
        print("\nLACE Explanation:")
        print(f"Original Prediction Probas: {explanation['original_prediction_proba']}")
        print(f"Predicted Class: {explanation['predicted_class']}")
        print(f"Target Class Explained: {explanation['target_class_explained']}")
        print("\nTop 10 Contributions:")
        sorted_contributions = sorted(explanation['contributions'].items(), key=lambda item: abs(item[1]), reverse=True)
        for feature, contribution in sorted_contributions[:10]:
            print(f"- {feature}: {contribution:.4f}")

        explainer._plot_contributions(explanation)
