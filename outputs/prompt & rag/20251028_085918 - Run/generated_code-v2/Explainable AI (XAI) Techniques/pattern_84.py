import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import binomtest
from mlxtend.frequent_patterns import apriori, association_rules


class BlackBoxModelWrapper:
    def __init__(self, model=None):
        if model is None:
            self.model = DummyClassifier(strategy="uniform", random_state=42)
        else:
            self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def load_and_preprocess_data(filepath="dummy_data.csv"):
    if filepath == "dummy_data.csv":
        data = pd.DataFrame({
            "AgeGroup": np.random.choice(["0-18", "19-45", "46-65", "65+"], size=1000),
            "Gender": np.random.choice(["Male", "Female"], size=1000),
            "DiseaseSeverity": np.random.choice(["Mild", "Moderate", "Severe"], size=1000),
            "Comorbidity": np.random.choice(["None", "Diabetes", "Hypertension", "Both"], size=1000),
            "TreatmentOutcome": np.random.choice([0, 1], size=1000, p=[0.7, 0.3]) # 0: Failure, 1: Success
        })
    else:
        data = pd.read_csv(filepath)

    target_column = "TreatmentOutcome"
    feature_columns = [col for col in data.columns if col != target_column]

    categorical_features = data[feature_columns].select_dtypes(include=["object", "category"]).columns
    numerical_features = data[feature_columns].select_dtypes(include=np.number).columns

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_features = encoder.fit_transform(data[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=data.index)

    preprocessed_data = pd.concat([encoded_df, data[numerical_features]], axis=1)
    preprocessed_data[target_column] = data[target_column]

    return preprocessed_data, data, target_column, list(encoded_feature_names) + list(numerical_features)


def calculate_h_divergence(subgroup_predictions, overall_predictions, true_labels_subgroup, true_labels_overall, behavior_metric="FPR"):
    def calculate_fpr(predictions, true_labels):
        fp = np.sum((predictions == 1) & (true_labels == 0))
        tn_plus_fp = np.sum(true_labels == 0)
        return fp / tn_plus_fp if tn_plus_fp > 0 else 0.0
    
    def calculate_fnr(predictions, true_labels):
        fn = np.sum((predictions == 0) & (true_labels == 1))
        tp_plus_fn = np.sum(true_labels == 1)
        return fn / tp_plus_fn if tp_plus_fn > 0 else 0.0

    if behavior_metric == "FPR":
        subgroup_metric = calculate_fpr(subgroup_predictions, true_labels_subgroup)
        overall_metric = calculate_fpr(overall_predictions, true_labels_overall)
    elif behavior_metric == "FNR":
        subgroup_metric = calculate_fnr(subgroup_predictions, true_labels_subgroup)
        overall_metric = calculate_fnr(overall_predictions, true_labels_overall)
    else:
        raise ValueError("Unsupported behavior_metric. Choose 'FPR' or 'FNR'.")

    return abs(subgroup_metric - overall_metric)


def is_significant(subgroup_metric_value, overall_metric_value, subgroup_size, overall_size, confidence_level=0.95):
    if subgroup_size < 10 or overall_size < 10: # Minimum size for statistical test
        return False
    
    # Simplified significance test for difference in proportions (e.g., FPR/FNR)
    # Using binomial test for one group against a reference probability
    # A more robust test would be a chi-squared test or z-test for two proportions

    # For FPR/FNR, the 