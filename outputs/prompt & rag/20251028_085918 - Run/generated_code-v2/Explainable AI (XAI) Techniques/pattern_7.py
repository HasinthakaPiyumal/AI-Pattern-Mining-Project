import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

def calculate_permutation_importance(
    model, X: pd.DataFrame, y: pd.Series, metric_func=accuracy_score, n_repeats: int = 5
) -> dict:
    """
    Calculates Permutation Feature Importance for a black-box model.

    Args:
        model: The trained black-box machine learning model.
        X (pd.DataFrame): The feature matrix (validation or test set).
        y (pd.Series): The true target values for the feature matrix.
        metric_func (callable): The performance metric function (e.g., accuracy_score, roc_auc_score).
                                 It should take (y_true, y_pred) or (y_true, y_proba) as input.
        n_repeats (int): Number of times to repeat the permutation for each feature to get a more robust estimate.

    Returns:
        dict: A dictionary where keys are feature names and values are their
              permutation importance scores (mean decrease in performance).
    """
    if hasattr(model, 'predict_proba') and len(np.unique(y)) == 2: # Binary classification for AUC
        baseline_preds = model.predict_proba(X)[:, 1]
    else:
        baseline_preds = model.predict(X)

    baseline_score = metric_func(y, baseline_preds)
    feature_importances = {}

    for feature in X.columns:
        feature_scores = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            # Permute the values of the current feature
            X_shuffled[feature] = np.random.permutation(X[feature])

            if hasattr(model, 'predict_proba') and len(np.unique(y)) == 2:
                shuffled_preds = model.predict_proba(X_shuffled)[:, 1]
            else:
                shuffled_preds = model.predict(X_shuffled)

            shuffled_score = metric_func(y, shuffled_preds)
            feature_scores.append(baseline_score - shuffled_score)
        feature_importances[feature] = np.mean(feature_scores)

    return dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

