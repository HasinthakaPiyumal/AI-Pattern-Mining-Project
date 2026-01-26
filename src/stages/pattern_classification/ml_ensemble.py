import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def _prepare_labels(y):
    le = LabelEncoder()
    y_vals = y.values if isinstance(y, pd.Series) else y
    y_enc = le.fit_transform(y_vals)
    return le, y_enc


def _get_classes(fitted_model):
    # Works for both bare estimators and pipelines
    if hasattr(fitted_model, "classes_"):
        return fitted_model.classes_
    if hasattr(fitted_model, "steps") and fitted_model.steps:
        return fitted_model.steps[-1][1].classes_
    raise AttributeError("Model does not expose classes_.")


def run_all_ensembles(X, y, n_splits=5, model_params=None, random_state=42):
    """Run all classifiers in a single StratifiedKFold loop to reuse splits."""
    X_vals = X.values if isinstance(X, pd.DataFrame) else X
    le, y_enc = _prepare_labels(y)

    unique_classes = np.unique(y_enc)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    defaults = {
        "LogReg": {'C': 10, 'class_weight': 'balanced', 'solver': 'lbfgs'},
        "SVC": {
            'C': 10,
            'kernel': 'rbf',
            'class_weight': 'balanced',
            'probability': True,
            'gamma': 'scale'
        },
        "KNN": {'n_neighbors': 15, 'weights': 'distance', 'p': 2},
    }

    builders = {
        "LogReg": lambda params, rs: LogisticRegression(**params, random_state=rs, max_iter=2000),
        "SVC": lambda params, rs: make_pipeline(StandardScaler(), SVC(**params, random_state=rs)),
        "KNN": lambda params, _rs: make_pipeline(StandardScaler(), KNeighborsClassifier(**params)),
    }

    oof_probs = {name: np.zeros((len(y_enc), len(unique_classes))) for name in builders}
    param_map = model_params or {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_vals, y_enc), 1):
        X_train, X_val = X_vals[train_idx], X_vals[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]

        for name, build in builders.items():
            params = param_map.get(name) or defaults[name]
            model = build(params, random_state)
            model.fit(X_train, y_train)

            val_probs = model.predict_proba(X_val)
            model_classes = _get_classes(model)
            class_indices = [np.where(unique_classes == c)[0][0] for c in model_classes]
            oof_probs[name][val_idx[:, None], class_indices] = val_probs

            print(f"Fold {fold}/{n_splits} completed for {name}.")

    results = {"__meta__": {"label_encoder": le, "classes": le.classes_}}
    for name, probs in oof_probs.items():
        probs_df = pd.DataFrame(probs, columns=unique_classes)
        y_pred_indices = probs_df.idxmax(axis=1).values
        y_pred = le.inverse_transform(y_pred_indices)
        report = classification_report(le.inverse_transform(y_enc), y_pred)

        print(f"\nClassification Report (OOF - {name}):")
        print(report)

        results[name] = (probs_df, report)

    return results


def run_lr_ensemble(X, y, n_splits=5, model_params=None, random_state=42):
    res = run_all_ensembles(X, y, n_splits=n_splits, model_params={"LogReg": model_params} if model_params is not None else None, random_state=random_state)
    return res["LogReg"]


def run_svc_ensemble(X, y, n_splits=5, model_params=None, random_state=42):
    res = run_all_ensembles(X, y, n_splits=n_splits, model_params={"SVC": model_params} if model_params is not None else None, random_state=random_state)
    return res["SVC"]


def run_knn_ensemble(X, y, n_splits=5, model_params=None, random_state=42):
    res = run_all_ensembles(X, y, n_splits=n_splits, model_params={"KNN": model_params} if model_params is not None else None, random_state=random_state)
    return res["KNN"]


def majority_vote(prob_results, y_true, label_encoder, none_label="none"):
    """Majority vote across models; if all disagree, assign none_label.

    Returns predictions, report, and a DataFrame with per-bucket totals and accuracy.
    """
    y_true_vals = y_true.values if isinstance(y_true, pd.Series) else y_true
    n = len(y_true_vals)

    preds = []
    buckets = ("all_three_agree", "two_agree", "all_disagree")
    vote_tally = {b: 0 for b in buckets}
    correct_tally = {b: 0 for b in buckets}

    for i in range(n):
        votes = []
        for probs_df in prob_results.values():
            enc_class = int(probs_df.iloc[i].astype(float).idxmax())
            votes.append(label_encoder.inverse_transform([enc_class])[0])

        counts = pd.Series(votes).value_counts()
        top_label = counts.idxmax()
        top_count = counts.iloc[0]

        if top_count == 3:
            bucket = "all_three_agree"
            final_label = top_label
        elif top_count == 2:
            bucket = "two_agree"
            final_label = top_label
        else:
            bucket = "all_disagree"
            final_label = none_label

        vote_tally[bucket] += 1
        correct_tally[bucket] += int(final_label == y_true_vals[i])
        preds.append(final_label)

    labels = sorted(set(y_true_vals) | {none_label})
    report = classification_report(y_true_vals, preds, labels=labels)

    rows = []
    for b in buckets:
        total = vote_tally[b]
        correct = correct_tally[b]
        acc = correct / total if total else float("nan")
        rows.append({"bucket": b, "total": total, "correct": correct, "accuracy": acc})
    summary_df = pd.DataFrame(rows)

    return preds, report, summary_df


def weighted_vote(prob_results, y_true, label_encoder, weights=None, none_label="none"):
    """Weighted vote using class probabilities and per-model weights.

    weights: dict mapping model name -> weight. Defaults to LogReg=4, SVC=3, KNN=3.
    Returns predictions, report, and summary stats of winning scores and accuracy.
    """
    y_true_vals = y_true.values if isinstance(y_true, pd.Series) else y_true
    n = len(y_true_vals)

    default_weights = {"LogReg": 1.06, "SVC": 1.01, "KNN": 0.93}
    w = default_weights if weights is None else {**default_weights, **weights}

    # Map names to DataFrames in a consistent order
    prob_results = {k: prob_results[k] for k in prob_results.keys()}

    preds = []
    winning_scores = []

    for i in range(n):
        # accumulate weighted scores per encoded class id
        score_vec = None
        for name, probs_df in prob_results.items():
            probs_row = probs_df.iloc[i].astype(float).values
            weighted = w.get(name, 0) * probs_row
            score_vec = weighted if score_vec is None else score_vec + weighted
        best_idx = int(np.argmax(score_vec))
        best_score = float(score_vec[best_idx])
        preds.append(label_encoder.inverse_transform([best_idx])[0])
        winning_scores.append(best_score)

    labels = sorted(set(y_true_vals) | {none_label})
    report = classification_report(y_true_vals, preds, labels=labels)

    total = len(preds)
    correct = int(np.sum(np.array(preds) == np.array(y_true_vals)))
    accuracy = correct / total if total else float("nan")
    score_min = float(np.min(winning_scores)) if winning_scores else float("nan")
    score_max = float(np.max(winning_scores)) if winning_scores else float("nan")
    score_mean = float(np.mean(winning_scores)) if winning_scores else float("nan")

    summary_df = pd.DataFrame([
        {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "score_min": score_min,
            "score_mean": score_mean,
            "score_max": score_max,
        }
    ])

    return preds, report, summary_df


def weighted_vote_predict(prob_results, label_encoder, weights=None):
    """Weighted vote for inference; returns predicted labels and winning scores."""
    default_weights = {"LogReg": 1.06, "SVC": 1.01, "KNN": 0.93}
    w = default_weights if weights is None else {**default_weights, **weights}

    # Ensure deterministic ordering
    prob_results = {k: prob_results[k] for k in prob_results.keys()}

    preds = []
    scores = []

    n = len(next(iter(prob_results.values()))) if prob_results else 0
    for i in range(n):
        score_vec = None
        for name, probs_df in prob_results.items():
            probs_row = probs_df.iloc[i].astype(float).values
            weighted = w.get(name, 0) * probs_row
            score_vec = weighted if score_vec is None else score_vec + weighted
        best_idx = int(np.argmax(score_vec))
        best_score = float(score_vec[best_idx])
        preds.append(label_encoder.inverse_transform([best_idx])[0])
        scores.append(best_score)

    return preds, scores

if __name__ == "__main__":
    TARGET_COLUMN = "verified_pattern"
    PREDICT_MODE = False  # Toggle: True = train on verified and predict unverified; False = CV eval

    labeled_data = pd.read_csv('/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/data/datasets/labeled_data.csv')
    embeddings   = pd.read_csv('/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/data/datasets/embeddings.csv')
    data = pd.merge(labeled_data, embeddings, on='file')

    synthetic_data       = pd.DataFrame()
    verified_communities = data[~data[TARGET_COLUMN].isna()]
    unverified_communities = data[data[TARGET_COLUMN].isna()]

    counts = verified_communities['verified_pattern'].value_counts()
    to_keep = counts[counts >= 20].index
    verified_communities = verified_communities[verified_communities['verified_pattern'].isin(to_keep)]
    # Trim 40 samples from the 'none' class to reduce imbalance
    # none_label = 'none'
    # if none_label in verified_communities[TARGET_COLUMN].values:
    #     none_indices = verified_communities[verified_communities[TARGET_COLUMN] == none_label].index
    #     trim_n = min(40, len(none_indices))
    #     drop_indices = np.random.default_rng(42).choice(none_indices, size=trim_n, replace=False)
    #     verified_communities = verified_communities.drop(index=drop_indices)


    columns = [col for col in verified_communities.columns if col.startswith('dim_')]

    if not PREDICT_MODE:
        X = verified_communities[columns]
        y = verified_communities[TARGET_COLUMN]
        
        # Run all ensembles in one CV loop
        results = run_all_ensembles(X, y)

        lr_probs_df, lr_report = results["LogReg"]
        lr_probs_df.to_csv("probs_lr.csv", index=False)

        svc_probs_df, svc_report = results["SVC"]
        svc_probs_df.to_csv("probs_svc.csv", index=False)

        knn_probs_df, knn_report = results["KNN"]
        knn_probs_df.to_csv("probs_knn.csv", index=False)

        # Majority vote across models; if all disagree, mark as none
        le = results["__meta__"]["label_encoder"]
        prob_map = {"LogReg": lr_probs_df, "SVC": svc_probs_df, "KNN": knn_probs_df}
        _, vote_report, vote_summary = majority_vote(prob_map, y, le, none_label="none")
        print("\nClassification Report (Majority Vote):")
        print(vote_report)
        print("Vote breakdown (totals, correct, accuracy):")
        print(vote_summary.to_string(index=False))

        # Weighted vote using per-model weights
        weighted_preds, weighted_report, weighted_summary = weighted_vote(prob_map, y, le, none_label="none")
        print("\nClassification Report (Weighted Vote):")
        print(weighted_report)
        print("Weighted vote summary (totals, correct, accuracy, score stats):")
        print(weighted_summary.to_string(index=False))

        # Confusion matrix for weighted voting
        labels = list(le.classes_)
        if "none" not in labels:
            labels.append("none")
        cm = confusion_matrix(y, weighted_preds, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        print("\nConfusion Matrix (Weighted Vote):")
        print(cm_df.to_string())

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Weighted Vote Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=90)
        plt.yticks(tick_marks, labels)
        thresh = cm.max() / 2 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig("weighted_confusion.png", dpi=150)
        plt.close()
    else:
        if unverified_communities.empty:
            print("No unverified data to predict.")
        else:
            X_train = verified_communities[columns]
            y_train = verified_communities[TARGET_COLUMN]
            X_pred = unverified_communities[columns]
            summaries = unverified_communities['code_summary']

            le, y_enc = _prepare_labels(y_train)
            unique_classes = np.unique(y_enc)

            defaults = {
                "LogReg": {'C': 10, 'class_weight': 'balanced', 'solver': 'lbfgs'},
                "SVC": {
                    'C': 10,
                    'kernel': 'rbf',
                    'class_weight': 'balanced',
                    'probability': True,
                    'gamma': 'scale'
                },
                "KNN": {'n_neighbors': 15, 'weights': 'distance', 'p': 2},
            }

            builders = {
                "LogReg": lambda params, rs: LogisticRegression(**params, random_state=rs, max_iter=2000),
                "SVC": lambda params, rs: make_pipeline(StandardScaler(), SVC(**params, random_state=rs)),
                "KNN": lambda params, _rs: make_pipeline(StandardScaler(), KNeighborsClassifier(**params)),
            }

            prob_map = {}
            X_train_vals = X_train.values
            X_pred_vals = X_pred.values
            for name, build in builders.items():
                params = defaults[name]
                model = build(params, 42)
                model.fit(X_train_vals, y_enc)
                probs = model.predict_proba(X_pred_vals)
                prob_map[name] = pd.DataFrame(probs, columns=unique_classes)

            preds, scores = weighted_vote_predict(prob_map, le)

            output = pd.DataFrame({
                "file": unverified_communities["file"],
                "weighted_pred": preds,
                "weighted_score": scores,
                "code_summary": summaries
            })
            output.to_csv("unverified_weighted_predictions.csv", index=False)
            print("Saved predictions to unverified_weighted_predictions.csv")