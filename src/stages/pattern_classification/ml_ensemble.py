"""Pattern identifier ensembles.

Expected data layout:
- labeled_data.csv: columns include `file` (identifier), `verified_pattern` (target label),
    and optional `code_summary` used only for export when predicting.
- embeddings.csv: columns include `file` plus numeric embedding features named `dim_*`.
    All `dim_*` columns are used as model inputs.

How it works:
- Configuration lives in CONFIG; adjust file paths, class filtering, and CV settings there.
- In evaluation mode (`predict_mode=False`), runs CV for Logistic Regression, SVC, and KNN,
    saves per-model probabilities, reports, and plots a confusion matrix for weighted voting.
- In prediction mode (`predict_mode=True`), trains on verified rows and predicts labels for
    unverified rows, saving predictions with scores and summaries.

To change variables:
- Point CONFIG paths to your datasets; ensure `file` keys match between labeled and embeddings.
- Adjust `min_samples_per_class` to include or drop infrequent labels.
- Tune model defaults or VOTE_WEIGHTS if you need different bias toward specific models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from umap import UMAP
from sklearn.decomposition import PCA

# === Configuration ===
CONFIG = {
    "target_column": "label",
    "predict_mode": True,
    "labeled_data_path": "/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/notebooks/result/metadata.csv",
    # "embeddings_path_optional": "/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/data/datasets/summary_embeddings_768.csv",
    "embeddings_path": "/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/notebooks/result/embeddings.csv",
    # "embeddings_path": "/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/data/datasets/embeddings.csv",
    "min_samples_per_class": 5,
    "n_splits": 5,
    "random_state": 42,
    "none_label": "Other",
    "dimensionality_reduction": {
        "enabled": False,
        "method": "umap",
        "n_components_list": [64, 128, 256,512,700],
        "params": {},
    },
}

MODEL_DEFAULTS = {
    "LogReg": {"C": 10, "class_weight": "balanced", "solver": "lbfgs"},
    "SVC": {
        "C": 10,
        "kernel": "rbf",
        "class_weight": "balanced",
        "probability": True,
        "gamma": "scale",
    },
    "KNN": {"n_neighbors": 15, "weights": "distance", "p": 2},
}

MODEL_BUILDERS = {
    "LogReg": lambda params, seed: LogisticRegression(**params, random_state=seed, max_iter=2000),
    "SVC": lambda params, seed: SVC(**params, random_state=seed),#make_pipeline(StandardScaler(), SVC(**params, random_state=seed)),
    "KNN": lambda params, _seed: make_pipeline(StandardScaler(), KNeighborsClassifier(**params)),
}
# Calibrate weights via averaged model F1 scores or researcher priors if one model should dominate
VOTE_WEIGHTS = {"LogReg": 1.06, "SVC": 1.01, "KNN": 0.93}


# === Label utilities ===
def prepare_labels(y):
    label_encoder = LabelEncoder()
    y_values = y.values if isinstance(y, pd.Series) else y
    encoded = label_encoder.fit_transform(y_values)
    return label_encoder, encoded


def get_model_classes(fitted_model):
    if hasattr(fitted_model, "classes_"):
        return fitted_model.classes_
    if hasattr(fitted_model, "steps") and fitted_model.steps:
        return fitted_model.steps[-1][1].classes_
    raise AttributeError("Model does not expose classes_.")


# === Model helpers ===
def build_model(name, params, seed):
    if name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model name: {name}")
    return MODEL_BUILDERS[name](params, seed)


def align_probabilities(prob_matrix, model_classes, global_classes):
    # Map model-specific class ordering to the shared global ordering
    aligned = np.zeros((prob_matrix.shape[0], len(global_classes)))
    for class_id, class_label in enumerate(model_classes):
        global_index = np.where(global_classes == class_label)[0][0]
        aligned[:, global_index] = prob_matrix[:, class_id]
    return aligned


def resolve_model_params(custom_params=None):
    if not custom_params:
        return MODEL_DEFAULTS
    merged = {}
    for name, defaults in MODEL_DEFAULTS.items():
        override = custom_params.get(name, {}) if custom_params else {}
        merged[name] = {**defaults, **override}
    return merged


def resolve_weights(custom_weights=None):
    if custom_weights is None:
        return VOTE_WEIGHTS
    return {**VOTE_WEIGHTS, **custom_weights}


# === Ensemble training ===
def run_all_ensembles(X, y, n_splits, model_params, random_state):
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    label_encoder, y_encoded = prepare_labels(y)
    global_classes = np.unique(y_encoded)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    params = resolve_model_params(model_params)
    prob_store = {name: np.zeros((len(y_encoded), len(global_classes))) for name in MODEL_BUILDERS}

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(X_values, y_encoded), 1):
        print(train_idx)
        X_train, X_val = X_values[train_idx], X_values[val_idx]
        y_train = y_encoded[train_idx]

        for name in MODEL_BUILDERS:
            model = build_model(name, params[name], random_state)
            model.fit(X_train, y_train)

            val_probs = model.predict_proba(X_val)
            model_classes = get_model_classes(model)
            aligned_probs = align_probabilities(val_probs, model_classes, global_classes)
            # Preserve validation fold scores in the correct global class slots
            prob_store[name][val_idx] = aligned_probs

        print(f"Completed fold {fold_index}/{n_splits}.")

    results = {"__meta__": {"label_encoder": label_encoder, "classes": label_encoder.classes_, "f1_macro": {}}}
    for name, probs in prob_store.items():
        probs_df = pd.DataFrame(probs, columns=global_classes)
        pred_indices = probs_df.idxmax(axis=1).values
        y_pred = label_encoder.inverse_transform(pred_indices)
        y_true_decoded = label_encoder.inverse_transform(y_encoded)
        report_dict = classification_report(y_true_decoded, y_pred, output_dict=True, zero_division=0)
        report = classification_report(y_true_decoded, y_pred)
        results[name] = (probs_df, report)
        results.setdefault("__meta__", {}).setdefault("f1_macro", {})[name] = report_dict.get("macro avg", {}).get(
            "f1-score", float("nan")
        )
        print(f"\nClassification Report (OOF - {name}):\n{report}")

    return results


def run_single_ensemble(X, y, model_name, model_params=None, n_splits=None, random_state=None):
    params = {model_name: model_params} if model_params is not None else None
    splits = n_splits or CONFIG["n_splits"]
    seed = CONFIG["random_state"] if random_state is None else random_state
    results = run_all_ensembles(X, y, n_splits=splits, model_params=params, random_state=seed)
    return results[model_name]


# === Voting strategies ===
def majority_vote(prob_results, y_true, label_encoder, none_label):
    y_true_values = y_true.values if isinstance(y_true, pd.Series) else y_true
    bucket_names = ("all_three_agree", "two_agree", "all_disagree")
    bucket_totals = {bucket: 0 for bucket in bucket_names}
    bucket_correct = {bucket: 0 for bucket in bucket_names}

    predictions = []
    for row_index in range(len(y_true_values)):
        votes = []
        for probs_df in prob_results.values():
            encoded_class = int(probs_df.iloc[row_index].astype(float).idxmax())
            votes.append(label_encoder.inverse_transform([encoded_class])[0])

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

        bucket_totals[bucket] += 1
        bucket_correct[bucket] += int(final_label == y_true_values[row_index])
        predictions.append(final_label)

    labels = sorted(set(y_true_values) | {none_label})
    report = classification_report(y_true_values, predictions, labels=labels)

    summary_rows = []
    for bucket in bucket_names:
        total = bucket_totals[bucket]
        correct = bucket_correct[bucket]
        accuracy = correct / total if total else float("nan")
        summary_rows.append({"bucket": bucket, "total": total, "correct": correct, "accuracy": accuracy})
    summary_df = pd.DataFrame(summary_rows)

    return predictions, report, summary_df


def weighted_vote(prob_results, y_true, label_encoder, weights, none_label):
    y_true_values = y_true.values if isinstance(y_true, pd.Series) else y_true
    weight_map = resolve_weights(weights)

    predictions = []
    winning_scores = []
    for row_index in range(len(y_true_values)):
        score_vector = None
        for name, probs_df in prob_results.items():
            weighted = weight_map.get(name, 0) * probs_df.iloc[row_index].astype(float).values
            score_vector = weighted if score_vector is None else score_vector + weighted
        best_index = int(np.argmax(score_vector))
        predictions.append(label_encoder.inverse_transform([best_index])[0])
        winning_scores.append(float(score_vector[best_index]))

    labels = sorted(set(y_true_values) | {none_label})
    report = classification_report(y_true_values, predictions, labels=labels)

    total = len(predictions)
    correct = int(np.sum(np.array(predictions) == np.array(y_true_values)))
    accuracy = correct / total if total else float("nan")
    summary_df = pd.DataFrame(
        [
            {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "score_min": float(np.min(winning_scores)) if winning_scores else float("nan"),
                "score_mean": float(np.mean(winning_scores)) if winning_scores else float("nan"),
                "score_max": float(np.max(winning_scores)) if winning_scores else float("nan"),
            }
        ]
    )

    return predictions, report, summary_df


def weighted_vote_predict(prob_results, label_encoder, weights):
    weight_map = resolve_weights(weights)
    predictions = []
    scores = []

    total_rows = len(next(iter(prob_results.values()))) if prob_results else 0
    for row_index in range(total_rows):
        score_vector = None
        for name, probs_df in prob_results.items():
            weighted = weight_map.get(name, 0) * probs_df.iloc[row_index].astype(float).values
            score_vector = weighted if score_vector is None else score_vector + weighted
        best_index = int(np.argmax(score_vector))
        predictions.append(label_encoder.inverse_transform([best_index])[0])
        scores.append(float(score_vector[best_index]))

    return predictions, scores


# === Data handling ===
def load_and_merge_data(config):
    labeled = pd.read_csv(config["labeled_data_path"])
    embeddings = pd.read_csv(config["embeddings_path"])
    merged_part_01 = pd.merge(labeled, embeddings, on="file")
    merged = merged_part_01
    if config.get("embeddings_path_optional"):
        optional_embeddings = pd.read_csv(config["embeddings_path_optional"])
        merged_part_02 = pd.merge(labeled, optional_embeddings, on="file", how="left")
        merged = pd.concat([merged_part_01, merged_part_02], ignore_index=True)
    return merged


def split_verified_sets(data, target_column, min_samples):
    verified = data[~data[target_column].isna()]
    counts = verified[target_column].value_counts()
    keep_labels = counts[counts >= min_samples].index
    verified_filtered = verified[verified[target_column].isin(keep_labels)]
    unverified = data[data[target_column].isna()]
    return verified_filtered, unverified


def select_embedding_columns(df):
    return [col for col in df.columns if col.startswith("emb_")]


def dimensionality_reduction(X, method, n_components, random_state, **kwargs):
    X_values = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    n_samples, n_features = X_values.shape

    if method == "pca":
        max_components = min(n_samples, n_features) - 1
        if n_components > max_components:
            if max_components < 1:
                raise ValueError(
                    f"Cannot reduce: need at least 2 samples/features (n_samples={n_samples}, n_features={n_features})"
                )
            print(
                f"Clamping PCA n_components from {n_components} to {max_components} (must be <= min(n_samples, n_features) - 1)"
            )
            n_components = max_components

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_values)
        reducer = PCA(n_components=n_components, random_state=random_state)
        X_reduced = reducer.fit_transform(X_scaled)
    elif method == "umap":
        max_components = min(n_samples - 2, n_features)
        if n_components > max_components:
            if max_components < 1:
                raise ValueError(
                    f"Cannot reduce with UMAP: need at least 3 samples and 1 feature (n_samples={n_samples}, n_features={n_features})"
                )
            print(
                f"Clamping UMAP n_components from {n_components} to {max_components} (must be < n_samples - 1 and <= n_features)"
            )
            n_components = max_components

        reducer = UMAP(n_components=n_components, random_state=random_state, **kwargs)
        X_reduced = reducer.fit_transform(X_values)
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")

    columns = [f"dim_{method}_{n_components}_{idx}" for idx in range(X_reduced.shape[1])]
    return pd.DataFrame(X_reduced, columns=columns), reducer


# === Evaluation mode ===
def run_cross_validation(verified_df, feature_columns, target_column, config):
    X_full = verified_df[feature_columns]
    y = verified_df[target_column]

    reduction_cfg = config.get("dimensionality_reduction", {})
    runs = [("full", X_full)]
    if reduction_cfg.get("enabled") and reduction_cfg.get("n_components_list"):
        method = reduction_cfg.get("method", "pca")
        reducer_params = reduction_cfg.get("params", {})
        for n_components in reduction_cfg.get("n_components_list", []):
            try:
                reduced_df, _ = dimensionality_reduction(
                    X_full,
                    method=method,
                    n_components=n_components,
                    random_state=config["random_state"],
                    **reducer_params,
                )
            except ValueError as exc:
                print(f"Skipping {method} with n_components={n_components}: {exc}")
                continue
            runs.append((f"{method}_{n_components}", reduced_df))

    for run_label, X in runs:
        print(f"\n=== Evaluation run: {run_label} ===")
        results = run_all_ensembles(
            X,
            y,
            n_splits=config["n_splits"],
            model_params=None,
            random_state=config["random_state"],
        )

        lr_probs, _ = results["LogReg"]
        svc_probs, _ = results["SVC"]
        knn_probs, _ = results["KNN"]

        suffix = "" if run_label == "full" else f"_{run_label}"
        lr_probs.to_csv(f"probs_lr{suffix}.csv", index=False)
        svc_probs.to_csv(f"probs_svc{suffix}.csv", index=False)
        knn_probs.to_csv(f"probs_knn{suffix}.csv", index=False)

        label_encoder = results["__meta__"]["label_encoder"]
        prob_map = {"LogReg": lr_probs, "SVC": svc_probs, "KNN": knn_probs}

        f1_scores = results["__meta__"].get("f1_macro", {})
        finite_scores = [score for score in f1_scores.values() if pd.notna(score) and np.isfinite(score)]
        dynamic_weights = None
        if finite_scores:
            avg_f1 = np.mean(finite_scores)
            if avg_f1:
                dynamic_weights = {
                    name: (score / avg_f1 if np.isfinite(score) else 0.0) for name, score in f1_scores.items()
                }

        if dynamic_weights:
            print("Dynamic vote weights (model F1 / average F1):")
            for name, weight in dynamic_weights.items():
                print(f"  {name}: {weight:.3f}")
        else:
            print("Dynamic vote weights could not be computed; falling back to defaults.")

        _, majority_report, majority_summary = majority_vote(prob_map, y, label_encoder, none_label=config["none_label"])
        print("\nClassification Report (Majority Vote):")
        print(majority_report)
        print("Vote breakdown (totals, correct, accuracy):")
        print(majority_summary.to_string(index=False))

        weighted_predictions, weighted_report, weighted_summary = weighted_vote(
            prob_map,
            y,
            label_encoder,
            weights=dynamic_weights,
            none_label=config["none_label"],
        )
        print("\nClassification Report (Weighted Vote):")
        print(weighted_report)
        print("Weighted vote summary (totals, correct, accuracy, score stats):")
        print(weighted_summary.to_string(index=False))

        labels = list(label_encoder.classes_)
        if config["none_label"] not in labels:
            labels.append(config["none_label"])
        cm = confusion_matrix(y, weighted_predictions, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        # print("\nConfusion Matrix (Weighted Vote):")
        # print(cm_df.to_string())

        pd.DataFrame(
            {
                "file": verified_df["file"],
                "true_label": y,
                "weighted_pred": weighted_predictions,
            }
        ).to_csv(f"ensemble_classification_predictions{suffix}.csv", index=False)

        plot_confusion_matrix(cm, labels, suffix=suffix)


def plot_confusion_matrix(cm, labels, suffix=""):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Weighted Vote Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=color)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"weighted_confusion{suffix}.png", dpi=150)
    plt.close()


# === Prediction mode ===
def run_prediction_mode(verified_df, unverified_df, feature_columns, target_column, config):
    if unverified_df.empty:
        print("No unverified data to predict.")
        return

    X_train = verified_df[feature_columns]
    y_train = verified_df[target_column]
    X_predict = unverified_df[feature_columns]

    label_encoder, y_encoded = prepare_labels(y_train)
    global_classes = np.unique(y_encoded)

    params = resolve_model_params(None)
    prob_map = {}
    for name in MODEL_BUILDERS:
        model = build_model(name, params[name], config["random_state"])
        model.fit(X_train.values, y_encoded)
        probabilities = model.predict_proba(X_predict.values)
        prob_map[name] = pd.DataFrame(probabilities, columns=global_classes)

    predictions, scores = weighted_vote_predict(prob_map, label_encoder, weights=None)

    pd.DataFrame(
        {
            "file": unverified_df["file"],
            "weighted_pred": predictions,
            "weighted_score": scores,
            "code_summary": unverified_df["code_summary"],
        }
    ).to_csv("unverified_weighted_predictions.csv", index=False)
    print("Saved predictions to unverified_weighted_predictions.csv")


# === Entry point ===
def main(config=CONFIG):
    data = load_and_merge_data(config)
    verified_df, unverified_df = split_verified_sets(
        data,
        target_column=config["target_column"],
        min_samples=config["min_samples_per_class"],
    )
    feature_columns = select_embedding_columns(verified_df)

    if config["predict_mode"]:
        run_prediction_mode(verified_df, unverified_df, feature_columns, config["target_column"], config)
    else:
        run_cross_validation(verified_df, feature_columns, config["target_column"], config)


if __name__ == "__main__":
    main()