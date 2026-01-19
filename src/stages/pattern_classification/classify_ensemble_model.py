from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

try:
	import tensorflow as tf
	from tensorflow import keras
except Exception as exc:  # pragma: no cover
	raise RuntimeError(
		"TensorFlow is required to run the NN classifier. "
		"Install the project requirements and ensure a working TF build."
	) from exc


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = (THIS_DIR / "../../../models/pattern_ensemble_classifier").resolve()
DEFAULT_THRESHOLD_CSV = (THIS_DIR / "overall_best_threshold.csv").resolve()


@keras.utils.register_keras_serializable()
class TemperatureScaling(tf.keras.layers.Layer):
	"""Matches the training-time layer so Keras can deserialize the model."""

	def __init__(self, **kwargs: Any):
		super().__init__(**kwargs)
		self.temperature = tf.Variable(
			initial_value=1.0,
			trainable=True,
			dtype=tf.float32,
			constraint=lambda t: tf.clip_by_value(t, 1e-6, 100.0),
		)

	def call(self, logits: tf.Tensor) -> tf.Tensor:
		return logits / self.temperature

	def get_config(self) -> dict[str, Any]:
		return super().get_config()


def _load_json(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {}
	return json.loads(path.read_text())


def _numeric_feature_names(df: pd.DataFrame, metadata: dict[str, Any]) -> list[str]:
	meta_cols = metadata.get("numeric_features")
	if isinstance(meta_cols, list) and meta_cols:
		return [c for c in meta_cols if c in df.columns]
	return df.select_dtypes(include=[np.number]).columns.tolist()


def _ensure_2d_float_array(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
	missing = [c for c in feature_cols if c not in df.columns]
	if missing:
		raise ValueError(f"Missing expected numeric feature columns: {missing[:10]}" + (" ..." if len(missing) > 10 else ""))
	return df[feature_cols].fillna(0.0).astype(np.float32).values


def _predict_topk(proba: np.ndarray, class_names: list[str], k: int = 3) -> tuple[list[str], list[float]]:
	k = max(1, min(k, proba.shape[1]))
	top_idx = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
	# Sort top-k indices by probability descending per row
	row_order = np.take_along_axis(proba, top_idx, axis=1)
	sort_idx = np.argsort(-row_order, axis=1)
	top_idx = np.take_along_axis(top_idx, sort_idx, axis=1)
	top_scores = np.take_along_axis(proba, top_idx, axis=1)

	top_labels = [[class_names[j] for j in row] for row in top_idx]
	top_scores_list = [row.tolist() for row in top_scores]
	# Return first choice lists flattened for convenience in simple outputs
	return [row[0] for row in top_labels], [row[0] for row in top_scores_list]


def _load_thresholds_csv(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Threshold CSV not found at: {path}")
	# File has an unnamed index column with class names.
	threshold_df = pd.read_csv(path, index_col=0)
	for col in ["Class Threshold", "None Threshold"]:
		if col not in threshold_df.columns:
			raise ValueError(f"Threshold CSV missing required column: {col}")
	return threshold_df


def _predict_without_threshold(meta_proba_row: np.ndarray, class_names: list[str]) -> str:
	return class_names[int(np.argmax(meta_proba_row))]


def _predict_with_threshold(
	meta_proba_row: np.ndarray,
	class_names: list[str],
	thresholds: pd.DataFrame,
	default_class_threshold: float = 0.3,
	default_none_threshold: float = 0.0,
) -> str:
	max_idx = int(np.argmax(meta_proba_row))
	max_class = class_names[max_idx]
	high_prob = float(meta_proba_row[max_idx])

	if max_class in thresholds.index:
		class_threshold = float(thresholds.at[max_class, "Class Threshold"])
		none_threshold = float(thresholds.at[max_class, "None Threshold"])
	else:
		class_threshold = default_class_threshold
		none_threshold = default_none_threshold

	if high_prob >= class_threshold:
		return max_class
	if high_prob <= none_threshold:
		return "None Type"
	return "Other"


def _align_proba_to_classes(
	proba: np.ndarray,
	proba_classes: list[str],
	target_classes: list[str],
) -> np.ndarray:
	"""Return probability matrix aligned to target_classes order."""
	class_to_idx = {c: i for i, c in enumerate(proba_classes)}
	aligned = np.zeros((proba.shape[0], len(target_classes)), dtype=np.float32)
	for j, c in enumerate(target_classes):
		idx = class_to_idx.get(c)
		if idx is not None:
			aligned[:, j] = proba[:, idx].astype(np.float32)
	return aligned


def _build_second_level_features(
	nn_proba_aligned: np.ndarray,
	lr_proba_aligned: np.ndarray,
	meta_proba: np.ndarray,
	class_names: list[str],
) -> np.ndarray:
	"""Build the same feature matrix used to train lr-sec/rr-sec.

	Training concatenation order (per class):
	- lr_<class>, nn_<class>, meta_<class>, max_<class>
	"""
	max_proba = np.maximum(nn_proba_aligned, lr_proba_aligned)
	parts = []
	for mat in [lr_proba_aligned, nn_proba_aligned, meta_proba, max_proba]:
		parts.append(mat.astype(np.float32))
	# Column order is implicitly [lr classes..., nn classes..., meta classes..., max classes...]
	# matching how DataFrames were created in training.
	return np.concatenate(parts, axis=1)


def run_inference(
	data_path: Path,
	artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
	output_path: Optional[Path] = None,
	threshold_csv: Optional[Path] = None,
	top_k: int = 3,
) -> pd.DataFrame:
	df = pd.read_csv(data_path)
	df = df[~df['file'].str.contains('pattern',na=False)]
	if "pattern" in df.columns:
		df["pattern"] = df["pattern"].fillna("None Type")

	nn_dir = artifact_dir / "nn"
	nn_model_path = nn_dir / "pattern_classifier.keras"
	nn_scaler_path = nn_dir / "scaler.joblib"
	nn_encoder_path = nn_dir / "label_encoder.joblib"
	nn_metadata_path = nn_dir / "metadata.json"

	if not nn_model_path.exists():
		raise FileNotFoundError(f"NN model not found at: {nn_model_path}")
	if not nn_scaler_path.exists():
		raise FileNotFoundError(f"NN scaler not found at: {nn_scaler_path}")
	if not nn_encoder_path.exists():
		raise FileNotFoundError(f"NN label encoder not found at: {nn_encoder_path}")

	nn_metadata = _load_json(nn_metadata_path)
	feature_cols = _numeric_feature_names(df, nn_metadata)
	X = _ensure_2d_float_array(df, feature_cols)

	nn_scaler = joblib.load(nn_scaler_path)
	nn_encoder = joblib.load(nn_encoder_path)
	nn_model = tf.keras.models.load_model(nn_model_path)

	X_nn = nn_scaler.transform(X)
	nn_proba = nn_model.predict(X_nn, verbose=0)
	nn_classes = list(getattr(nn_encoder, "classes_", nn_metadata.get("label_classes", [])))
	if not nn_classes:
		raise RuntimeError("Could not determine NN class labels (missing encoder classes_ and metadata label_classes).")

	nn_pred, nn_conf = _predict_topk(nn_proba, nn_classes, k=top_k)

	out = pd.DataFrame(index=df.index)
	if "file" in df.columns:
		out["file"] = df["file"].astype(str)
	if "pattern" in df.columns:
		out["true_pattern"] = df["pattern"].astype(str)

	# out["nn_pred"] = nn_pred
	# out["nn_conf"] = nn_conf

	# Optional: also load LR artifacts if present and compute meta proba like training:
	# meta = (nn + lr*0.7)/2
	lr_dir = artifact_dir / "lr"
	lr_model_path = lr_dir / "logistic_regression_model.joblib"
	lr_scaler_path = lr_dir / "scaler.joblib"
	lr_encoder_path = lr_dir / "label_encoder.joblib"

	thresholds = None
	if threshold_csv is not None:
		thresholds = _load_thresholds_csv(threshold_csv)

	if lr_model_path.exists() and lr_scaler_path.exists() and lr_encoder_path.exists():
		lr_model = joblib.load(lr_model_path)
		lr_scaler = joblib.load(lr_scaler_path)
		lr_encoder = joblib.load(lr_encoder_path)

		X_lr = lr_scaler.transform(X)
		lr_proba = lr_model.predict_proba(X_lr)
		lr_classes = list(getattr(lr_encoder, "classes_", []))
		if not lr_classes:
			raise RuntimeError("LR label encoder has no classes_.")

		lr_pred, lr_conf = _predict_topk(lr_proba, lr_classes, k=top_k)
		# out["lr_pred"] = lr_pred
		# out["lr_conf"] = lr_conf

		# Align to NN class order (matches second-level training)
		nn_aligned = _align_proba_to_classes(nn_proba, nn_classes, nn_classes)
		lr_aligned = _align_proba_to_classes(lr_proba, lr_classes, nn_classes)

		meta_proba = (nn_aligned + lr_aligned * 0.7) / 2.0
		meta_pred, meta_conf = _predict_topk(meta_proba, nn_classes, k=top_k)
		# out["meta_pred"] = meta_pred
		# out["meta_conf"] = meta_conf

		# Threshold method + without-threshold method (both use meta distribution)
		out["without_threshold_class"] = [
			_predict_without_threshold(meta_proba[i], nn_classes) for i in range(meta_proba.shape[0])
		]
		if thresholds is not None:
			out["threshold_class"] = [
				_predict_with_threshold(meta_proba[i], nn_classes, thresholds) for i in range(meta_proba.shape[0])
			]
		else:
			out["threshold_class"] = None

		# Second-level models (lr-sec, rr-sec) trained on engineered features
		sec_X = _build_second_level_features(nn_aligned, lr_aligned, meta_proba, nn_classes)

		# LR second-level
		lr_sec_dir = artifact_dir / "lr-sec"
		lr_sec_model_path = lr_sec_dir / "logistic_regression_second_level_model.joblib"
		lr_sec_scaler_path = lr_sec_dir / "logistic_regression_second_level_scaler.joblib"
		lr_sec_encoder_path = lr_sec_dir / "logistic_regression_second_level_label_encoder.joblib"
		if lr_sec_model_path.exists() and lr_sec_scaler_path.exists() and lr_sec_encoder_path.exists():
			lr_sec_model = joblib.load(lr_sec_model_path)
			lr_sec_scaler = joblib.load(lr_sec_scaler_path)
			lr_sec_encoder = joblib.load(lr_sec_encoder_path)
			lr_sec_pred_idx = lr_sec_model.predict(lr_sec_scaler.transform(sec_X))
			out["second_level_model_class_lr"] = lr_sec_encoder.inverse_transform(lr_sec_pred_idx)
		else:
			out["second_level_model_class_lr"] = None

		# Ridge second-level
		rr_sec_dir = artifact_dir / "rr-sec"
		rr_sec_model_path = rr_sec_dir / "ridge_regression_second_level_model.joblib"
		rr_sec_scaler_path = rr_sec_dir / "ridge_regression_second_level_scaler.joblib"
		rr_sec_encoder_path = rr_sec_dir / "ridge_regression_second_level_label_encoder.joblib"
		if rr_sec_model_path.exists() and rr_sec_scaler_path.exists() and rr_sec_encoder_path.exists():
			rr_sec_model = joblib.load(rr_sec_model_path)
			rr_sec_scaler = joblib.load(rr_sec_scaler_path)
			rr_sec_encoder = joblib.load(rr_sec_encoder_path)
			rr_sec_pred_idx = rr_sec_model.predict(rr_sec_scaler.transform(sec_X))
			out["second_level_model_class_rr"] = rr_sec_encoder.inverse_transform(rr_sec_pred_idx)
		else:
			out["second_level_model_class_rr"] = None
	else:
		# out["lr_pred"] = None
		# out["lr_conf"] = np.nan
		# out["meta_pred"] = None
		# out["meta_conf"] = np.nan
		out["threshold_class"] = None
		out["without_threshold_class"] = None
		out["second_level_model_class_lr"] = None
		out["second_level_model_class_rr"] = None

	if output_path is not None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		out.to_csv(output_path, index=False)

	return out


def _print_accuracy_summary(preds: pd.DataFrame) -> None:
	if "true_pattern" not in preds.columns:
		print("No true labels found (missing column: true_pattern); skipping accuracy summary.")
		return

	y_true = preds["true_pattern"].astype(str)
	if y_true.isna().all() or len(y_true) == 0:
		print("true_pattern is empty; skipping accuracy summary.")
		return

	columns_to_score = [
		"threshold_class",
		"without_threshold_class",
		"second_level_model_class_lr",
		"second_level_model_class_rr",
		# "meta_pred",
		# "nn_pred",
		# "lr_pred",
	]

	print("\nAccuracy summary (exact match):")
	for col in columns_to_score:
		if col not in preds.columns:
			continue
		y_pred = preds[col]
		if y_pred.isna().all():
			continue
		mask = ~y_true.isna() & ~y_pred.isna()
		if mask.sum() == 0:
			continue
		acc = (y_true[mask].astype(str).values == y_pred[mask].astype(str).values).mean()
		print(f"- {col}: {acc:.4f}  (n={int(mask.sum())})")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Load trained pattern ensemble artifacts and run inference on an embeddings CSV."
	)
	parser.add_argument(
		"--data",
		type=Path,
		default=Path(
			"/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/experiments-llm/results/pattern_embeddings/gemini_communities_embedding.csv"
		),
		help="Path to the embeddings CSV.",
	)
	parser.add_argument(
		"--artifacts",
		type=Path,
		default=DEFAULT_ARTIFACT_DIR,
		help="Path to models/pattern_ensemble_classifier.",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=Path("./predictions_ensemble.csv"),
		help="Where to write predictions CSV.",
	)
	parser.add_argument(
		"--thresholds",
		type=Path,
		default=DEFAULT_THRESHOLD_CSV,
		help="Path to overall_best_threshold.csv (per-class thresholds).",
	)
	parser.add_argument(
		"--top-k",
		type=int,
		default=3,
		help="Top-k used for convenience outputs (default: 3).",
	)
	args = parser.parse_args()

	preds = run_inference(
		data_path=args.data,
		artifact_dir=args.artifacts,
		output_path=args.out,
		threshold_csv=args.thresholds,
		top_k=args.top_k,
	)
	print(f"Wrote predictions to: {args.out}")
	_print_accuracy_summary(preds)


if __name__ == "__main__":
	main()