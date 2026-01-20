import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

ARTIFACT_DIR = Path("/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/models").resolve()


# Custom layer required for loading the NN model
@keras.utils.register_keras_serializable()
class TemperatureScaling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temperature = tf.Variable(
            initial_value=1.0,
            trainable=True,
            dtype=tf.float32,
            constraint=lambda t: tf.clip_by_value(t, 1e-6, 100.0),
        )

    def call(self, logits):
        return logits / self.temperature

    def get_config(self):
        return super().get_config()

# Load LR model and artifacts
lr_model = joblib.load(ARTIFACT_DIR / "lr" / "logistic_regression_model.joblib")
lr_scaler = joblib.load(ARTIFACT_DIR / "lr" / "scaler.joblib")
lr_label_encoder = joblib.load(ARTIFACT_DIR / "lr" / "label_encoder.joblib")

# Load NN model and artifacts
nn_model = tf.keras.models.load_model(ARTIFACT_DIR / "nn" / "pattern_classifier.keras")
nn_scaler = joblib.load(ARTIFACT_DIR / "nn" / "scaler.joblib")
nn_label_encoder = joblib.load(ARTIFACT_DIR / "nn" / "label_encoder.joblib")
with open(ARTIFACT_DIR / "nn" / "metadata.json") as f:
    nn_metadata = json.load(f)

# Load lr-sec model and artifacts
lr_sec_model = joblib.load(ARTIFACT_DIR / "lr-sec" / "logistic_regression_second_level_model.joblib")
lr_sec_scaler = joblib.load(ARTIFACT_DIR / "lr-sec" / "logistic_regression_second_level_scaler.joblib")
lr_sec_label_encoder = joblib.load(ARTIFACT_DIR / "lr-sec" / "logistic_regression_second_level_label_encoder.joblib")


def max_probabilities(df1, df2):
    return df1.combine(df2, np.maximum)


def _compute_top5_avg(proba_df, numeric_cols):
    def row_top5_avg(row):
        sorted_probs = np.sort(row.values)[::-1]
        return np.mean(sorted_probs[:5])
    return proba_df[numeric_cols].apply(row_top5_avg, axis=1).mean()


def predict(embeddings: pd.DataFrame) -> np.ndarray:
    """
    Predict pattern classes for given embeddings.
    
    Pipeline: embeddings -> LR -> NN -> build dataset -> lr-sec
    """
    numeric_features = embeddings.select_dtypes(include=[np.number]).columns.tolist()
    X = embeddings[numeric_features].fillna(0.0).values

    # LR predictions
    X_lr = lr_scaler.transform(X)
    lr_proba = lr_model.predict_proba(X_lr)
    lr_proba_df = pd.DataFrame(lr_proba, columns=lr_label_encoder.classes_)

    # NN predictions
    X_nn = nn_scaler.transform(X)
    nn_proba = nn_model.predict(X_nn)
    nn_proba_df = pd.DataFrame(nn_proba, columns=nn_label_encoder.classes_)

    numeric_cols = nn_proba_df.columns.tolist()

    # Compute correction factor
    lr_correction = _compute_top5_avg(lr_proba_df, numeric_cols)
    nn_correction = _compute_top5_avg(nn_proba_df, numeric_cols)
    if nn_correction == 0:
        nn_correction = 1e-10
    correction_factor = lr_correction / nn_correction

    # Corrected NN and meta probabilities
    nn_proba_corrected = nn_proba_df[numeric_cols] * correction_factor
    meta_proba = (lr_proba_df[numeric_cols] + nn_proba_corrected) / 2

    # Build lr-sec input dataset
    max_proba = max_probabilities(nn_proba_df[numeric_cols], lr_proba_df[numeric_cols])
    max_proba = max_proba.add_prefix('max_')
    lr_prefixed = lr_proba_df[numeric_cols].add_prefix('lr_')
    nn_prefixed = nn_proba_df[numeric_cols].add_prefix('nn_')
    meta_prefixed = meta_proba.add_prefix('meta_')

    sec_input = pd.concat([lr_prefixed, nn_prefixed, meta_prefixed, max_proba], axis=1)

    # lr-sec prediction
    X_sec = lr_sec_scaler.transform(sec_input.values)
    sec_pred_enc = lr_sec_model.predict(X_sec)
    sec_pred = lr_sec_model.predict_proba(X_sec)
    predictions = lr_sec_label_encoder.inverse_transform(sec_pred_enc)

    predictions_df = pd.DataFrame(predictions, columns=['predicted_pattern'])
    predictions_df['file'] = embeddings['file']
    predictions_df = pd.concat([predictions_df,pd.DataFrame(sec_pred,columns=lr_sec_label_encoder.classes_)],axis=1)
    predictions_df.to_csv("/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/data/datasets/predictions.csv", index=False)

    return predictions


if __name__ == "__main__":
    embeddings = pd.read_csv("/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/data/datasets/embeddings.csv")
    predict(embeddings)
