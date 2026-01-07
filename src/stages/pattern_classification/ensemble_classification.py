import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow import keras
# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)


data = pd.read_csv('/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/experiments-llm/results/pattern_embeddings/gemini_pattern_embedding_v3.csv')

data = data.drop(columns=['file'])
data['pattern'] = data['pattern'].fillna('None Type')

EVALUATING_ENABLED = False

TARGET_COLUMN = "pattern"

ARTIFACT_DIR = Path("../../../models/pattern_nn_classifier").resolve()
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / "pattern_classifier.keras"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
ENCODER_PATH = ARTIFACT_DIR / "label_encoder.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

def build_classifier(input_dim: int, num_classes: int) -> Sequential:
    """Return a tuned dense network regularized for high-dimensional embeddings."""
    regularizer = tf.keras.regularizers.l2(1e-4)
    return Sequential(
        [
            tf.keras.Input(shape=(input_dim,)),
            Dense(768, activation="relu", kernel_regularizer=regularizer),
            BatchNormalization(),
            Dropout(0.35),
            Dense(512, activation="relu", kernel_regularizer=regularizer),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation="relu", kernel_regularizer=regularizer),
            BatchNormalization(),
            Dropout(0.25),
            Dense(128, activation="relu"),
            Dropout(0.2),
            # Dense(num_classes, activation="softmax"),
            Dense(num_classes, activation=None),
        ]
    )

@keras.utils.register_keras_serializable()
class TemperatureScaling(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
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

def build_calibrated_model(base_model: tf.keras.Model) -> tf.keras.Model:
    base_model.trainable = False

    inputs = tf.keras.Input(shape=base_model.input_shape[1:])
    logits = base_model(inputs)

    scaled_logits = TemperatureScaling()(logits)
    outputs = tf.keras.layers.Softmax()(scaled_logits)

    return tf.keras.Model(inputs, outputs)

import numpy as np

def nn_train(data=data):
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_features:
        raise ValueError("No numeric features detected – please ensure embeddings/features are present.")

    X = data[numeric_features].fillna(0.0).values
    y = data[TARGET_COLUMN].astype(str).values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    X_train, X_temp, y_train_enc, y_temp_enc = train_test_split(
        X,
        y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded,
    )
    X_val, X_test, y_val_enc, y_test_enc = train_test_split(
        X_temp,
        y_temp_enc,
        test_size=0.5,
        random_state=42,
        stratify=y_temp_enc,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train = tf.keras.utils.to_categorical(y_train_enc, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val_enc, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test_enc, num_classes)


    if not EVALUATING_ENABLED:
        # X_train = np.vstack([X_train, X_val])
        # y_train = np.vstack([y_train, y_val])
        X_val = np.vstack([X_val, X_test])
        y_val = np.vstack([y_val, y_test])

    def expected_calibration_error(
        probs: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 15,
    ) -> float:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == y_true).astype(float)

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        N = len(y_true)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_size = np.sum(in_bin)

            if bin_size > 0:
                bin_accuracy = np.mean(accuracies[in_bin])
                bin_confidence = np.mean(confidences[in_bin])

                ece += (bin_size / N) * abs(bin_accuracy - bin_confidence)

        return ece


    model = build_classifier(X_train.shape[1], num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=20, min_delta=1e-4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=8, min_lr=1e-5),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    calibrated_model = build_calibrated_model(model)

    calibrated_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )


    calibrated_model.fit(
        X_val,
        y_val,
        epochs=50,
        batch_size=256,
        verbose=0,
    )


    if not EVALUATING_ENABLED:
        # Persist artifacts for downstream inference pipelines
        calibrated_model.save(MODEL_PATH, include_optimizer=True)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(label_encoder, ENCODER_PATH)
        metadata = {
            "target_column": TARGET_COLUMN,
            "numeric_features": numeric_features,
            "num_classes": num_classes,
            "label_classes": label_encoder.classes_.tolist(),
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2))
        print(f"Saved model to {MODEL_PATH}")
        print(f"Saved scaler to {SCALER_PATH}")
        print(f"Saved label encoder to {ENCODER_PATH}")
        print(f"Saved metadata to {METADATA_PATH}")
    else:
        print("\nEvaluation enabled; not saving model artifacts.")

        test_loss, test_acc, test_top3 = calibrated_model.evaluate(X_test, y_test, verbose=0)
        print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f} | Test top-3 accuracy: {test_top3:.4f}")

        y_pred = calibrated_model.predict(X_test)
        y_pred_labels = y_pred.argmax(axis=1)
        report = classification_report(
            y_test_enc,
            y_pred_labels,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        summary = report_df.loc[["accuracy", "macro avg", "weighted avg"]].round(3)
        class_breakdown = (
            report_df.drop(index=["accuracy", "macro avg", "weighted avg"]).sort_values("support", ascending=False).head(8).round(3)
        )
        probs = calibrated_model.predict(X_test)
        ece = expected_calibration_error(probs, y_test_enc, n_bins=15)

        print(f"ECE: {ece:.4f}")


        print("\nKey metrics:")
        print("\nTop classes by support:")

        raw_preds = model.predict(X_test).argmax(axis=1)
        cal_preds = calibrated_model.predict(X_test).argmax(axis=1)

        print("Accuracy identical:", np.all(raw_preds == cal_preds))
    
    return calibrated_model,scaler,label_encoder

def lr_train(data=data):
    # Logistic regression stage intentionally skipped per latest workflow requirements.
    # The end-to-end classifier now relies solely on the neural network above.
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_features:
        raise ValueError("No numeric features detected – please ensure embeddings/features are present.")

    X = data[numeric_features].fillna(0.0).values
    y = data[TARGET_COLUMN].astype(str).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42,stratify=y_encoded
    )

    from sklearn.linear_model import LogisticRegression

    logreg = LogisticRegression(max_iter=1000,n_jobs=-1)

    if EVALUATING_ENABLED:
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        summary = report_df.loc[["accuracy", "macro avg", "weighted avg"]].round(3)
        class_breakdown = (
            report_df.drop(index=["accuracy", "macro avg", "weighted avg"]).sort_values("support", ascending=False).head(8).round(3)
        )
    else:
        logreg.fit(X, y)
        # Save model
        ARTIFACT_DIR = Path("../models/pattern_logreg_classifier").resolve()
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(logreg, ARTIFACT_DIR / "logistic_regression_model.joblib")
        joblib.dump(label_encoder, ARTIFACT_DIR / "label_encoder.joblib")
        joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
        print(f"Saved logistic regression model and artifacts to {ARTIFACT_DIR}")

    return logreg,scaler,label_encoder

from sklearn.model_selection import StratifiedKFold

raw_data = pd.read_csv('/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/experiments-llm/results/pattern_embeddings/gemini_pattern_embedding_v3.csv')
raw_data['pattern'] = raw_data['pattern'].fillna('None')

synthetic_data       = raw_data[raw_data['file'].str.contains('pattern',na=False)]
verified_communities = raw_data[~raw_data['file'].str.contains('pattern',na=False)]
synthetic_data = synthetic_data.drop(columns=['file'])
verified_communities = verified_communities.drop(columns=['file'])
def get_folded_splits(fold_count=5,random_state=42,use_only_verified=True,min_samples_per_class=5):
    folded_data = []

    _sd = synthetic_data.copy()
    _vd = verified_communities.copy()

    _vd_c = _vd['pattern'].value_counts()
    _vp_i = _vd_c[_vd_c>=min_samples_per_class].index.tolist()
    _vd   = _vd[_vd['pattern'].isin(_vp_i)]

    skf = StratifiedKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=random_state
    )

    folds = list(skf.split(_vd, _vd['pattern']))

    for test_fold in range(fold_count):
        val_fold = (test_fold + 1) % fold_count
        train_folds = [
            i for i in range(fold_count)
            if i not in [test_fold, val_fold]
        ]

        train_idx = np.concatenate([folds[i][1] for i in train_folds])
        val_idx   = folds[val_fold][1]
        test_idx  = folds[test_fold][1]

        vd_train = _vd.iloc[train_idx]
        vd_val   = _vd.iloc[val_idx]
        vd_test  = _vd.iloc[test_idx]

        if use_only_verified:
            train_data = vd_train
        else:
            train_data = pd.concat([vd_train, _sd], ignore_index=True)

        folded_data.append(
            (train_data, vd_val, vd_test)
        )

    return folded_data


def get_lr_probabilities(fold):
    train_data, val_data, test_data = fold
    logreg,scaler,label_encoder = lr_train(train_data)

    numeric_features = train_data.select_dtypes(include=[np.number]).columns.tolist()

    X_train = scaler.transform(train_data[numeric_features].fillna(0.0).values)
    X_test  = scaler.transform(test_data[numeric_features].fillna(0.0).values)
    X_val   = scaler.transform(val_data[numeric_features].fillna(0.0).values)

    lr_train_probab = logreg.predict_proba(X_train)
    lr_test_probab  = logreg.predict_proba(X_test)
    lr_val_probab   = logreg.predict_proba(X_val)

    lr_train_probab = pd.DataFrame(lr_train_probab, columns=label_encoder.classes_)
    lr_test_probab  = pd.DataFrame(lr_test_probab, columns=label_encoder.classes_)
    lr_val_probab   = pd.DataFrame(lr_val_probab, columns=label_encoder.classes_)

    lr_train_probab['pattern'] = train_data['pattern'].values
    lr_test_probab['pattern']  = test_data['pattern'].values
    lr_val_probab['pattern']   = val_data['pattern'].values

    return lr_train_probab,lr_test_probab,lr_val_probab

def get_nn_probabilities(fold):
    train_data, val_data, test_data = fold
    calibrated_model,scaler,label_encoder = nn_train(train_data)

    numeric_features = train_data.select_dtypes(include=[np.number]).columns.tolist()

    X_train = scaler.transform(train_data[numeric_features].fillna(0.0).values)
    X_test  = scaler.transform(test_data[numeric_features].fillna(0.0).values)
    X_val   = scaler.transform(val_data[numeric_features].fillna(0.0).values)

    nn_train_probab = calibrated_model.predict(X_train)
    nn_test_probab  = calibrated_model.predict(X_test)
    nn_val_probab   = calibrated_model.predict(X_val)

    nn_train_probab = pd.DataFrame(nn_train_probab, columns=label_encoder.classes_)
    nn_test_probab  = pd.DataFrame(nn_test_probab, columns=label_encoder.classes_)
    nn_val_probab   = pd.DataFrame(nn_val_probab, columns=label_encoder.classes_)

    nn_train_probab['pattern'] = train_data['pattern'].values
    nn_test_probab['pattern']  = test_data['pattern'].values
    nn_val_probab['pattern']   = val_data['pattern'].values

    return nn_train_probab,nn_test_probab,nn_val_probab


def get_meta_probabilities(lr_data,nn_data):
    lr_train_probab,lr_test_probab,lr_val_probab = lr_data
    nn_train_probab,nn_test_probab,nn_val_probab = nn_data

    numeric_cols = nn_train_probab.select_dtypes(include=['number']).columns.tolist()

    meta_proba_val = pd.DataFrame()
    meta_proba_test = pd.DataFrame()
    meta_proba_train = pd.DataFrame()

    meta_proba_val = (nn_val_probab[numeric_cols] + lr_val_probab[numeric_cols]*0.7)/2
    meta_proba_test = (nn_test_probab[numeric_cols] + lr_test_probab[numeric_cols]*0.7)/2
    meta_proba_train = (nn_train_probab[numeric_cols] + lr_train_probab[numeric_cols]*0.7)/2

    meta_proba_train['pattern'] = lr_train_probab['pattern'].values
    meta_proba_test['pattern']  = lr_test_probab['pattern'].values
    meta_proba_val['pattern']   = lr_val_probab['pattern'].values

    return meta_proba_train,meta_proba_test,meta_proba_val

def get_prediction(row, class_threshold=0.5,none_threshold=0.5):
    high_prob = row.max()
    if high_prob >= class_threshold:
        max_class = row.idxmax()
        return max_class
    elif high_prob <= none_threshold:
        return 'None Type'
    return "Other"


def classify(class_threshold,none_threshold,data):
    y_true = data['pattern'].fillna('None Type')
    y_pred = data.drop('pattern', axis=1).apply(get_prediction, class_threshold=class_threshold,none_threshold=none_threshold, axis=1)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro = report['macro avg']
    report_df = pd.DataFrame(report).T
    return macro['precision'], macro['recall'], macro['f1-score'],report_df

def find_best_threshold(meta_test_probab):
    best = None
    best_class_wise = pd.DataFrame()
    for th_t in range(0,100,10):
        for thn in range(0,100,100):
            th = th_t/100
            thn=thn/100
            precision, recall, f1, report_df = classify(th,thn,meta_test_probab)
            best_class_wise[str(th)+':'+str(thn)] = report_df['f1-score']

            if best is None or f1 > best[0]:
                best = (f1, th, precision, recall)
    best_class_wise['best_threshold'] = best_class_wise.idxmax(axis=1)
    best_threshold = pd.DataFrame()
    best_threshold['Class Threshold'] = best_class_wise['best_threshold'].apply(lambda x: x.split(":")[0])
    best_threshold['None Threshold'] = best_class_wise['best_threshold'].apply(lambda x: x.split(":")[1])
    return best_threshold

def get_prediction_trained(row,best_threshold):
    high_prob = row.max()
    max_class = row.idxmax()
    if float(best_threshold['Class Threshold'][max_class]) <= float(high_prob):
        return max_class
    return "Other"

def apply_threshold(data,best_threshold):
    y_true = data['pattern'].fillna('None Type')
    y_pred = data.drop('pattern', axis=1).apply(get_prediction_trained, best_threshold=best_threshold, axis=1)

    return y_true,y_pred

def get_prediction_no_threshold(row):
    high_prob = row.max()
    max_class = row.idxmax()
    return max_class

def classify_trained_no_threshold(data):
    y_true = data['pattern'].fillna('None Type')
    y_pred = data.drop('pattern', axis=1).apply(get_prediction_no_threshold, axis=1)
    return y_true,y_pred

def get_classification_report(y_true,y_pred):
    ignore_class = "Other"
    labels = [c for c in y_true.unique() if c != ignore_class]
    labels.extend([c for c in y_pred.unique() if c != ignore_class])
    labels = list(set(labels))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    macro = report['macro avg']
    report_df = pd.DataFrame(report).T
    return macro['precision'], macro['recall'], macro['f1-score'],report_df

def max_probabilities(df1,df2):
    max_vals_df = pd.DataFrame()
    for row in df1.iterrows():
        for col in df1.columns:
            max_vals_df.at[row[0], col] = max(row[1][col], df2.at[row[0], col])
    return max_vals_df

def min_probabilities(df1,df2):
    min_vals_df = pd.DataFrame()
    for row in df1.iterrows():
        for col in df1.columns:
            min_vals_df.at[row[0], col] = min(row[1][col], df2.at[row[0], col])
    return min_vals_df

def get_second_level_model_predictions_lr(lr_data,nn_data,meta_data):
    lr_train_probab,lr_val_probab,lr_test_probab = lr_data
    nn_train_probab,nn_val_probab,nn_test_probab = nn_data
    meta_train_probab,meta_val_probab,meta_test_probab = meta_data

    y_train = meta_train_probab['pattern'].values
    y_val   = meta_val_probab['pattern'].values
    y_test  = meta_test_probab['pattern'].values

    numeric_cols = nn_train_probab.select_dtypes(include=['number']).columns.tolist()

    max_train_probab = max_probabilities(nn_train_probab[numeric_cols],lr_train_probab[numeric_cols])
    max_test_probab  = max_probabilities(nn_test_probab[numeric_cols],lr_test_probab[numeric_cols])
    max_val_probab   = max_probabilities(nn_val_probab[numeric_cols],lr_val_probab[numeric_cols])

    max_train_probab = max_train_probab.add_prefix('max_')
    max_test_probab  = max_test_probab.add_prefix('max_')
    max_val_probab   = max_val_probab.add_prefix('max_')

    min_train_probab = min_probabilities(nn_train_probab[numeric_cols],lr_train_probab[numeric_cols])
    min_test_probab  = min_probabilities(nn_test_probab[numeric_cols],lr_test_probab[numeric_cols])
    min_val_probab   = min_probabilities(nn_val_probab[numeric_cols],lr_val_probab[numeric_cols])

    min_train_probab = min_train_probab.add_prefix('min_')
    min_test_probab  = min_test_probab.add_prefix('min_')
    min_val_probab   = min_val_probab.add_prefix('min_')

    lr_train_probab = lr_train_probab[numeric_cols].add_prefix('lr_')
    lr_test_probab  = lr_test_probab[numeric_cols].add_prefix('lr_')
    lr_val_probab   = lr_val_probab[numeric_cols].add_prefix('lr_')

    nn_train_probab = nn_train_probab[numeric_cols].add_prefix('nn_')
    nn_test_probab  = nn_test_probab[numeric_cols].add_prefix('nn_')
    nn_val_probab   = nn_val_probab[numeric_cols].add_prefix('nn_')

    meta_train_probab = meta_train_probab[numeric_cols].add_prefix('meta_')
    meta_test_probab  = meta_test_probab[numeric_cols].add_prefix('meta_')
    meta_val_probab   = meta_val_probab[numeric_cols].add_prefix('meta_')

    temp_train_set = pd.concat([lr_train_probab,nn_train_probab,meta_train_probab,max_train_probab,pd.Series(y_train, name='pattern')],axis=1)
    temp_test_set  = pd.concat([lr_test_probab,nn_test_probab,meta_test_probab,max_test_probab,pd.Series(y_test, name='pattern')],axis=1)
    temp_val_set   = pd.concat([lr_val_probab,nn_val_probab,meta_val_probab,max_val_probab,pd.Series(y_val, name='pattern')],axis=1)

    train_set = pd.concat([temp_train_set,temp_val_set],ignore_index=True)
    test_set  = temp_test_set

    lr_sec_model,scaler,label_encoder = lr_train(train_set)
    X_test  = scaler.transform(test_set.drop('pattern', axis=1).values)
    sec_test_probab  = lr_sec_model.predict(X_test)
    return sec_test_probab

def get_second_level_model_predictions_rr(lr_data,nn_data,meta_data):
    lr_train_probab,lr_val_probab,lr_test_probab = lr_data
    nn_train_probab,nn_val_probab,nn_test_probab = nn_data
    meta_train_probab,meta_val_probab,meta_test_probab = meta_data

    y_train = meta_train_probab['pattern'].values
    y_val   = meta_val_probab['pattern'].values
    y_test  = meta_test_probab['pattern'].values

    numeric_cols = nn_train_probab.select_dtypes(include=['number']).columns.tolist()

    max_train_probab = max_probabilities(nn_train_probab[numeric_cols],lr_train_probab[numeric_cols])
    max_test_probab  = max_probabilities(nn_test_probab[numeric_cols],lr_test_probab[numeric_cols])
    max_val_probab   = max_probabilities(nn_val_probab[numeric_cols],lr_val_probab[numeric_cols])

    max_train_probab = max_train_probab.add_prefix('max_')
    max_test_probab  = max_test_probab.add_prefix('max_')
    max_val_probab   = max_val_probab.add_prefix('max_')

    min_train_probab = min_probabilities(nn_train_probab[numeric_cols],lr_train_probab[numeric_cols])
    min_test_probab  = min_probabilities(nn_test_probab[numeric_cols],lr_test_probab[numeric_cols])
    min_val_probab   = min_probabilities(nn_val_probab[numeric_cols],lr_val_probab[numeric_cols])

    min_train_probab = min_train_probab.add_prefix('min_')
    min_test_probab  = min_test_probab.add_prefix('min_')
    min_val_probab   = min_val_probab.add_prefix('min_')

    lr_train_probab = lr_train_probab[numeric_cols].add_prefix('lr_')
    lr_test_probab  = lr_test_probab[numeric_cols].add_prefix('lr_')
    lr_val_probab   = lr_val_probab[numeric_cols].add_prefix('lr_')

    nn_train_probab = nn_train_probab[numeric_cols].add_prefix('nn_')
    nn_test_probab  = nn_test_probab[numeric_cols].add_prefix('nn_')
    nn_val_probab   = nn_val_probab[numeric_cols].add_prefix('nn_')

    meta_train_probab = meta_train_probab[numeric_cols].add_prefix('meta_')
    meta_test_probab  = meta_test_probab[numeric_cols].add_prefix('meta_')
    meta_val_probab   = meta_val_probab[numeric_cols].add_prefix('meta_')

    temp_train_set = pd.concat([lr_train_probab,nn_train_probab,meta_train_probab,max_train_probab,pd.Series(y_train, name='pattern')],axis=1)
    temp_test_set  = pd.concat([lr_test_probab,nn_test_probab,meta_test_probab,max_test_probab,pd.Series(y_test, name='pattern')],axis=1)
    temp_val_set   = pd.concat([lr_val_probab,nn_val_probab,meta_val_probab,max_val_probab,pd.Series(y_val, name='pattern')],axis=1)

    train_set = pd.concat([temp_train_set,temp_val_set],ignore_index=True)
    test_set  = temp_test_set

    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    from sklearn.linear_model import RidgeClassifier
    rr_sec_model = RidgeClassifier(alpha=1.0)
    rr_sec_model.fit(scaler.fit_transform(train_set.drop('pattern', axis=1).values), label_encoder.fit_transform(train_set['pattern'].values))
    X_test  = scaler.transform(test_set.drop('pattern', axis=1).values)
    sec_test_pred_enc = rr_sec_model.predict(X_test)
    sec_test_pred = label_encoder.inverse_transform(sec_test_pred_enc)
    return sec_test_pred


############################
#  MAIN LOGIC
############################

def main():
    folds = get_folded_splits(fold_count=5,random_state=42,use_only_verified=True,min_samples_per_class=12)
    prediction_df = pd.DataFrame(columns=['pattern','threshold_class','without_threshold_class','second_level_model_class_lr','second_level_model_class_rr'])

    for fold in folds:
        lr_train_probab,lr_val_probab,lr_test_probab = get_lr_probabilities(fold)
        nn_train_probab,nn_val_probab,nn_test_probab = get_nn_probabilities(fold)

        nn_data = (nn_train_probab,nn_val_probab,nn_test_probab)
        lr_data = (lr_train_probab,lr_val_probab,lr_test_probab)
        meta_train_probab,meta_val_probab,meta_test_probab = get_meta_probabilities(lr_data,nn_data)

        meta_data = (meta_train_probab,meta_val_probab,meta_test_probab)
        
        best_threshold = find_best_threshold(meta_val_probab)
        
        threshold_predictions = apply_threshold(meta_test_probab,best_threshold)
        no_threshold_predictions = classify_trained_no_threshold(meta_test_probab)
        second_level_model_predictions_lr = get_second_level_model_predictions_lr(lr_data,nn_data,meta_data)
        second_level_model_predictions_rr = get_second_level_model_predictions_rr(lr_data,nn_data,meta_data)
        
        temp_predictions = pd.DataFrame()
        temp_predictions['pattern'] = meta_test_probab['pattern']
        temp_predictions['threshold_class'] = threshold_predictions[1]
        temp_predictions['without_threshold_class'] = no_threshold_predictions[1]
        temp_predictions['second_level_model_class_lr'] = second_level_model_predictions_lr
        temp_predictions['second_level_model_class_rr'] = second_level_model_predictions_rr
        prediction_df = pd.concat([prediction_df, temp_predictions], ignore_index=True)

    # Log classification reports
    print(f"Classification with thresholding [f1]: {get_classification_report(prediction_df['pattern'], prediction_df['threshold_class'])[2]}")
    print(f"Classification without thresholding [f1]: {get_classification_report(prediction_df['pattern'], prediction_df['without_threshold_class'])[2]}")
    print(f"Classification with second level model lr [f1]: {get_classification_report(prediction_df['pattern'], prediction_df['second_level_model_class_lr'])[2]}")
    print(f"Classification with second level model rr [f1]: {get_classification_report(prediction_df['pattern'], prediction_df['second_level_model_class_rr'])[2]}")

    # Save final predictions
    prediction_df.to_csv('ensemble_classification_predictions.csv', index=False)
    get_classification_report(prediction_df['pattern'], prediction_df['threshold_class'])[3].to_csv('ensemble_classification_report_with_threshold.csv')
    get_classification_report(prediction_df['pattern'], prediction_df['without_threshold_class'])[3].to_csv('ensemble_classification_report_without_threshold.csv')
    get_classification_report(prediction_df['pattern'], prediction_df['second_level_model_class_lr'])[3].to_csv('ensemble_classification_report_second_level_model_lr.csv')
    get_classification_report(prediction_df['pattern'], prediction_df['second_level_model_class_rr'])[3].to_csv('ensemble_classification_report_second_level_model_rr.csv')


if __name__ == "__main__":
    main()