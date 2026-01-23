import random
import json,os,time
from pathlib import Path

import joblib,random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow import keras

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Setting an item of incompatible dtype"
)

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)

np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()


EVALUATING_ENABLED = True
TARGET_COLUMN = "verified_pattern"

labeled_data = pd.read_csv('/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/data/datasets/labeled_data.csv')
embeddings   = pd.read_csv('/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/data/datasets/embeddings.csv')
data = pd.merge(labeled_data, embeddings, on='file')

synthetic_data       = pd.DataFrame()
verified_communities = data[~data[TARGET_COLUMN].isna()]
# verified_communities = verified_communities[verified_communities['verified_pattern']!='none']
# removable_50 = verified_communities[verified_communities['verified_pattern']=='none']#.sample(n=50, random_state=42).index
# verified_communities = verified_communities.drop(removable_50)

run_time = time.time()

ARTIFACT_DIR = Path("/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/models").resolve()
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
Path(ARTIFACT_DIR / "nn").resolve().mkdir(parents=True, exist_ok=True)
Path(ARTIFACT_DIR / "lr").resolve().mkdir(parents=True, exist_ok=True)
Path(ARTIFACT_DIR / "lr-sec").resolve().mkdir(parents=True, exist_ok=True)
Path(ARTIFACT_DIR / "rr-sec").resolve().mkdir(parents=True, exist_ok=True)

def build_classifier(input_dim: int, num_classes: int,l2_alpha) -> Sequential:
    """Return a tuned dense network regularized for high-dimensional embeddings."""
    regularizer = tf.keras.regularizers.l2(l2_alpha)
    return Sequential(
        [
            tf.keras.Input(shape=(input_dim,)),
            # Dropout(0.3),
            # Dense(2048, activation="relu", kernel_regularizer=regularizer),
            # BatchNormalization(),
            # Dropout(0.3),
            Dense(1024, activation="relu", kernel_regularizer=regularizer),
            # BatchNormalization(),
            # Dropout(0.3),
            # Dense(768, activation="relu", kernel_regularizer=regularizer),


            # Dense(768, activation="relu", kernel_regularizer=regularizer),
            # BatchNormalization(),
            # Dropout(0.35),
            # Dense(512, activation="relu", kernel_regularizer=regularizer),
            # BatchNormalization(),
            # Dropout(0.3),
            # Dense(256, activation="relu", kernel_regularizer=regularizer),
            # BatchNormalization(),
            # Dropout(0.25),
            # Dense(128, activation="relu"),
            # Dropout(0.2),
            Dense(num_classes, activation="softmax"),
            # Dense(num_classes, activation=None),
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

    X_val = np.vstack([X_val, X_test])
    y_val = np.vstack([y_val, y_test])

    # l2_alpha = 0.001
    # bath_size = 32
    # epoch = 30
    # learning_rate = 0.001

    l2_alpha = 1e-4
    batch_size = 64
    epoch = 200
    learning_rate = 1e-3

    model = build_classifier(X_train.shape[1], num_classes,l2_alpha=l2_alpha)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=20, min_delta=1e-4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=8, min_lr=1e-5),
    ]

    classes = np.unique(y_train_enc)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_enc
    )


    class_weights = dict(zip(classes, weights))

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epoch,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
        class_weight=class_weights
    )

    # calibrated_model = build_calibrated_model(model)

    # calibrated_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    #     loss=tf.keras.losses.CategoricalCrossentropy(),
    #     metrics=[
    #         "accuracy",
    #         tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
    #     ],
    # )


    # calibrated_model.fit(
    #     X_val,
    #     y_val,
    #     epochs=50,
    #     batch_size=256,
    #     shuffle=False,
    #     verbose=0,
    #     class_weight = class_weights
    # )

    MODEL_PATH = ARTIFACT_DIR / "nn" / "pattern_classifier.keras"
    SCALER_PATH = ARTIFACT_DIR / "nn" / "scaler.joblib"
    ENCODER_PATH = ARTIFACT_DIR / "nn" / "label_encoder.joblib"
    METADATA_PATH = ARTIFACT_DIR / "nn" / "metadata.json"

    # Persist artifacts for downstream inference pipelines
    # calibrated_model.save(MODEL_PATH, include_optimizer=True)
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
    
    return model,scaler,label_encoder
    # return calibrated_model,scaler,label_encoder

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

    logreg = LogisticRegression(
        C=10,
        max_iter=1000,
        penalty="l2",
        solver='lbfgs',
        n_jobs=-1,
        class_weight="balanced",
        multi_class="multinomial"
    )

    logreg.fit(X, y)
    # Save model
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg, ARTIFACT_DIR / "lr" / "logistic_regression_model.joblib")
    joblib.dump(label_encoder, ARTIFACT_DIR / "lr" / "label_encoder.joblib")
    joblib.dump(scaler, ARTIFACT_DIR / "lr" / "scaler.joblib")
    print(f"Saved logistic regression model and artifacts to {ARTIFACT_DIR}")

    return logreg,scaler,label_encoder

from sklearn.model_selection import StratifiedKFold

def get_folded_splits(fold_count=5,random_state=42,use_only_verified=True,min_samples_per_class=5):
    folded_data = []

    _sd = synthetic_data.copy()
    _vd = verified_communities.copy()

    _vd_c = _vd[TARGET_COLUMN].value_counts()
    _vp_i = _vd_c[_vd_c>=min_samples_per_class].index.tolist()
    _vd   = _vd[_vd[TARGET_COLUMN].isin(_vp_i)]

    skf = StratifiedKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=random_state
    )

    folds = list(skf.split(_vd, _vd[TARGET_COLUMN]))

    for test_fold in range(fold_count):
        val_fold = (test_fold + 1) % fold_count
        train_folds = [
            i for i in range(fold_count)
            if i not in [test_fold, val_fold]
        ]

        train_idx = np.concatenate([folds[i][1] for i in train_folds])
        val_idx   = folds[val_fold][1]
        test_idx  = folds[test_fold][1]

        if EVALUATING_ENABLED:
            vd_train = _vd.iloc[train_idx]
            vd_val   = _vd.iloc[val_idx]
            vd_test  = _vd.iloc[test_idx]
        else:
            vd_train = _vd.iloc[train_idx.tolist()+test_idx.tolist()]
            vd_val   = _vd.iloc[val_idx]
            vd_test  = _vd.iloc[[]]

        if use_only_verified:
            train_data = vd_train
        else:
            train_data = pd.concat([vd_train, _sd], ignore_index=True)

        folded_data.append(
            (train_data, vd_val, vd_test)
        )

    return folded_data


def get_lr_probabilities(fold,train_only=False):
    train_data, val_data, test_data = fold

    if train_only:
        logreg,scaler,label_encoder = lr_train(pd.concat([train_data,val_data],ignore_index=True))
        lr_train_probab = pd.DataFrame()
        lr_test_probab = pd.DataFrame()
        lr_val_probab = pd.DataFrame()
    else:
        logreg,scaler,label_encoder = lr_train(train_data)
        numeric_features = train_data.select_dtypes(include=[np.number]).columns.tolist()

        X_train = scaler.transform(train_data[numeric_features].fillna(0.0).values)
        X_val   = scaler.transform(val_data[numeric_features].fillna(0.0).values)

        lr_train_probab = logreg.predict_proba(X_train)
        lr_val_probab   = logreg.predict_proba(X_val)

        if EVALUATING_ENABLED:
            X_test  = scaler.transform(test_data[numeric_features].fillna(0.0).values)
            lr_test_probab  = logreg.predict_proba(X_test)
            lr_test_probab  = pd.DataFrame(lr_test_probab, columns=label_encoder.classes_)
            lr_test_probab[TARGET_COLUMN]  = test_data[TARGET_COLUMN].values
            lr_test_probab['file']  = test_data['file'].values
        else:
            lr_test_probab = pd.DataFrame()
        lr_train_probab = pd.DataFrame(lr_train_probab, columns=label_encoder.classes_)
        lr_val_probab   = pd.DataFrame(lr_val_probab, columns=label_encoder.classes_)

        lr_train_probab[TARGET_COLUMN] = train_data[TARGET_COLUMN].values
        lr_train_probab['file']  = train_data['file'].values
        lr_val_probab[TARGET_COLUMN]   = val_data[TARGET_COLUMN].values
        lr_val_probab['file']  = val_data['file'].values

    return lr_train_probab,lr_val_probab,lr_test_probab

def get_nn_probabilities(fold,train_only=False):
    train_data, val_data, test_data = fold

    if train_only:
        calibrated_model,scaler,label_encoder = nn_train(pd.concat([train_data,val_data],ignore_index=True))
        nn_train_probab = pd.DataFrame()
        nn_test_probab = pd.DataFrame()
        nn_val_probab = pd.DataFrame()
    else:
        calibrated_model,scaler,label_encoder = nn_train(train_data)

        numeric_features = train_data.select_dtypes(include=[np.number]).columns.tolist()

        X_train = scaler.transform(train_data[numeric_features].fillna(0.0).values)
        X_val   = scaler.transform(val_data[numeric_features].fillna(0.0).values)

        nn_train_probab = calibrated_model.predict(X_train)
        nn_val_probab   = calibrated_model.predict(X_val)

        nn_train_probab = pd.DataFrame(nn_train_probab, columns=label_encoder.classes_)
        nn_val_probab   = pd.DataFrame(nn_val_probab, columns=label_encoder.classes_)

        nn_train_probab[TARGET_COLUMN] = train_data[TARGET_COLUMN].values
        nn_train_probab['file']  = train_data['file'].values
        nn_val_probab[TARGET_COLUMN]   = val_data[TARGET_COLUMN].values
        nn_val_probab['file']  = val_data['file'].values

        if EVALUATING_ENABLED:
            X_test  = scaler.transform(test_data[numeric_features].fillna(0.0).values)
            nn_test_probab  = calibrated_model.predict(X_test)
            nn_test_probab  = pd.DataFrame(nn_test_probab, columns=label_encoder.classes_)
            nn_test_probab[TARGET_COLUMN]  = test_data[TARGET_COLUMN].values
            nn_test_probab['file']  = test_data['file'].values
        else:
            nn_test_probab = pd.DataFrame()

    return nn_train_probab,nn_val_probab,nn_test_probab

def plot_confusion_matrix(y_true, y_pred, title, filename):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import numpy as np

    class_labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    fig, ax = plt.subplots(figsize=(12, 10))  # <-- IMPORTANT
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)

    ax.set_title(title)
    ax.set_xticklabels(class_labels, rotation=90)
    ax.set_yticklabels(class_labels)

    plt.tight_layout()  # <-- KEY FIX
    plt.savefig(filename, bbox_inches="tight", dpi=300)  # <-- KEY FIX
    plt.close(fig)



def _compute_top5_avg(proba_df, numeric_cols):
    """Compute average of top 5 class probabilities for each row, then average across all rows."""
    def row_top5_avg(row):
        sorted_probs = np.sort(row.values)[::-1]  # descending order
        top5 = sorted_probs[:5]
        return np.mean(top5)
    
    row_averages = proba_df[numeric_cols].apply(row_top5_avg, axis=1)
    return row_averages.mean()


def get_meta_probabilities(lr_data, nn_data):
    lr_train_probab, lr_val_probab, lr_test_probab = lr_data
    nn_train_probab, nn_val_probab, nn_test_probab = nn_data

    numeric_cols = nn_train_probab.select_dtypes(include=['number']).columns.tolist()

    # Compute correction factors: avg of top 5 class probabilities over all data
    # Use train + val data for computing the correction factors
    lr_combined = pd.concat([lr_train_probab, lr_val_probab], ignore_index=True)
    nn_combined = pd.concat([nn_train_probab, nn_val_probab], ignore_index=True)
    
    lr_correction = _compute_top5_avg(lr_combined, numeric_cols)
    nn_correction = _compute_top5_avg(nn_combined, numeric_cols)
    
    # Avoid division by zero
    if nn_correction == 0:
        nn_correction = 1e-10
    
    correction_factor = lr_correction / nn_correction

    # Correct for bias in NN predictions
    nn_train_corrected = nn_train_probab[numeric_cols] * correction_factor
    nn_val_corrected = nn_val_probab[numeric_cols] * correction_factor

    # Average corrected NN and LR predictions
    meta_proba_train = (lr_train_probab[numeric_cols] + nn_train_corrected) / 2
    meta_proba_val = (lr_val_probab[numeric_cols] + nn_val_corrected) / 2

    meta_proba_train[TARGET_COLUMN] = lr_train_probab[TARGET_COLUMN].values
    meta_proba_val[TARGET_COLUMN] = lr_val_probab[TARGET_COLUMN].values

    if EVALUATING_ENABLED:
        nn_test_corrected = nn_test_probab[numeric_cols] * correction_factor
        meta_proba_test = (lr_test_probab[numeric_cols] + nn_test_corrected) / 2
        meta_proba_test[TARGET_COLUMN] = lr_test_probab[TARGET_COLUMN].values
    else:
        meta_proba_test = pd.DataFrame(columns=meta_proba_train.columns)

    return meta_proba_train, meta_proba_val, meta_proba_test

def get_prediction(row, margin_threshold=0.1):
    """Classify based on margin between top1 and top2 probabilities.
    
    Returns top1_class if (top1_prob - top2_prob) > margin_threshold, else 'skip'.
    """
    sorted_probs = row.sort_values(ascending=False)
    top1_prob = sorted_probs.iloc[0]
    top2_prob = sorted_probs.iloc[1] if len(sorted_probs) > 1 else 0.0
    top1_class = sorted_probs.index[0]
    
    margin = top1_prob - top2_prob
    if margin > margin_threshold:
        return top1_class
    return "skip"


def classify(margin_threshold, data):
    """Classify data using margin-based threshold and return precision metrics."""
    y_true = data[TARGET_COLUMN].fillna('none')
    y_pred = data.drop([TARGET_COLUMN], axis=1).apply(
        get_prediction, margin_threshold=margin_threshold, axis=1
    )
    
    # Filter out 'skip' predictions for precision calculation
    mask = y_pred != 'skip'
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0, pd.DataFrame(),0.0
    
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    report = classification_report(y_true_filtered, y_pred_filtered, output_dict=True, zero_division=0)
    macro = report['macro avg']
    report_df = pd.DataFrame(report).T
    
    # Also compute coverage (percentage of non-skipped samples)
    coverage = mask.sum() / len(y_true)
    
    return macro['precision'], macro['recall'], macro['f1-score'], report_df, coverage

def find_best_threshold(meta_val_probab):
    """Find the best margin threshold for each class individually to maximize F1-score.
    
    Tests thresholds from 0.0 to 0.5 and picks the one that yields the highest
    F1-score for each specific class.
    """
    best_per_class = {}
    max_f1_per_class = {}
    threshold_df = pd.DataFrame()
    
    print("="*40)
    print("Calculating individual class thresholds:")
    
    for th_int in range(0, 51, 5):
        th = th_int / 100.0
        _, _, _, report_df, _ = classify(th, meta_val_probab)
        
        for cls in report_df.index:
            if cls in ['accuracy', 'macro avg', 'weighted avg', 'none', 'skip']:
                continue
            
            f1_score = report_df.loc[cls, 'f1-score']
            threshold_df.at[th, cls] = f1_score

            
            # Store threshold that gives best F1-score. 
            # If same F1-score, higher threshold is safer.
            if cls not in max_f1_per_class or f1_score >= max_f1_per_class[cls]:
                max_f1_per_class[cls] = f1_score
                best_per_class[cls] = th
    
    print("="*40)
    time_stamp = int(time.time())
    print(f"Calculated individual thresholds for {len(best_per_class)} classes. : {time_stamp}")
    print(best_per_class)
    threshold_df.to_csv(ARTIFACT_DIR / f"class_thresholds{time_stamp}.csv")
    return best_per_class

def get_prediction_trained(row, best_thresholds):
    """Apply class-specific trained margin thresholds.
    
    Returns top1_class if (top1_prob - top2_prob) > threshold[top1_class], else 'skip'.
    """
    sorted_probs = row.sort_values(ascending=False)
    top1_prob = sorted_probs.iloc[0]
    top2_prob = sorted_probs.iloc[1] if len(sorted_probs) > 1 else 0.0
    top1_class = sorted_probs.index[0]
    
    margin = top1_prob - top2_prob
    threshold = best_thresholds.get(top1_class, 0.0)
    
    if margin > threshold:
        return top1_class

    #need to skip
    return "none"

def apply_threshold(data, best_thresholds):
    """Apply class-specific margin thresholds to data."""
    y_true = data[TARGET_COLUMN].fillna('none')
    y_pred = data.drop([TARGET_COLUMN], axis=1).apply(
        get_prediction_trained, best_thresholds=best_thresholds, axis=1
    )
    return y_true, y_pred

def get_prediction_no_threshold(row):
    high_prob = row.max()
    max_class = row.idxmax()
    return max_class

def classify_trained_no_threshold(data):
    y_true = data[TARGET_COLUMN].fillna('none')
    y_pred = data.drop([TARGET_COLUMN], axis=1).apply(get_prediction_no_threshold, axis=1)
    return y_true,y_pred

def get_classification_report(y_true,y_pred):
    ignore_class = "none"
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

    y_train = meta_train_probab[TARGET_COLUMN].values
    y_val   = meta_val_probab[TARGET_COLUMN].values

    numeric_cols = nn_train_probab.select_dtypes(include=['number']).columns.tolist()

    max_train_probab = max_probabilities(nn_train_probab[numeric_cols],lr_train_probab[numeric_cols])
    max_val_probab   = max_probabilities(nn_val_probab[numeric_cols],lr_val_probab[numeric_cols])

    max_train_probab = max_train_probab.add_prefix('max_')
    max_val_probab   = max_val_probab.add_prefix('max_')

    min_train_probab = min_probabilities(nn_train_probab[numeric_cols],lr_train_probab[numeric_cols])
    min_val_probab   = min_probabilities(nn_val_probab[numeric_cols],lr_val_probab[numeric_cols])

    min_train_probab = min_train_probab.add_prefix('min_')
    min_val_probab   = min_val_probab.add_prefix('min_')

    lr_train_probab = lr_train_probab[numeric_cols].add_prefix('lr_')
    lr_val_probab   = lr_val_probab[numeric_cols].add_prefix('lr_')

    nn_train_probab = nn_train_probab[numeric_cols].add_prefix('nn_')
    nn_val_probab   = nn_val_probab[numeric_cols].add_prefix('nn_')

    meta_train_probab = meta_train_probab[numeric_cols].add_prefix('meta_')
    meta_val_probab   = meta_val_probab[numeric_cols].add_prefix('meta_')

    temp_train_set = pd.concat([lr_train_probab,nn_train_probab,meta_train_probab,max_train_probab,pd.Series(y_train, name=TARGET_COLUMN)],axis=1)
    temp_val_set   = pd.concat([lr_val_probab,nn_val_probab,meta_val_probab,max_val_probab,pd.Series(y_val, name=TARGET_COLUMN)],axis=1)

    train_set = pd.concat([temp_train_set,temp_val_set],ignore_index=True)
    if EVALUATING_ENABLED:
        y_test  = meta_test_probab[TARGET_COLUMN].values
        max_test_probab  = max_probabilities(nn_test_probab[numeric_cols],lr_test_probab[numeric_cols])
        max_test_probab  = max_test_probab.add_prefix('max_')
        min_test_probab  = min_probabilities(nn_test_probab[numeric_cols],lr_test_probab[numeric_cols])
        min_test_probab  = min_test_probab.add_prefix('min_')
        lr_test_probab  = lr_test_probab[numeric_cols].add_prefix('lr_')
        nn_test_probab  = nn_test_probab[numeric_cols].add_prefix('nn_')
        meta_test_probab  = meta_test_probab[numeric_cols].add_prefix('meta_')
        temp_test_set  = pd.concat([lr_test_probab,nn_test_probab,meta_test_probab,max_test_probab,pd.Series(y_test, name=TARGET_COLUMN)],axis=1)
        test_set  = temp_test_set

    lr_sec_model,scaler,label_encoder = lr_train(train_set)

    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    lr_sec_model = LogisticRegression(C=1.0,class_weight="balanced",    multi_class="auto")
    lr_sec_model.fit(scaler.fit_transform(train_set.drop([TARGET_COLUMN], axis=1).values), label_encoder.fit_transform(train_set[TARGET_COLUMN].values))
    
    sec_test_pred = []
    if EVALUATING_ENABLED:
        X_test  = scaler.transform(test_set.drop([TARGET_COLUMN], axis=1).values)
        sec_test_pred_enc = lr_sec_model.predict(X_test)
        sec_test_prob = lr_sec_model.predict_proba(X_test)
        sec_test_pred = label_encoder.inverse_transform(sec_test_pred_enc)

    # Save model
    joblib.dump(lr_sec_model, ARTIFACT_DIR / "lr-sec" / "logistic_regression_second_level_model.joblib")
    joblib.dump(scaler, ARTIFACT_DIR / "lr-sec" / "logistic_regression_second_level_scaler.joblib")
    joblib.dump(label_encoder, ARTIFACT_DIR / "lr-sec" / "logistic_regression_second_level_label_encoder.joblib")
    return sec_test_pred,sec_test_prob

def get_second_level_model_predictions_rr(lr_data,nn_data,meta_data):
    lr_train_probab,lr_val_probab,lr_test_probab = lr_data
    nn_train_probab,nn_val_probab,nn_test_probab = nn_data
    meta_train_probab,meta_val_probab,meta_test_probab = meta_data

    y_train = meta_train_probab[TARGET_COLUMN].values
    y_val   = meta_val_probab[TARGET_COLUMN].values

    numeric_cols = nn_train_probab.select_dtypes(include=['number']).columns.tolist()

    max_train_probab = max_probabilities(nn_train_probab[numeric_cols],lr_train_probab[numeric_cols])
    max_val_probab   = max_probabilities(nn_val_probab[numeric_cols],lr_val_probab[numeric_cols])

    max_train_probab = max_train_probab.add_prefix('max_')
    max_val_probab   = max_val_probab.add_prefix('max_')

    min_train_probab = min_probabilities(nn_train_probab[numeric_cols],lr_train_probab[numeric_cols])
    min_val_probab   = min_probabilities(nn_val_probab[numeric_cols],lr_val_probab[numeric_cols])

    min_train_probab = min_train_probab.add_prefix('min_')
    min_val_probab   = min_val_probab.add_prefix('min_')

    lr_train_probab = lr_train_probab[numeric_cols].add_prefix('lr_')
    lr_val_probab   = lr_val_probab[numeric_cols].add_prefix('lr_')

    nn_train_probab = nn_train_probab[numeric_cols].add_prefix('nn_')
    nn_val_probab   = nn_val_probab[numeric_cols].add_prefix('nn_')

    meta_train_probab = meta_train_probab[numeric_cols].add_prefix('meta_')
    meta_val_probab   = meta_val_probab[numeric_cols].add_prefix('meta_')

    temp_train_set = pd.concat([lr_train_probab,nn_train_probab,meta_train_probab,max_train_probab,pd.Series(y_train, name=TARGET_COLUMN)],axis=1)
    temp_val_set   = pd.concat([lr_val_probab,nn_val_probab,meta_val_probab,max_val_probab,pd.Series(y_val, name=TARGET_COLUMN)],axis=1)

    train_set = pd.concat([temp_train_set,temp_val_set],ignore_index=True)

    
    if EVALUATING_ENABLED:
        y_test  = meta_test_probab[TARGET_COLUMN].values
        max_test_probab  = max_probabilities(nn_test_probab[numeric_cols],lr_test_probab[numeric_cols])
        max_test_probab  = max_test_probab.add_prefix('max_')
        min_test_probab  = min_probabilities(nn_test_probab[numeric_cols],lr_test_probab[numeric_cols])
        min_test_probab  = min_test_probab.add_prefix('min_')
        lr_test_probab  = lr_test_probab[numeric_cols].add_prefix('lr_')
        nn_test_probab  = nn_test_probab[numeric_cols].add_prefix('nn_')
        meta_test_probab  = meta_test_probab[numeric_cols].add_prefix('meta_')
        temp_test_set  = pd.concat([lr_test_probab,nn_test_probab,meta_test_probab,max_test_probab,pd.Series(y_test, name=TARGET_COLUMN)],axis=1)
        test_set  = temp_test_set


    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    rr_sec_model = RidgeClassifier(alpha=1.0,random_state=42)
    rr_sec_model.fit(scaler.fit_transform(train_set.drop(TARGET_COLUMN, axis=1).values), label_encoder.fit_transform(train_set[TARGET_COLUMN].values))
    
    sec_test_pred = []
    if EVALUATING_ENABLED:
        X_test  = scaler.transform(test_set.drop(TARGET_COLUMN, axis=1).values)
        sec_test_pred_enc = rr_sec_model.predict(X_test)
        sec_test_pred = label_encoder.inverse_transform(sec_test_pred_enc)
    # Save model
    joblib.dump(rr_sec_model, ARTIFACT_DIR / "rr-sec" / "ridge_regression_second_level_model.joblib")
    joblib.dump(scaler, ARTIFACT_DIR / "rr-sec" / "ridge_regression_second_level_scaler.joblib")
    joblib.dump(label_encoder, ARTIFACT_DIR / "rr-sec" / "ridge_regression_second_level_label_encoder.joblib")
    return sec_test_pred

def get_class_by_avg_threshold(row,mean_thresholds):
    mean_thresholds['none'] = 0
    row = row.drop('verified_pattern')
    max_prob = row.max()
    max_prob_index = row.idxmax()
    # return max_prob_index

    threshold = mean_thresholds[max_prob_index]
    if max_prob >= threshold:
        return max_prob_index
    else:
        return 'none'

############################
#  MAIN LOGIC
############################

def main():
    folds = get_folded_splits(fold_count=5,random_state=42,use_only_verified=True,min_samples_per_class=20)
    prediction_df = pd.DataFrame(columns=[TARGET_COLUMN,'file','avg_threshold_class','threshold_class','without_threshold_class','second_level_model_class_lr','second_level_model_class_rr'])
    prediction_prob_df = pd.DataFrame()
    probab_distribution = None

    probab_distributions = pd.DataFrame()
    best_thresholds = []

    for fold in folds:
        lr_train_probab,lr_val_probab,lr_test_probab = get_lr_probabilities(fold)
        nn_train_probab,nn_val_probab,nn_test_probab = get_nn_probabilities(fold)

        if EVALUATING_ENABLED:
            lr_test_file = lr_test_probab['file']
            # lr_test_code_summary = lr_test_probab['code_summary']

        nn_train_probab.drop('file',axis=1,inplace=True,errors='ignore')
        nn_val_probab.drop('file',axis=1,inplace=True,errors='ignore')
        nn_test_probab.drop('file',axis=1,inplace=True,errors='ignore')
        lr_test_probab.drop('file',axis=1,inplace=True,errors='ignore')
        lr_val_probab.drop('file',axis=1,inplace=True,errors='ignore')
        lr_train_probab.drop('file',axis=1,inplace=True,errors='ignore')

        nn_data = (nn_train_probab,nn_val_probab,nn_test_probab)
        lr_data = (lr_train_probab,lr_val_probab,lr_test_probab)
        meta_train_probab,meta_val_probab,meta_test_probab = get_meta_probabilities(lr_data,nn_data)

        meta_data = (meta_train_probab,meta_val_probab,meta_test_probab)
        if probab_distribution is not None:
            probab_distribution = pd.concat([probab_distribution,meta_val_probab])
        else:
            probab_distribution = meta_val_probab
        
        best_threshold = find_best_threshold(meta_val_probab)

        best_thresholds.append(best_threshold)
        probab_distributions = pd.concat([probab_distributions, meta_test_probab], ignore_index=True)
        
        second_level_model_predictions_lr, second_level_model_prob_lr = get_second_level_model_predictions_lr(lr_data,nn_data,meta_data)
        second_level_model_predictions_rr = get_second_level_model_predictions_rr(lr_data,nn_data,meta_data)
        
        if(EVALUATING_ENABLED):
            threshold_predictions = apply_threshold(meta_test_probab,best_threshold)
            no_threshold_predictions = classify_trained_no_threshold(meta_test_probab)
            temp_predictions = pd.DataFrame()
            temp_predictions[TARGET_COLUMN] = meta_test_probab[TARGET_COLUMN]
            temp_predictions['threshold_class'] = threshold_predictions[1]
            temp_predictions['without_threshold_class'] = no_threshold_predictions[1]
            temp_predictions['second_level_model_class_lr'] = second_level_model_predictions_lr
            temp_predictions['second_level_model_class_rr'] = second_level_model_predictions_rr
            temp_predictions['file'] = lr_test_file
            prediction_df = pd.concat([prediction_df, temp_predictions], ignore_index=True)

            # Second level model LR + Probabilities
            temp_predictions = temp_predictions.drop(['threshold_class','without_threshold_class','second_level_model_class_rr'], axis=1)
            # temp_predictions['code_summary'] = lr_test_code_summary
            temp_predictions = pd.concat([temp_predictions, pd.DataFrame(second_level_model_prob_lr, columns=meta_test_probab.drop([TARGET_COLUMN], axis=1).columns)], axis=1)
            prediction_prob_df = pd.concat([prediction_prob_df, temp_predictions], ignore_index=True)

        
        else:
            lr_train_probab,lr_val_probab,lr_test_probab = get_lr_probabilities(fold,train_only=True)
            nn_train_probab,nn_val_probab,nn_test_probab = get_nn_probabilities(fold,train_only=True)


    if EVALUATING_ENABLED:
        best_thresholds = pd.DataFrame(best_thresholds)

        # Log classification reports
        print("-"*20)

        prediction_df['avg_threshold_class'] = probab_distributions.apply(get_class_by_avg_threshold, axis=1, mean_thresholds=best_thresholds.mean())
        avg_threshold_report = get_classification_report(prediction_df[TARGET_COLUMN], prediction_df['avg_threshold_class'])
        threshold_report = get_classification_report(prediction_df[TARGET_COLUMN], prediction_df['threshold_class'])
        without_threshold_report = get_classification_report(prediction_df[TARGET_COLUMN], prediction_df['without_threshold_class'])
        stacking_lr_report = get_classification_report(prediction_df[TARGET_COLUMN], prediction_df['second_level_model_class_lr'])
        stacking_rr_report = get_classification_report(prediction_df[TARGET_COLUMN], prediction_df['second_level_model_class_rr'])
        
        print(f"avg thresholding [f1]: {avg_threshold_report[2]}     [precision]: {avg_threshold_report[0]}     [recall]: {avg_threshold_report[1]}")
        print(f"thresholding     [f1]: {threshold_report[2]}     [precision]: {threshold_report[0]}     [recall]: {threshold_report[1]}")
        print(f"no thresholding  [f1]: {without_threshold_report[2]}    [precision]: {without_threshold_report[0]}     [recall]: {without_threshold_report[1]}")
        print(f"stacking lr      [f1]: {stacking_lr_report[2]}     [precision]: {stacking_lr_report[0]}     [recall]: {stacking_lr_report[1]}")
        print(f"stacking rr      [f1]: {stacking_rr_report[2]}     [precision]: {stacking_rr_report[0]}     [recall]: {stacking_rr_report[1]}")
        
        # Compusion matrix plots
        plot_confusion_matrix(prediction_df[TARGET_COLUMN], prediction_df['avg_threshold_class'], title='Confusion Matrix with Average Thresholding', filename='./models/metrics/confusion_matrix_with_avg_threshold.png')
        plot_confusion_matrix(prediction_df[TARGET_COLUMN], prediction_df['threshold_class'], title='Confusion Matrix with Thresholding', filename='./models/metrics/confusion_matrix_with_threshold.png')
        plot_confusion_matrix(prediction_df[TARGET_COLUMN], prediction_df['without_threshold_class'], title='Confusion Matrix without Thresholding', filename='./models/metrics/confusion_matrix_without_threshold.png')
        plot_confusion_matrix(prediction_df[TARGET_COLUMN], prediction_df['second_level_model_class_lr'], title='Confusion Matrix Second Level Model LR', filename='./models/metrics/confusion_matrix_second_level_model_lr.png')
        plot_confusion_matrix(prediction_df[TARGET_COLUMN], prediction_df['second_level_model_class_rr'], title='Confusion Matrix Second Level Model RR', filename='./models/metrics/confusion_matrix_second_level_model_rr.png')

        result_dir = f"./models/metrics/{run_time}"
        os.makedirs(result_dir)

        # Save final predictions
        prediction_df.to_csv(f'{result_dir}/ensemble_classification_predictions.csv', index=False)
        prediction_prob_df.to_csv(f'{result_dir}/l2_lr_ensemble_classification_predictions_with_probabilities.csv', index=False)
        get_classification_report(prediction_df[TARGET_COLUMN], prediction_df['threshold_class'])[3].to_csv(f'{result_dir}/ensemble_classification_report_with_threshold.csv')
        get_classification_report(prediction_df[TARGET_COLUMN], prediction_df['without_threshold_class'])[3].to_csv(f'{result_dir}/ensemble_classification_report_without_threshold.csv')
        get_classification_report(prediction_df[TARGET_COLUMN], prediction_df['second_level_model_class_lr'])[3].to_csv(f'{result_dir}/ensemble_classification_report_second_level_model_lr.csv')
        get_classification_report(prediction_df[TARGET_COLUMN], prediction_df['second_level_model_class_rr'])[3].to_csv(f'{result_dir}/ensemble_classification_report_second_level_model_rr.csv')

        probab_distributions.to_csv(f'{result_dir}/test_probability_distributions.csv', index=False)

        best_thresholds.to_csv(f'{result_dir}/best_thresholds_per_fold.csv', index=False)
    else:
        overrall_best_thresholds = find_best_threshold(probab_distribution)
        with open('./overall_best_thresholds.json', 'w') as f:
            json.dump(overrall_best_thresholds, f, indent=2)
if __name__ == "__main__":
    main()