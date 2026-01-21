"""
Pattern classification validation using Logistic Regression and Neural Network models.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def load_data(labeled_data_path: str, embeddings_path: str) -> pd.DataFrame:
    """
    Load and merge labeled data with embeddings.
    
    Args:
        labeled_data_path: Path to labeled data CSV file
        embeddings_path: Path to embeddings CSV file
        
    Returns:
        Merged DataFrame with labeled data and embeddings
    """
    labeled_data = pd.read_csv(labeled_data_path)
    embeddings = pd.read_csv(embeddings_path)
    
    merged_data = pd.merge(labeled_data, embeddings, on='file', how='inner')
    merged_data = merged_data.drop(columns=['Unnamed: 0'], errors='ignore')
    
    # Filter out rows without verified patterns
    merged_data = merged_data[~merged_data['verified_pattern'].isna()]
    
    return merged_data


def preprocess_data(data: pd.DataFrame) -> tuple:
    """
    Preprocess data by extracting features and creating binary labels.
    
    Args:
        data: DataFrame containing embeddings and verified patterns
        
    Returns:
        Tuple of (features, labels, feature_columns)
    """
    # Extract embedding dimensions
    feature_columns = [col for col in data.columns if col.startswith('dim_')]
    
    # Create binary labels: 'pattern' or 'none'
    data['label'] = data['verified_pattern'].apply(
        lambda x: 'none' if x == 'none' else 'pattern'
    )
    
    features = data[feature_columns]
    labels = data['label']
    
    return features, labels, feature_columns


def evaluate_logistic_regression(
    features: pd.DataFrame, 
    labels: pd.Series,
    n_splits: int = 3,
    random_state: int = 42
) -> dict:
    """
    Train and evaluate Logistic Regression model using cross-validation.
    
    Args:
        features: Feature matrix
        labels: Target labels
        n_splits: Number of cross-validation folds
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing model name, F1 scores, trained model, scaler, and encoder
    """
    # Initialize model and preprocessing
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    # Scale features and encode labels
    features_scaled = scaler.fit_transform(features)
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Cross-validation
    cv_splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    
    cv_scores = cross_val_score(
        model,
        features_scaled,
        labels_encoded,
        cv=cv_splitter,
        scoring="f1_macro",
    )
    
    # Train final model on all data for confusion matrix
    model.fit(features_scaled, labels_encoded)
    
    return {
        'model': 'Logistic Regression',
        'cv_scores': cv_scores,
        'mean_f1': cv_scores.mean(),
        'std_f1': cv_scores.std(),
        'trained_model': model,
        'scaler': scaler,
        'label_encoder': label_encoder
    }


def build_keras_classifier(input_dim: int, num_classes: int) -> Sequential:
    """
    Build a Keras neural network classifier with the architecture from model_fit.py.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras Sequential model
    """
    regularizer = tf.keras.regularizers.l2(1e-4)
    
    model = Sequential([
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
        Dense(num_classes, activation="softmax"),
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def evaluate_neural_network(
    features: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = 3,
    random_state: int = 42,
    epochs: int = 100,
    batch_size: int = 64
) -> dict:
    """
    Train and evaluate Keras Neural Network model using cross-validation.
    
    Args:
        features: Feature matrix
        labels: Target labels
        n_splits: Number of cross-validation folds
        random_state: Random seed for reproducibility
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary containing model name, F1 scores, trained model, scaler, and encoder
    """
    # Set random seeds for reproducibility
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    
    # Initialize preprocessing
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    # Scale features and encode labels
    features_scaled = scaler.fit_transform(features)
    labels_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    # Manual cross-validation for Keras
    cv_splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    
    cv_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(features_scaled, labels_encoded)):
        print(f"\nTraining fold {fold_idx + 1}/{n_splits}...")
        
        X_train_fold = features_scaled[train_idx]
        y_train_fold = labels_encoded[train_idx]
        X_val_fold = features_scaled[val_idx]
        y_val_fold = labels_encoded[val_idx]
        
        # Build and train model
        model = build_keras_classifier(features_scaled.shape[1], num_classes)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                min_delta=1e-4,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=0
            )
        ]
        
        model.fit(
            X_train_fold,
            y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate on validation fold
        y_pred_fold = np.argmax(model.predict(X_val_fold, verbose=0), axis=1)
        
        # Calculate F1-score
        from sklearn.metrics import f1_score
        f1 = f1_score(y_val_fold, y_pred_fold, average='macro', zero_division=0)
        cv_scores.append(f1)
        print(f"Fold {fold_idx + 1} F1 Score: {f1:.4f}")
    
    cv_scores = np.array(cv_scores)
    
    # Train final model on all data for confusion matrix
    print("\nTraining final model on full dataset...")
    final_model = build_keras_classifier(features_scaled.shape[1], num_classes)
    
    # Split for validation during training
    X_train, X_val, y_train, y_val = train_test_split(
        features_scaled, labels_encoded,
        test_size=0.2,
        random_state=random_state,
        stratify=labels_encoded
    )
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=0
        )
    ]
    
    final_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return {
        'model': 'Neural Network (Keras)',
        'architecture': '768-512-256-128',
        'cv_scores': cv_scores,
        'mean_f1': cv_scores.mean(),
        'std_f1': cv_scores.std(),
        'trained_model': final_model,
        'scaler': scaler,
        'label_encoder': label_encoder
    }


def generate_confusion_matrix(
    model,
    scaler,
    label_encoder,
    features: pd.DataFrame,
    labels: pd.Series,
    model_name: str,
    output_dir: Path,
    is_keras: bool = False
) -> np.ndarray:
    """
    Generate and save confusion matrix for a trained model.
    
    Args:
        model: Trained classifier
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        features: Feature matrix
        labels: True labels
        model_name: Name of the model for saving
        output_dir: Directory to save confusion matrix
        is_keras: Whether the model is a Keras model
        
    Returns:
        Confusion matrix as numpy array
    """
    # Scale features and encode labels
    features_scaled = scaler.transform(features)
    labels_encoded = label_encoder.transform(labels)
    
    # Generate predictions
    if is_keras:
        predictions = np.argmax(model.predict(features_scaled, verbose=0), axis=1)
    else:
        predictions = model.predict(features_scaled)
    
    # Compute confusion matrix
    cm = confusion_matrix(labels_encoded, predictions)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")
    
    return cm


def print_results(results: dict) -> None:
    """
    Print evaluation results in a formatted manner.
    
    Args:
        results: Dictionary containing evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Model: {results['model']}")
    if 'hidden_layers' in results:
        print(f"Hidden Layers: {results['hidden_layers']}")
    if 'architecture' in results:
        print(f"Architecture: {results['architecture']}")
    print(f"{'='*60}")
    print(f"Cross-Validation Scores: {results['cv_scores']}")
    print(f"Mean F1 Score: {results['mean_f1']:.4f} (+/- {results['std_f1']:.4f})")
    print(f"{'='*60}\n")


def main():
    """
    Main function to orchestrate the validation process.
    """
    # Define data paths
    base_path = Path('/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline')
    labeled_data_path = base_path / 'data/datasets/labeled_data.csv'
    embeddings_path = base_path / 'data/datasets/embeddings.csv'
    
    # Load and preprocess data
    print("Loading data...")
    data = load_data(str(labeled_data_path), str(embeddings_path))
    print(f"Loaded {len(data)} samples")
    
    print("\nPreprocessing data...")
    features, labels, feature_columns = preprocess_data(data)
    print(f"Features shape: {features.shape}")
    print(f"Label distribution:\n{labels.value_counts()}")
    
    # Evaluate Logistic Regression
    print("\n" + "="*60)
    print("Evaluating Logistic Regression...")
    print("="*60)
    lr_results = evaluate_logistic_regression(features, labels)
    print_results(lr_results)
    
    # Evaluate Neural Network
    print("="*60)
    print("Evaluating Neural Network...")
    print("="*60)
    nn_results = evaluate_neural_network(features, labels)
    print_results(nn_results)
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"Logistic Regression - Mean F1: {lr_results['mean_f1']:.4f}")
    print(f"Neural Network      - Mean F1: {nn_results['mean_f1']:.4f}")
    
    best_model = "Neural Network" if nn_results['mean_f1'] > lr_results['mean_f1'] else "Logistic Regression"
    print(f"\nBest Model: {best_model}")
    print("="*60)
    
    # Generate and save confusion matrices
    print("\n" + "="*60)
    print("GENERATING CONFUSION MATRICES")
    print("="*60)
    
    output_dir = Path(__file__).parent / 'models/metrics'
    
    # Logistic Regression confusion matrix
    print("\nGenerating confusion matrix for Logistic Regression...")
    lr_cm = generate_confusion_matrix(
        lr_results['trained_model'],
        lr_results['scaler'],
        lr_results['label_encoder'],
        features,
        labels,
        'Logistic Regression',
        output_dir,
        is_keras=False
    )
    
    # Neural Network confusion matrix
    print("\nGenerating confusion matrix for Neural Network...")
    nn_cm = generate_confusion_matrix(
        nn_results['trained_model'],
        nn_results['scaler'],
        nn_results['label_encoder'],
        features,
        labels,
        'Neural Network (Keras)',
        output_dir,
        is_keras=True
    )
    
    print("\n" + "="*60)
    print("All confusion matrices saved successfully!")
    print("="*60)


if __name__ == "__main__":
    main()