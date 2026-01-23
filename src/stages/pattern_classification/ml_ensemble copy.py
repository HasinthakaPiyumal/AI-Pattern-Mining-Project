import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder,StandardScaler

def run_lr_ensemble(X, y, n_splits=5, model_params=None, random_state=42):
    X_vals = X.values if isinstance(X, pd.DataFrame) else X
    y_vals = y.values if isinstance(y, pd.Series) else y
    
    le = LabelEncoder()
    y_vals = le.fit_transform(y_vals)
    # scaler = StandardScaler()
    # X_vals = scaler.fit_transform(X_vals)
    unique_classes = np.unique(y_vals)
    n_classes = len(unique_classes)
    
    oof_probs = np.zeros((len(y_vals), n_classes))
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Best params found via tuning
    if model_params is None:
        model_params = {'C': 10, 'class_weight': 'balanced', 'solver': 'lbfgs'}
    
    print(f"Starting {n_splits}-fold Cross-Validation with parameters: {model_params}...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_vals, y_vals), 1):
        X_train, X_val = X_vals[train_idx], X_vals[val_idx]
        y_train, y_val = y_vals[train_idx], y_vals[val_idx]
        
        clf = LogisticRegression(**model_params, random_state=random_state, max_iter=2000, multi_class='auto')
        clf.fit(X_train, y_train)
        
        val_probs = clf.predict_proba(X_val)
        
        class_indices = [np.where(unique_classes == c)[0][0] for c in clf.classes_]
        oof_probs[val_idx[:, None], class_indices] = val_probs
        
        print(f"Fold {fold}/{n_splits} completed.")
        
    probs_df = pd.DataFrame(oof_probs, columns=unique_classes)
    
    y_pred_indices = probs_df.idxmax(axis=1).values
    y_pred = le.inverse_transform(y_pred_indices)
    
    report = classification_report(le.inverse_transform(y_vals), y_pred)
    
    print("\nClassification Report (OOF):")
    print(report)
    
    return probs_df, report

if __name__ == "__main__":
    TARGET_COLUMN = "verified_pattern"

    labeled_data = pd.read_csv('/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/data/datasets/labeled_data.csv')
    embeddings   = pd.read_csv('/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/data/datasets/embeddings.csv')
    data = pd.merge(labeled_data, embeddings, on='file')

    synthetic_data       = pd.DataFrame()
    verified_communities = data[~data[TARGET_COLUMN].isna()]

    counts = verified_communities['verified_pattern'].value_counts()
    to_keep = counts[counts >= 20].index
    verified_communities = verified_communities[verified_communities['verified_pattern'].isin(to_keep)]

    # Trim 40 samples from the 'none' class to reduce imbalance
    none_label = 'none'
    if none_label in verified_communities[TARGET_COLUMN].values:
        none_indices = verified_communities[verified_communities[TARGET_COLUMN] == none_label].index
        trim_n = min(50, len(none_indices))
        drop_indices = np.random.default_rng(42).choice(none_indices, size=trim_n, replace=False)
        verified_communities = verified_communities.drop(index=drop_indices)


    columns = [col for col in verified_communities.columns if col.startswith('dim_')]
    X = verified_communities[columns]
    y = verified_communities[TARGET_COLUMN]
    
    # Run ensemble with best params (default in function)
    probs_df, report = run_lr_ensemble(X, y)
    probs_df.to_csv("probs.csv", index=False)
    print(report)