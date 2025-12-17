import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def generate_dummy_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 90, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'ethnicity': np.random.choice(['White', 'Black', 'Asian', 'Other'], n_samples),
        'num_diagnoses': np.random.randint(1, 10, n_samples),
        'num_procedures': np.random.randint(0, 5, n_samples),
        'length_of_stay': np.random.randint(1, 30, n_samples),
        'num_medications': np.random.randint(5, 30, n_samples),
        'diabetes': np.random.randint(0, 2, n_samples),
        'heart_disease': np.random.randint(0, 2, n_samples),
        'hypertension': np.random.randint(0, 2, n_samples),
        'previous_admissions': np.random.randint(0, 5, n_samples),
        'readmitted': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    
    missing_mask = np.random.rand(n_samples) < 0.05
    df.loc[missing_mask, 'length_of_stay'] = np.nan
    
    return df

def main():
    df = generate_dummy_data()

    X = df.drop('readmitted', axis=1)
    y = df['readmitted']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numerical_features = ['age', 'num_diagnoses', 'num_procedures', 'length_of_stay', 'num_medications', 'previous_admissions']
    categorical_features = ['gender', 'ethnicity', 'diabetes', 'heart_disease', 'hypertension']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    param_grids = {
        'Logistic Regression': {
            'classifier__C': [0.1, 1.0, 10.0]
        },
        'Random Forest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10]
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1]
        }
    }

    best_model = None
    best_model_name = ""
    best_roc_auc = -1

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])

        grid_search = GridSearchCV(pipeline, param_grids[name], cv=StratifiedKFold(n_splits=3), scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        print(f"Best parameters for {name}: {grid_search.best_params_}")
        y_pred = grid_search.predict(X_test)
        y_proba = grid_search.predict_proba(X_test)[:, 1]

        current_roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC on test set for {name}: {current_roc_auc:.4f}")
        print(classification_report(y_test, y_pred))

        if current_roc_auc > best_roc_auc:
            best_roc_auc = current_roc_auc
            best_model = grid_search.best_estimator_
            best_model_name = name

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {current_roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.show()

    print(f"\nBest performing model: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")
    joblib.dump(best_model, 'best_readmission_model.pkl')
    print("Best model saved as 'best_readmission_model.pkl'")

    if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        print("\n--- Feature Importances (from best model) ---")
        
        # Get feature names after one-hot encoding
        ohe_feature_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        all_feature_names = numerical_features + list(ohe_feature_names)

        importances = best_model.named_steps['classifier'].feature_importances_
        feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        print(feature_importance_df.head(10))

    elif hasattr(best_model.named_steps['classifier'], 'coef_'):
        print("\n--- Feature Coefficients (from best model) ---")
        
        # Get feature names after one-hot encoding
        ohe_feature_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        all_feature_names = numerical_features + list(ohe_feature_names)
        
        coefficients = best_model.named_steps['classifier'].coef_[0]
        feature_coef_df = pd.DataFrame({'feature': all_feature_names, 'coefficient': coefficients})
        feature_coef_df = feature_coef_df.sort_values(by='coefficient', ascending=False)
        print(feature_coef_df.head(10))


if __name__ == "__main__":
    main()