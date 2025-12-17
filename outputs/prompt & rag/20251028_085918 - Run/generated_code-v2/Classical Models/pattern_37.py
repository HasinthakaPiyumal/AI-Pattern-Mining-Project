import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Generation (for demonstration purposes)
def generate_dummy_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 70, num_samples),
        'income': np.random.randint(20000, 150000, num_samples),
        'loan_amount': np.random.randint(1000, 50000, num_samples),
        'loan_term': np.random.choice([12, 24, 36, 48, 60], num_samples),
        'employment_status': np.random.choice(['Employed', 'Self-employed', 'Unemployed', 'Retired'], num_samples, p=[0.6, 0.2, 0.1, 0.1]),
        'education': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], num_samples, p=[0.25, 0.4, 0.25, 0.1]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], num_samples, p=[0.4, 0.5, 0.1]),
        'has_mortgage': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
        'num_credit_cards': np.random.randint(0, 5, num_samples),
        'credit_score': np.random.randint(300, 850, num_samples),
        'default': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]) # 0: No Default, 1: Default
    }
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    for col in ['income', 'employment_status', 'credit_score']:
        df.loc[df.sample(frac=0.05).index, col] = np.nan
        
    # Make 'default' correlated with some features
    df.loc[(df['credit_score'] < 600) | (df['income'] < 30000) & (df['loan_amount'] > 30000), 'default'] = np.random.choice([0, 1], sum((df['credit_score'] < 600) | (df['income'] < 30000) & (df['loan_amount'] > 30000)), p=[0.3, 0.7])
    df.loc[(df['credit_score'] > 750) & (df['income'] > 80000) & (df['loan_amount'] < 10000), 'default'] = np.random.choice([0, 1], sum((df['credit_score'] > 750) & (df['income'] > 80000) & (df['loan_amount'] < 10000)), p=[0.9, 0.1])
    
    return df

df = generate_dummy_data()

# Separate target variable
X = df.drop('default', axis=1)
y = df['default']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

# 2. Preprocessing Pipelines
# Numerical pipeline: impute missing values with mean, then scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute missing values with most frequent, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Model Training and Evaluation
models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(random_state=42)
}

results = {}
best_model = None
best_roc_auc = -1

for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'model': pipeline
    }
    
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} - Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_model = pipeline
        best_model_name = name

print(f"\nBest performing model: {best_model_name} with ROC AUC: {best_roc_auc:.4f}")

# 4. Model Persistence
model_filename = f'{best_model_name.replace(" ", "_").lower()}_credit_risk_model.joblib'
joblib.dump(best_model, model_filename)
print(f"Best model saved to {model_filename}")

# Example of loading and using the model
# loaded_model = joblib.load(model_filename)
# dummy_new_data = generate_dummy_data(num_samples=5).drop('default', axis=1)
# predictions = loaded_model.predict(dummy_new_data)
# print(f"\nPredictions on new data: {predictions}")

# 5. Interpretability (Feature Importance for Tree-based models)
print("\n--- Feature Importance Analysis ---")
# Get feature names after one-hot encoding
def get_feature_names(preprocessor, numerical_features, categorical_features):
    ohe_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_features = list(numerical_features) + list(ohe_features)
    return all_features

for name, result in results.items():
    if isinstance(result['model'].named_steps['classifier'], (RandomForestClassifier, xgb.XGBClassifier, lgb.LGBMClassifier)):
        print(f"\nFeature Importance for {name}:")
        classifier = result['model'].named_steps['classifier']
        feature_importances = classifier.feature_importances_
        
        # Get feature names from the preprocessor
        preprocessed_feature_names = get_feature_names(preprocessor, numerical_features, categorical_features)

        if len(feature_importances) == len(preprocessed_feature_names):
            importance_df = pd.DataFrame({
                'Feature': preprocessed_feature_names,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)
            print(importance_df.head(10))
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
            plt.title(f'{name} - Top 10 Feature Importances')
            plt.show()
        else:
            print("Could not match feature importances to feature names. This might happen with specific model/pipeline configurations.")
