import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def create_dummy_data(filepath="heart_disease_data.csv"):
    np.random.seed(42)
    data = {
        'age': np.random.randint(29, 77, 100),
        'sex': np.random.choice([0, 1], 100), 
        'chest_pain_type': np.random.choice([0, 1, 2, 3], 100),
        'resting_bp_s': np.random.randint(90, 200, 100),
        'cholesterol': np.random.randint(120, 564, 100),
        'fasting_blood_sugar': np.random.choice([0, 1], 100),
        'resting_ecg': np.random.choice([0, 1, 2], 100),
        'max_heart_rate': np.random.randint(71, 202, 100),
        'exercise_angina': np.random.choice([0, 1], 100),
        'oldpeak': np.round(np.random.uniform(0.0, 6.2, 100), 1),
        'st_slope': np.random.choice([0, 1, 2], 100),
        'num_major_vessels': np.random.choice([0, 1, 2, 3, 4], 100),
        'thal': np.random.choice([0, 1, 2, 3], 100),
        'target': np.random.choice([0, 1], 100) 
    }
    df = pd.DataFrame(data)
    
    missing_indices = np.random.choice(df.index, 10, replace=False)
    df.loc[missing_indices, 'cholesterol'] = np.nan
    
    df.to_csv(filepath, index=False)
    return df

def train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"--- {model_name} Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\n")
    return model

def predict_cardiovascular_disease(model_pipeline, new_data):
    prediction = model_pipeline.predict(new_data)
    prediction_proba = model_pipeline.predict_proba(new_data)[:, 1] if hasattr(model_pipeline, "predict_proba") else None
    return prediction, prediction_proba

if __name__ == "__main__":
    data_filepath = "heart_disease_data.csv"
    df = create_dummy_data(data_filepath)

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
    if 'cholesterol' in numerical_features:
        numerical_features.remove('cholesterol') 
    numerical_features_with_nan = ['cholesterol'] 

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    numerical_transformer_no_nan = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_nan', numerical_transformer, numerical_features_with_nan),
            ('num', numerical_transformer_no_nan, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    logistic_regression_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                                   ('classifier', LogisticRegression(random_state=42, solver='liblinear'))])
    svm_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', SVC(probability=True, random_state=42))])
    decision_tree_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                             ('classifier', DecisionTreeClassifier(random_state=42))])

    print("Training and evaluating models...")
    trained_lr_model = train_and_evaluate_model("Logistic Regression", logistic_regression_pipeline, X_train, X_test, y_train, y_test)
    trained_svm_model = train_and_evaluate_model("Support Vector Machine", svm_pipeline, X_train, X_test, y_train, y_test)
    trained_dt_model = train_and_evaluate_model("Decision Tree", decision_tree_pipeline, X_train, X_test, y_train, y_test)
    
    joblib.dump(trained_lr_model, 'logistic_regression_model.pkl')
    joblib.dump(trained_svm_model, 'svm_model.pkl')
    joblib.dump(trained_dt_model, 'decision_tree_model.pkl')
    print("Models saved as .pkl files.")

    print("\nDemonstrating prediction with a new sample...")
    new_patient_data = pd.DataFrame([{
        'age': 55,
        'sex': 1,
        'chest_pain_type': 2,
        'resting_bp_s': 130,
        'cholesterol': 240,
        'fasting_blood_sugar': 0,
        'resting_ecg': 1,
        'max_heart_rate': 150,
        'exercise_angina': 0,
        'oldpeak': 1.0,
        'st_slope': 2,
        'num_major_vessels': 0,
        'thal': 2
    }])
    
    prediction, probability = predict_cardiovascular_disease(trained_lr_model, new_patient_data)
    print(f"Prediction for new patient (Logistic Regression): {'Disease' if prediction[0] == 1 else 'No Disease'}")
    if probability is not None:
        print(f"Probability of Disease: {probability[0]:.4f}")