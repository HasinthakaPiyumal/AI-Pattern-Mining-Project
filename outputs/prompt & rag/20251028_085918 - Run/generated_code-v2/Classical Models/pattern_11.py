import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import joblib

def generate_dummy_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 90, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'InsuranceType': np.random.choice(['Private', 'Medicare', 'Medicaid'], num_samples),
        'NumDiagnoses': np.random.randint(1, 10, num_samples),
        'NumProcedures': np.random.randint(0, 5, num_samples),
        'MedicationCount': np.random.randint(1, 15, num_samples),
        'LOS': np.random.randint(1, 30, num_samples), # Length of Stay
        'Diabetes': np.random.randint(0, 2, num_samples), # Binary (0/1)
        'HeartDisease': np.random.randint(0, 2, num_samples), # Binary (0/1)
        'Cancer': np.random.randint(0, 2, num_samples), # Binary (0/1)
        'Readmitted': np.random.randint(0, 2, num_samples) # Target variable (0/1)
    }
    df = pd.DataFrame(data)

    # Introduce some correlation to make the target more realistic
    df['Readmitted'] = df.apply(
        lambda row: 1 if (row['Age'] > 70 and row['NumDiagnoses'] > 5 and row['LOS'] > 15) or \
                        (row['Diabetes'] == 1 and row['HeartDisease'] == 1 and row['MedicationCount'] > 10) else row['Readmitted'],
        axis=1
    )
    # Ensure a reasonable distribution of the target variable
    readmitted_count = df['Readmitted'].sum()
    if readmitted_count < num_samples * 0.2:
        # Artificially increase readmissions if too low for demonstration
        high_risk_indices = df[(df['Age'] > 65) & (df['NumDiagnoses'] >= 5) & (df['LOS'] >= 10)].index
        num_to_flip = int(num_samples * 0.25) - readmitted_count # Aim for ~25% readmission rate
        if num_to_flip > 0 and len(high_risk_indices) > 0:
            flip_indices = np.random.choice(high_risk_indices, min(num_to_flip, len(high_risk_indices)), replace=False)
            df.loc[flip_indices, 'Readmitted'] = 1

    return df

def main():
    print("Generating dummy patient data...")
    df = generate_dummy_data(num_samples=2000)
    print(f"Generated {len(df)} samples. Readmission rate: {df['Readmitted'].mean():.2f}")

    # Define features and target
    X = df.drop('Readmitted', axis=1)
    y = df['Readmitted']

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    trained_models = {}
    print("\n--- Training and Evaluating Models ---")

    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

        print(f"Evaluating {name}...")
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")

        # Save the trained model
        model_filename = f"{name.replace(' ', '_').lower()}_model.joblib"
        joblib.dump(pipeline, model_filename)
        print(f"  Model saved to {model_filename}")

    print("\n--- Demonstrating Prediction with Logistic Regression Model ---")
    # Load one of the saved models for demonstration
    lr_model_loaded = joblib.load('logistic_regression_model.joblib')

    # Create a new patient profile for prediction
    new_patient_data = pd.DataFrame({
        'Age': [75],
        'Gender': ['Female'],
        'InsuranceType': ['Medicare'],
        'NumDiagnoses': [7],
        'NumProcedures': [3],
        'MedicationCount': [12],
        'LOS': [20],
        'Diabetes': [1],
        'HeartDisease': [1],
        'Cancer': [0]
    })

    prediction_proba = lr_model_loaded.predict_proba(new_patient_data)[:, 1]
    prediction_class = lr_model_loaded.predict(new_patient_data)[0]

    print(f"New Patient Data:\n{new_patient_data}")
    print(f"Predicted Readmission Probability (Logistic Regression): {prediction_proba[0]:.4f}")
    print(f"Predicted Readmission Class (0=No, 1=Yes): {prediction_class}")

    if hasattr(lr_model_loaded.named_steps['classifier'], 'coef_'):
        print("\n--- Feature Importance/Coefficients (Logistic Regression) ---")
        # Get feature names after one-hot encoding and scaling
        ohe_features = lr_model_loaded.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_features = list(numerical_features) + list(ohe_features)
        coefficients = lr_model_loaded.named_steps['classifier'].coef_[0]
        feature_importance = pd.Series(coefficients, index=all_features)
        print(feature_importance.sort_values(ascending=False))


if __name__ == "__main__":
    main()