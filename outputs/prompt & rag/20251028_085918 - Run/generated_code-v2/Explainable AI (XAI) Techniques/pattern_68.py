import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

def generate_simulated_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 90, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'Diagnosis': np.random.choice(['Heart Disease', 'Diabetes', 'Pneumonia', 'Asthma', 'Cancer'], num_samples),
        'Num_Admissions': np.random.randint(1, 10, num_samples),
        'Length_of_Stay_Days': np.random.randint(1, 30, num_samples),
        'Medication_Count': np.random.randint(1, 15, num_samples),
        'Readmitted': np.random.randint(0, 2, num_samples) # 0 for No, 1 for Yes
    }
    df = pd.DataFrame(data)

    # Introduce some correlation for 'Readmitted'
    df.loc[df['Age'] > 70, 'Readmitted'] = np.random.choice([0, 1], sum(df['Age'] > 70), p=[0.4, 0.6])
    df.loc[df['Diagnosis'] == 'Heart Disease', 'Readmitted'] = np.random.choice([0, 1], sum(df['Diagnosis'] == 'Heart Disease'), p=[0.5, 0.5])
    df.loc[df['Num_Admissions'] > 5, 'Readmitted'] = np.random.choice([0, 1], sum(df['Num_Admissions'] > 5), p=[0.3, 0.7])

    return df

def preprocess_data(df):
    X = df.drop('Readmitted', axis=1)
    y = df['Readmitted']

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return preprocessor, X_train, X_test, y_train, y_test

def train_model(preprocessor, X_train, y_train):
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42))
    ])

    model_pipeline.fit(X_train, y_train)
    return model_pipeline

def evaluate_model(model_pipeline, X_test, y_test):
    y_pred = model_pipeline.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def save_artifacts(model_pipeline, preprocessor, model_path='readmission_model.joblib', preprocessor_path='readmission_preprocessor.joblib'):
    joblib.dump(model_pipeline, model_path)
    # The preprocessor is part of the pipeline, so saving the pipeline is sufficient.
    # However, if we needed the preprocessor separately for other tasks, we would save it like this:
    # joblib.dump(preprocessor, preprocessor_path)
    print(f"Model saved to {model_path}")

def load_artifacts(model_path='readmission_model.joblib'):
    model_pipeline = joblib.load(model_path)
    return model_pipeline

def predict_and_explain(model_pipeline, new_patient_data):
    # Convert new_patient_data to a DataFrame, ensuring all original columns are present
    # and in the correct order for the preprocessor in the pipeline
    # We need to reconstruct the expected columns for the preprocessor
    # This part assumes a specific structure of how the pipeline handles columns.
    # For robust production, ensure new_patient_data always matches training features.

    # Create a DataFrame from the new patient data
    new_df = pd.DataFrame([new_patient_data])

    # Get feature names after one-hot encoding from the preprocessor within the pipeline
    # This is a bit tricky as OHE feature names are dynamic.
    # A simpler approach for Logistic Regression's interpretability is to look at coefficients
    # after the model is trained and its features are transformed.
    # Let's assume the preprocessor inside the pipeline correctly handles new data.

    prediction_proba = model_pipeline.predict_proba(new_df)[0]
    prediction = model_pipeline.predict(new_df)[0]

    readmission_risk = prediction_proba[1] * 100 # Probability of readmission (class 1)

    explanation = f"Based on the patient's data, the model predicts a {readmission_risk:.2f}% risk of readmission. "

    # Extracting coefficients for explanation (Logistic Regression)
    if isinstance(model_pipeline.named_steps['classifier'], LogisticRegression):
        logistic_model = model_pipeline.named_steps['classifier']
        feature_names_out = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
        coefficients = pd.Series(logistic_model.coef_[0], index=feature_names_out)

        # Explain based on the most influential positive and negative coefficients
        top_positive_coeffs = coefficients.nlargest(3)
        top_negative_coeffs = coefficients.nsmallest(3)

        explanation += "\n\nFactors increasing readmission risk (higher values contribute more):\n"
        for feature, coef in top_positive_coeffs.items():
            original_feature = feature.split('__')[-1] # Clean up feature names from OHE
            explanation += f"- {original_feature.replace('cat__', '')} (Impact: {coef:.4f})\n"

        explanation += "\nFactors decreasing readmission risk (higher values contribute less):\n"
        for feature, coef in top_negative_coeffs.items():
            original_feature = feature.split('__')[-1]
            explanation += f"- {original_feature.replace('cat__', '')} (Impact: {coef:.4f})\n"

    return "Yes" if prediction == 1 else "No", explanation

if __name__ == "__main__":
    print("--- Generating Simulated Patient Data ---")
    df = generate_simulated_data(num_samples=1000)
    print("Data Generated (first 5 rows):")
    print(df.head())

    print("\n--- Preprocessing Data ---")
    preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    print("\n--- Training Logistic Regression Model ---")
    model_pipeline = train_model(preprocessor, X_train, y_train)
    print("Model Training Complete.")

    print("\n--- Evaluating Model Performance ---")
    evaluate_model(model_pipeline, X_test, y_test)

    print("\n--- Saving Model and Preprocessor ---")
    save_artifacts(model_pipeline, preprocessor)

    print("\n--- Loading Model for Prediction Service ---")
    loaded_model = load_artifacts()
    print("Model Loaded.")

    print("\n--- Performing Sample Prediction and Explanation ---")
    # Example new patient data
    new_patient = {
        'Age': 75,
        'Gender': 'Female',
        'Diagnosis': 'Heart Disease',
        'Num_Admissions': 8,
        'Length_of_Stay_Days': 20,
        'Medication_Count': 10
    }

    prediction, explanation = predict_and_explain(loaded_model, new_patient)

    print(f"\nNew Patient Data: {new_patient}")
    print(f"Predicted Readmission: {prediction}")
    print(f"Explanation: {explanation}")

    print("\n--- Another Sample Prediction ---")
    new_patient_2 = {
        'Age': 30,
        'Gender': 'Male',
        'Diagnosis': 'Asthma',
        'Num_Admissions': 1,
        'Length_of_Stay_Days': 3,
        'Medication_Count': 2
    }

    prediction_2, explanation_2 = predict_and_explain(loaded_model, new_patient_2)

    print(f"\nNew Patient Data: {new_patient_2}")
    print(f"Predicted Readmission: {prediction_2}")
    print(f"Explanation: {explanation_2}")
