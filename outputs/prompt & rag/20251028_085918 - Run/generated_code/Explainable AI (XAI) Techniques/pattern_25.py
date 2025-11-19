import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_synthetic_patient_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'Blood_Pressure_Systolic': np.random.randint(100, 180, num_samples),
        'Blood_Pressure_Diastolic': np.random.randint(60, 110, num_samples),
        'Cholesterol_LDL': np.random.randint(80, 200, num_samples),
        'Cholesterol_HDL': np.random.randint(30, 80, num_samples),
        'Kidney_Function_eGFR': np.random.randint(40, 120, num_samples),
        'Diabetes_History': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'Smoking_Status': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
        'Medication_Adherence': np.random.uniform(0.5, 1.0, num_samples),
        'Treatment_Outcome': np.random.choice([0, 1], num_samples, p=[0.5, 0.5]) # 0: Poor, 1: Good
    }
    df = pd.DataFrame(data)

    # Introduce some correlation for a more realistic outcome
    df['Treatment_Outcome'] = (
        (df['Age'] < 50) * 0.2 +
        (df['Blood_Pressure_Systolic'] < 140) * 0.3 +
        (df['Cholesterol_LDL'] < 130) * 0.2 +
        (df['Diabetes_History'] == 0) * 0.1 +
        (df['Smoking_Status'] == 0) * 0.1 +
        (df['Medication_Adherence'] > 0.8) * 0.1 +
        np.random.rand(num_samples) * 0.1 # Add some noise
    ) > 0.5
    df['Treatment_Outcome'] = df['Treatment_Outcome'].astype(int)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    return df

def preprocess_data(df):
    X = df.drop('Treatment_Outcome', axis=1)
    y = df['Treatment_Outcome']
    return X, y

def get_preprocessed_data(num_samples=1000):
    df = generate_synthetic_patient_data(num_samples)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X.columns.tolist()

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, feature_names = get_preprocessed_data()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Feature names: {feature_names}")
    print("Sample of X_train:")
    print(X_train.head())
    print("Sample of y_train:")
    print(y_train.head())
