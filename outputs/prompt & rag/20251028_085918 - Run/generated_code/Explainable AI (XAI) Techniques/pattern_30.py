import pandas as pd
import numpy as np

def generate_cardio_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)

    data = {
        'Age': np.random.randint(25, 80, n_samples),
        'Gender': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
        'Cholesterol': np.random.randint(150, 300, n_samples),
        'BloodPressure_Systolic': np.random.randint(100, 180, n_samples),
        'BloodPressure_Diastolic': np.random.randint(60, 120, n_samples),
        'Glucose': np.random.randint(70, 200, n_samples),
        'BMI': np.random.uniform(18.0, 40.0, n_samples),
        'Smoker': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'FamilyHistory': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    }

    df = pd.DataFrame(data)

    # Generate a target variable (CardioDisease) based on some plausible relationships
    # Simplified linear combination with noise and sigmoid for binary output
    disease_risk = (
        0.05 * df['Age']
        + 0.8 * df['Gender']  # Males might have higher risk in this synthetic data
        + 0.02 * df['Cholesterol']
        + 0.03 * df['BloodPressure_Systolic']
        - 0.01 * df['BMI']  # Lower BMI might indicate less risk, but not always
        + 0.5 * df['Smoker']
        + 0.7 * df['FamilyHistory']
        + np.random.normal(0, 1.5, n_samples)
    )

    # Apply a sigmoid-like transformation to get probabilities and then convert to binary
    probability = 1 / (1 + np.exp(-(disease_risk - np.mean(disease_risk))))
    df['CardioDisease'] = (probability > 0.5).astype(int)

    return df

if __name__ == "__main__":
    df = generate_cardio_data(n_samples=2000)
    df.to_csv("cardio_data.csv", index=False)
    print("Generated cardio_data.csv with 2000 samples.")