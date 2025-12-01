import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples=1000):
    """
    Generates synthetic patient data for cardiovascular disease prediction.
    Features include age, cholesterol, blood pressure, BMI, smoking status, exercise, stress.
    The target variable 'CVD_Risk' is generated with some correlation to features.
    """
    np.random.seed(42)

    data = {
        'Age': np.random.randint(30, 80, num_samples),
        'Cholesterol': np.random.normal(200, 30, num_samples).astype(int),
        'BloodPressure_Systolic': np.random.normal(120, 15, num_samples).astype(int),
        'BloodPressure_Diastolic': np.random.normal(80, 10, num_samples).astype(int),
        'BMI': np.random.normal(27, 5, num_samples),
        'Smoking': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]), # 0: No, 1: Yes
        'Exercise_Hours_Week': np.random.normal(3, 2, num_samples).clip(0, 10), # Hours per week
        'Stress_Level': np.random.randint(1, 10, num_samples), # 1: Low, 10: High
    }

    df = pd.DataFrame(data)

    # Generate CVD_Risk with some logical correlation
    # Higher age, cholesterol, BP, BMI, smoking, stress, lower exercise -> higher risk
    cvd_risk_base = (
        0.02 * df['Age'] +
        0.01 * df['Cholesterol'] +
        0.03 * df['BloodPressure_Systolic'] +
        0.02 * df['BMI'] +
        0.5 * df['Smoking'] +
        -0.1 * df['Exercise_Hours_Week'] +
        0.05 * df['Stress_Level']
    )

    # Scale to a probability and add noise
    cvd_risk_prob = 1 / (1 + np.exp(-(cvd_risk_base - np.mean(cvd_risk_base)) / np.std(cvd_risk_base) + np.random.normal(0, 0.8, num_samples)))
    df['CVD_Risk'] = (cvd_risk_prob > 0.5).astype(int) # Binary classification for simplicity

    return df

if __name__ == "__main__":
    synthetic_data = generate_synthetic_data(num_samples=1000)
    print(f"Generated data shape: {synthetic_data.shape}")
    print(f"First 5 rows:\n{synthetic_data.head()}")
    print(f"CVD Risk distribution:\n{synthetic_data['CVD_Risk'].value_counts()}")
    synthetic_data.to_csv("cardiac_patient_data.csv", index=False)
    print("Synthetic data saved to cardiac_patient_data.csv")
