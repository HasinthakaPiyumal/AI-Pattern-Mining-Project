import pandas as pd
import numpy as np

def generate_synthetic_ehr_data(num_samples=1000, random_state=42):
    """
    Generates a synthetic dataset mimicking Electronic Health Record (EHR) data
    for patient readmission prediction.
    """
    np.random.seed(random_state)

    data = {
        'patient_id': np.arange(1, num_samples + 1),
        'age': np.random.randint(20, 90, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples, p=[0.48, 0.52]),
        'ethnicity': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], num_samples, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
        'admission_type': np.random.choice(['Emergency', 'Elective', 'Urgent'], num_samples, p=[0.5, 0.3, 0.2]),
        'major_diagnosis': np.random.choice([
            'Heart Failure', 'Pneumonia', 'Diabetes Complications', 'COPD Exacerbation',
            'Stroke', 'Sepsis', 'Kidney Disease', 'Hypertension Crisis'
        ], num_samples, p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1]),
        'num_diagnoses': np.random.randint(1, 10, num_samples),
        'num_procedures': np.random.randint(0, 5, num_samples),
        'num_medications': np.random.randint(1, 15, num_samples),
        'lab_result_creatinine': np.random.normal(1.2, 0.5, num_samples).clip(0.5, 5.0).round(2), # Simulating kidney function
        'lab_result_hemoglobin': np.random.normal(13.5, 1.5, num_samples).clip(8.0, 18.0).round(2), # Simulating anemia
        'length_of_stay': np.random.randint(1, 30, num_samples),
        'prior_readmissions': np.random.randint(0, 3, num_samples),
        'readmitted_30_days': np.random.choice([0, 1], num_samples, p=[0.75, 0.25]) # 0: Not readmitted, 1: Readmitted
    }

    df = pd.DataFrame(data)

    # Introduce some correlations to make the data more realistic
    # For example, older patients with more diagnoses might have higher readmission rates
    df['readmitted_30_days'] = df.apply(
        lambda row: 1 if (
            (row['age'] > 65 and row['num_diagnoses'] > 4 and np.random.rand() < 0.4) or 
            (row['major_diagnosis'] == 'Heart Failure' and np.random.rand() < 0.45) or
            (row['prior_readmissions'] > 0 and np.random.rand() < 0.5)
        ) else row['readmitted_30_days'], axis=1
    )
    # Ensure target balance is somewhat maintained but influenced by factors
    df['readmitted_30_days'] = df['readmitted_30_days'].astype(int)

    return df

if __name__ == '__main__':
    ehr_data = generate_synthetic_ehr_data(num_samples=10000)
    print(f"Generated {len(ehr_data)} synthetic EHR records.")
    print(ehr_data.head())
    print("\nReadmission rate:", ehr_data['readmitted_30_days'].mean().round(3))
    ehr_data.to_csv("synthetic_ehr_data.csv", index=False)
    print("Data saved to synthetic_ehr_data.csv")
