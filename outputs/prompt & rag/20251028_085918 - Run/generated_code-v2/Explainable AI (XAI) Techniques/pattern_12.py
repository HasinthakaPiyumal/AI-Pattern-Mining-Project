import random

def generate_patient_data(num_samples=1000):
    """
    Generates synthetic patient data for sepsis readmission prediction.
    Each patient is a dictionary.
    """
    data = []
    age_groups = ["young_adult", "adult", "elderly"]
    genders = ["male", "female"]
    comorbidity_groups = ["none", "low", "moderate", "high"]
    treatment_response_types = ["good", "average", "poor"]

    for i in range(num_samples):
        patient = {
            "patient_id": f"P{i+1}",
            "age_group": random.choice(age_groups),
            "gender": random.choice(genders),
            "comorbidity_group": random.choice(comorbidity_groups),
            "treatment_response_type": random.choice(treatment_response_types),
            "true_sepsis_readmission": 0 # Default
        }

        # Introduce some simple rules for true readmission for variety
        if patient["age_group"] == "elderly" and patient["comorbidity_group"] == "high":
            if random.random() < 0.6: # 60% chance of readmission for this group
                patient["true_sepsis_readmission"] = 1
        elif patient["treatment_response_type"] == "poor" and patient["gender"] == "female":
            if random.random() < 0.4:
                patient["true_sepsis_readmission"] = 1
        elif random.random() < 0.15: # Base readmission rate
            patient["true_sepsis_readmission"] = 1
        
        data.append(patient)
    return data

if __name__ == "__main__":
    sample_data = generate_patient_data(5)
    for patient in sample_data:
        print(patient)