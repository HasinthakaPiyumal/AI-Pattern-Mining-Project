"""data_loader.py: Module for generating and preparing synthetic patient data.
"""

def generate_synthetic_patient_data(num_patients=1000):
    """
    Generates synthetic patient data for demonstration purposes.
    Each patient has attributes like age, gender, pre_existing_condition, 
    and a binary outcome (e.g., disease).
    Returns a list of dictionaries, where each dictionary represents a patient.
    """
    data = []
    import random

    genders = ["male", "female"]
    conditions = ["none", "hypertension", "diabetes", "cardiovascular"]

    for i in range(num_patients):
        patient = {
            "patient_id": i,
            "age_group": random.choice(["young", "adult", "elderly"]),
            "gender": random.choice(genders),
            "pre_existing_condition": random.choice(conditions),
            "treatment_response": random.choice(["good", "poor"]),
            "model_prediction": random.choice([0, 1]), # 0: no disease, 1: disease
            "true_label": random.choice([0, 1])      # 0: no disease, 1: disease
        }
        # Introduce some correlation for demonstration
        if patient["age_group"] == "elderly" and patient["pre_existing_condition"] == "cardiovascular":
            if random.random() < 0.7: # Higher chance of disease in this subgroup
                patient["true_label"] = 1
            if random.random() < 0.6: # Model might perform worse here
                patient["model_prediction"] = 1 if random.random() < 0.3 else 0 # Introduce some errors
        
        # Introduce a scenario where model performance diverges
        if patient["gender"] == "female" and patient["age_group"] == "adult":
            if patient["true_label"] == 1 and random.random() < 0.2: # Higher False Negative Rate for this subgroup
                patient["model_prediction"] = 0
            elif patient["true_label"] == 0 and random.random() < 0.15: # Higher False Positive Rate
                 patient["model_prediction"] = 1

        data.append(patient)
    return data

def preprocess_data_for_fpm(data):
    """
    Converts raw patient data into a transaction-like format suitable for FPM.
    Each patient becomes a list of 'items' (attribute:value pairs).
    """
    preprocessed_data = []
    for patient in data:
        transaction = []
        for key, value in patient.items():
            if key not in ["patient_id", "model_prediction", "true_label"]:
                transaction.append(f"{key}:{value}")
        preprocessed_data.append(sorted(transaction)) # Sort for consistent itemset representation
    return preprocessed_data

def get_ground_truth_and_predictions(data):
    """
    Extracts true labels and model predictions from the raw data.
    """
    true_labels = [p["true_label"] for p in data]
    model_predictions = [p["model_prediction"] for p in data]
    return true_labels, model_predictions

