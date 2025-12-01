def load_simulated_healthcare_data():
    """
    Simulates loading a small healthcare dataset for demonstration.
    Each record includes patient attributes, a true outcome, and black-box model predictions.
    """
    data = [
        {"id": 1, "age_group": "30-40", "gender": "male", "diabetes": "yes", "hypertension": "no", "true_label": 0, "model_prediction": 0, "model_proba": 0.1},
        {"id": 2, "age_group": "50-60", "gender": "female", "diabetes": "yes", "hypertension": "yes", "true_label": 1, "model_prediction": 1, "model_proba": 0.9},
        {"id": 3, "age_group": "30-40", "gender": "male", "diabetes": "no", "hypertension": "no", "true_label": 0, "model_prediction": 1, "model_proba": 0.7}, # False Positive
        {"id": 4, "age_group": "70+", "gender": "male", "diabetes": "yes", "hypertension": "yes", "true_label": 1, "model_prediction": 0, "model_proba": 0.3}, # False Negative
        {"id": 5, "age_group": "50-60", "gender": "female", "diabetes": "no", "hypertension": "yes", "true_label": 0, "model_prediction": 0, "model_proba": 0.2},
        {"id": 6, "age_group": "30-40", "gender": "female", "diabetes": "yes", "hypertension": "no", "true_label": 1, "model_prediction": 1, "model_proba": 0.8},
        {"id": 7, "age_group": "70+", "gender": "male", "diabetes": "no", "hypertension": "no", "true_label": 0, "model_prediction": 0, "model_proba": 0.1},
        {"id": 8, "age_group": "30-40", "gender": "male", "diabetes": "yes", "hypertension": "no", "true_label": 0, "model_prediction": 1, "model_proba": 0.6}, # False Positive
        {"id": 9, "age_group": "50-60", "gender": "female", "diabetes": "yes", "hypertension": "yes", "true_label": 1, "model_prediction": 0, "model_proba": 0.4}, # False Negative
        {"id": 10, "age_group": "70+", "gender": "female", "diabetes": "no", "hypertension": "no", "true_label": 0, "model_prediction": 0, "model_proba": 0.1},
        {"id": 11, "age_group": "30-40", "gender": "male", "diabetes": "yes", "hypertension": "no", "true_label": 0, "model_prediction": 0, "model_proba": 0.15},
        {"id": 12, "age_group": "50-60", "gender": "female", "diabetes": "yes", "hypertension": "yes", "true_label": 1, "model_prediction": 1, "model_proba": 0.95},
        {"id": 13, "age_group": "30-40", "gender": "male", "diabetes": "no", "hypertension": "no", "true_label": 0, "model_prediction": 1, "model_proba": 0.75}, # False Positive
        {"id": 14, "age_group": "70+", "gender": "male", "diabetes": "yes", "hypertension": "yes", "true_label": 1, "model_prediction": 0, "model_proba": 0.35}, # False Negative
        {"id": 15, "age_group": "50-60", "gender": "female", "diabetes": "no", "hypertension": "yes", "true_label": 0, "model_prediction": 0, "model_proba": 0.25},
        {"id": 16, "age_group": "30-40", "gender": "female", "diabetes": "yes", "hypertension": "no", "true_label": 1, "model_prediction": 1, "model_proba": 0.85},
        {"id": 17, "age_group": "70+", "gender": "male", "diabetes": "no", "hypertension": "no", "true_label": 0, "model_prediction": 0, "model_proba": 0.15},
        {"id": 18, "age_group": "30-40", "gender": "male", "diabetes": "yes", "hypertension": "no", "true_label": 0, "model_prediction": 1, "model_proba": 0.65}, # False Positive
        {"id": 19, "age_group": "50-60", "gender": "female", "diabetes": "yes", "hypertension": "yes", "true_label": 1, "model_prediction": 0, "model_proba": 0.45}, # False Negative
        {"id": 20, "age_group": "70+", "gender": "female", "diabetes": "no", "hypertension": "no", "true_label": 0, "model_prediction": 0, "model_proba": 0.15}
    ]
    return data

def get_attributes(record, exclude_keys=None):
    """
    Extracts attribute-value pairs from a record, excluding specified keys.
    Returns a list of strings, e.g., ['age_group=30-40', 'gender=male'].
    """
    if exclude_keys is None:
        exclude_keys = ["id", "true_label", "model_prediction", "model_proba"]
    
    attributes = []
    for key, value in record.items():
        if key not in exclude_keys:
            attributes.append(f"{key}={value}")
    return attributes

