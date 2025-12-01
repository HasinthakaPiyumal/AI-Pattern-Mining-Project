def diagnose_patient_with_confidence(symptoms: str) -> dict:
    symptoms = symptoms.lower()
    
    if "fever" in symptoms and "cough" in symptoms and "shortness of breath" in symptoms:
        diagnosis = "Pneumonia"
        confidence_score = 8
    elif "headache" in symptoms and "nausea" in symptoms and "sensitivity to light" in symptoms:
        diagnosis = "Migraine"
        confidence_score = 9
    elif "sore throat" in symptoms and "fatigue" in symptoms:
        diagnosis = "Common Cold"
        confidence_score = 7
    elif "abdominal pain" in symptoms and "vomiting" in symptoms:
        diagnosis = "Gastroenteritis"
        confidence_score = 6
    else:
        diagnosis = "Undetermined"
        confidence_score = 4

    return {"diagnosis": diagnosis, "confidence_score": confidence_score}

