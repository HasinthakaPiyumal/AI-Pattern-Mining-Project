import json

def query_medical_database(disease: str) -> str:
    """
    Queries a simulated medical database for information about a specific disease.
    Returns a JSON string with disease details or a 'not found' message.
    """
    medical_data = {
        "common cold": {
            "symptoms": ["runny nose", "sore throat", "cough", "sneezing"],
            "causes": ["rhinoviruses"],
            "description": "A viral infection of the nose and throat."
        },
        "influenza": {
            "symptoms": ["fever", "body aches", "fatigue", "cough", "sore throat"],
            "causes": ["influenza viruses"],
            "description": "A common viral infection that can be deadly."
        },
        "diabetes type 2": {
            "symptoms": ["increased thirst", "frequent urination", "increased hunger", "fatigue", "blurred vision"],
            "causes": ["insulin resistance", "genetics", "lifestyle"],
            "description": "A chronic condition that affects the way the body processes blood sugar."
        }
    }
    
    if disease.lower() in medical_data:
        return json.dumps(medical_data[disease.lower()])
    else:
        return json.dumps({"error": f"Information for '{disease}' not found in the database."})

def get_treatment_protocol(disease: str, patient_conditions: list) -> str:
    """
    Retrieves a simulated treatment protocol based on the disease and patient-specific conditions.
    Returns a JSON string with recommended treatments or a 'no specific protocol' message.
    """
    treatment_protocols = {
        "common cold": {
            "general": ["rest", "fluids", "over-the-counter pain relievers"],
            "severe_symptoms": ["consult doctor for antiviral options"], # Example of condition-based advice
            "considerations": ["avoid antibiotics as it's viral"]
        },
        "influenza": {
            "general": ["antiviral medications (e.g., oseltamivir)", "rest", "fluids"],
            "high_risk_patient": ["hospitalization", "close monitoring"],
            "considerations": ["vaccination is key for prevention"]
        },
        "diabetes type 2": {
            "general": ["dietary changes", "exercise", "blood sugar monitoring"],
            "medication_needed": ["metformin", "insulin therapy"], # Example of condition-based advice
            "complications": ["referral to specialist for neuropathy/nephropathy"]
        }
    }

    protocol = treatment_protocols.get(disease.lower())
    if protocol:
        recommendations = protocol.get("general", [])
        if "severe_symptoms" in patient_conditions and disease.lower() == "common cold":
            recommendations.extend(protocol.get("severe_symptoms", []))
        if "high_risk_patient" in patient_conditions and disease.lower() == "influenza":
            recommendations.extend(protocol.get("high_risk_patient", []))
        if "medication_needed" in patient_conditions and disease.lower() == "diabetes type 2":
            recommendations.extend(protocol.get("medication_needed", []))
        
        return json.dumps({"disease": disease, "treatment_recommendations": recommendations, "considerations": protocol.get("considerations", [])})
    else:
        return json.dumps({"error": f"No specific treatment protocol found for '{disease}'."})

def log_diagnosis_and_treatment(patient_id: str, diagnosis: str, treatment: str) -> str:
    """
    Simulates logging a diagnosis and treatment for a patient.
    In a real system, this would store data for feedback and learning.
    """
    print(f"[LOG] Patient ID: {patient_id}, Diagnosis: {diagnosis}, Treatment: {treatment}")
    # In a real system, this would write to a database or a feedback queue
    return json.dumps({"status": "success", "message": "Diagnosis and treatment logged for feedback."})
