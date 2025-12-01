import json

def mock_ehr_api(patient_id: str):
    if patient_id == "P123":
        return {"patient_id": "P123", "name": "John Doe", "age": 65, "conditions": ["Hypertension", "Type 2 Diabetes"], "medications": ["Lisinopril", "Metformin"]}
    return {"error": "Patient not found"}

def mock_medical_kb_api(query: str):
    if "hypertension treatment" in query.lower():
        return {"guideline": "Maintain blood pressure below 130/80 mmHg. Lifestyle modifications, ACE inhibitors, diuretics.", "source": "AHA/ACC Guidelines"}
    if "diabetes management" in query.lower():
        return {"guideline": "Monitor blood glucose, diet control, exercise, Metformin as first-line therapy, consider SGLT2 inhibitors or GLP-1 receptor agonists.", "source": "ADA Guidelines"}
    return {"guideline": "No specific guideline found for this query.", "source": "N/A"}

def mock_diagnostic_imaging_analysis(image_data_identifier: str):
    if image_data_identifier == "XRAY_LUNG_P123":
        return {"findings": "Mild cardiomegaly, clear lung fields.", "impression": "No acute cardiopulmonary process.", "timestamp": "2023-10-27T10:00:00Z"}
    return {"error": "Image data not found or analysis failed"}

def mock_drug_interaction_api(drugs: list):
    if "Lisinopril" in drugs and "Metformin" in drugs:
        return {"interactions": [], "severity": "None", "details": "No significant interactions between Lisinopril and Metformin."}
    if "Warfarin" in drugs and "Aspirin" in drugs:
        return {"interactions": [{"drug1": "Warfarin", "drug2": "Aspirin", "effect": "Increased bleeding risk"}], "severity": "High", "details": "Concurrent use increases the risk of bleeding."}
    return {"interactions": [], "severity": "None", "details": "No known interactions for the provided drugs."}

def foundation_model_controller(doctor_query: str):
    response = {"diagnosis_support": "", "treatment_recommendations": [], "raw_tool_outputs": {}}

    if "patient details" in doctor_query.lower() or "patient info" in doctor_query.lower():
        patient_id = "P123" # In a real system, extract from query or context
        ehr_data = mock_ehr_api(patient_id)
        response["raw_tool_outputs"]["ehr_data"] = ehr_data
        if "name" in ehr_data:
            response["diagnosis_support"] += f"Patient {ehr_data['name']} (ID: {ehr_data['patient_id']}, Age: {ehr_data['age']}) has existing conditions: {', '.join(ehr_data['conditions'])} and is on medications: {', '.join(ehr_data['medications'])}. "

    if "medical guidelines" in doctor_query.lower() or "treatment for hypertension" in doctor_query.lower():
        kb_query = "hypertension treatment" # Example query, would be more dynamic
        kb_data = mock_medical_kb_api(kb_query)
        response["raw_tool_outputs"]["medical_knowledge_base"] = kb_data
        response["treatment_recommendations"].append(f"Based on medical guidelines ({kb_data['source']}): {kb_data['guideline']}")

    if "analyze X-ray" in doctor_query.lower() or "imaging results" in doctor_query.lower():
        image_id = "XRAY_LUNG_P123" # Example image identifier
        imaging_data = mock_diagnostic_imaging_analysis(image_id)
        response["raw_tool_outputs"]["diagnostic_imaging"] = imaging_data
        if "findings" in imaging_data:
            response["diagnosis_support"] += f"Diagnostic Imaging Analysis (X-ray {image_id}): Findings - {imaging_data['findings']}. Impression - {imaging_data['impression']}. "

    if "drug interactions" in doctor_query.lower() or "check medications" in doctor_query.lower():
        # Assuming patient P123's medications are relevant for interaction check
        patient_drugs = mock_ehr_api("P123").get("medications", [])
        interaction_data = mock_drug_interaction_api(patient_drugs)
        response["raw_tool_outputs"]["drug_interactions"] = interaction_data
        if interaction_data["interactions"]:
            response["diagnosis_support"] += f"Warning: Potential drug interactions detected with severity {interaction_data['severity']}. Details: {interaction_data['details']}. "
        else:
            response["diagnosis_support"] += f"No significant drug interactions found for current medications. {interaction_data['details']}. "

    if not response["diagnosis_support"] and not response["treatment_recommendations"]:
        response["diagnosis_support"] = "Could not provide specific support based on the query. Please refine your request."

    return response

if __name__ == "__main__":
    print("\n--- Query 1: Get patient details and general hypertension treatment ---")
    query1 = "Please provide patient details for P123 and general treatment guidelines for hypertension."
    result1 = foundation_model_controller(query1)
    print(json.dumps(result1, indent=2))

    print("\n--- Query 2: Analyze X-ray and check for drug interactions ---")
    query2 = "I need to analyze the X-ray for P123 and check for any drug interactions for his current medications."
    result2 = foundation_model_controller(query2)
    print(json.dumps(result2, indent=2))

    print("\n--- Query 3: General query with no specific tool match ---")
    query3 = "What is the best way to live a healthy life?"
    result3 = foundation_model_controller(query3)
    print(json.dumps(result3, indent=2))
