"""This module defines mock external tools for the Smart Medical Assistant.
In a real-world scenario, these would integrate with actual APIs and services.
"""

def get_patient_ehr(patient_id: str) -> str:
    """Mocks fetching Electronic Health Record (EHR) data for a given patient ID.
    In a real application, this would call an EHR system API.
    """
    print(f"Fetching EHR for patient: {patient_id}")
    if patient_id == "P1001":
        return "EHR for P1001: Age 65, Male, Diabetes Type 2, Hypertension. Medications: Metformin, Lisinopril. Last visit: 2023-10-26."
    elif patient_id == "P1002":
        return "EHR for P1002: Age 42, Female, Asthma. Medications: Albuterol. Last visit: 2023-11-15."
    else:
        return f"No EHR found for patient ID: {patient_id}"

def check_drug_interaction(drug1: str, drug2: str) -> str:
    """Mocks checking for potential drug interactions between two drugs.
    In a real application, this would call a Drug Interaction Database API.
    """
    print(f"Checking drug interaction between {drug1} and {drug2}")
    drug1_lower = drug1.lower()
    drug2_lower = drug2.lower()

    if ("metformin" in drug1_lower and "lisinopril" in drug2_lower) or \
       ("lisinopril" in drug1_lower and "metformin" in drug2_lower):
        return "Potential for increased risk of kidney issues. Monitor renal function."
    elif ("albuterol" in drug1_lower and "propranolol" in drug2_lower) or \
         ("propranolol" in drug1_lower and "albuterol" in drug2_lower):
        return "Beta-blockers (like Propranolol) can reduce the effectiveness of Albuterol. Use with caution."
    else:
        return f"No significant interaction found between {drug1} and {drug2}."

def perform_medical_calculation(formula: str, values: dict) -> str:
    """Mocks performing a medical calculation based on a formula and input values.
    This is a simplified example; a real calculator would handle complex formulas.
    """
    print(f"Performing calculation: {formula} with values {values}")
    try:
        if formula.lower() == "bmi":
            weight_kg = values.get("weight_kg")
            height_m = values.get("height_m")
            if weight_kg is not None and height_m is not None and height_m > 0:
                bmi = weight_kg / (height_m ** 2)
                return f"BMI calculated: {bmi:.2f}"
            else:
                return "Invalid input for BMI calculation. Requires 'weight_kg' and 'height_m' > 0."
        elif formula.lower() == "creatinine_clearance": # Simplified example
            creatinine = values.get("creatinine_mg_dl")
            age = values.get("age_years")
            if creatinine is not None and age is not None:
                # Very simplified, just for demonstration
                clearance = (140 - age) / creatinine
                return f"Estimated Creatinine Clearance: {clearance:.2f} mL/min (simplified calculation)"
            else:
                return "Invalid input for Creatinine Clearance. Requires 'creatinine_mg_dl' and 'age_years'."
        else:
            return f"Unknown medical calculation formula: {formula}"
    except Exception as e:
        return f"Error during calculation: {e}"

def search_medical_knowledge_base(query: str) -> str:
    """Mocks searching a medical knowledge base for information.
    In a real application, this would query a service like PubMed or a proprietary database.
    """
    print(f"Searching medical knowledge base for: {query}")
    query_lower = query.lower()
    if "diabetes type 2 treatment" in query_lower:
        return "According to recent guidelines, first-line treatment for Diabetes Type 2 often involves lifestyle modifications (diet, exercise) and Metformin. SGLT2 inhibitors or GLP-1 receptor agonists may be added depending on cardiovascular or renal risk factors. (Source: Mock Medical Journal, 2023)"
    elif "asthma exacerbation management" in query_lower:
        return "Acute asthma exacerbations typically involve inhaled short-acting beta-agonists (SABAs) like Albuterol. Oral corticosteroids may be necessary for moderate to severe exacerbations. (Source: Mock Pulmonology Review, 2022)"
    elif "hypertension guidelines" in query_lower:
        return "Current hypertension guidelines recommend a target blood pressure of <130/80 mmHg for most adults. Lifestyle modifications are crucial. Pharmacological treatment often starts with ACE inhibitors, ARBs, Thiazide diuretics, or Calcium Channel Blockers. (Source: Mock Cardiology Association, 2021)"
    else:
        return f"No direct information found for '{query}' in the mock knowledge base. Please refine your query."
