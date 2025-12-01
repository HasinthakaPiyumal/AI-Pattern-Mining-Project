from pydantic import BaseModel
from typing import List, Dict

# --- data_collector.py ---

class PatientDataModel(BaseModel):
    ehr_data: str
    lab_results: Dict[str, str]
    symptoms: List[str]

class ExtractedMedicalInfoModel(BaseModel):
    patient_id: str
    key_symptoms: List[str]
    relevant_lab_values: Dict[str, str]
    conditions: List[str]
    medications: List[str]
    allergies: List[str]

def collect_and_extract_data(raw_patient_data: PatientDataModel) -> ExtractedMedicalInfoModel:
    patient_id = "PAT_001"
    
    # Simulate extracting key symptoms from raw text/list
    key_symptoms = [symptom.lower() for symptom in raw_patient_data.symptoms if symptom in ["fever", "cough", "sore throat", "fatigue"]]
    
    # Simulate extracting relevant lab values
    relevant_lab_values = {
        k: v for k, v in raw_patient_data.lab_results.items() if k in ["CRP", "WBC", "viral_load"]
    }
    
    # Simulate extracting existing conditions and medications from EHR data
    conditions = []
    if "diabetes" in raw_patient_data.ehr_data.lower():
        conditions.append("Diabetes")
    if "hypertension" in raw_patient_data.ehr_data.lower():
        conditions.append("Hypertension")
        
    medications = []
    if "metformin" in raw_patient_data.ehr_data.lower():
        medications.append("Metformin")
    if "lisinopril" in raw_patient_data.ehr_data.lower():
        medications.append("Lisinopril")
        
    allergies = []
    if "penicillin allergy" in raw_patient_data.ehr_data.lower():
        allergies.append("Penicillin")

    return ExtractedMedicalInfoModel(
        patient_id=patient_id,
        key_symptoms=key_symptoms,
        relevant_lab_values=relevant_lab_values,
        conditions=conditions,
        medications=medications,
        allergies=allergies
    )

# --- diagnoser_planner.py ---

class DiagnosisOutputModel(BaseModel):
    differential_diagnoses: List[str]
    probabilities: Dict[str, float]
    recommended_tests: List[str]
    treatment_plan: List[str]
    constraints: List[str]

def formulate_diagnosis_and_plan(extracted_info: ExtractedMedicalInfoModel) -> DiagnosisOutputModel:
    differential_diagnoses = []
    probabilities = {}
    recommended_tests = []
    treatment_plan = []
    constraints = []

    # Rule-based diagnosis simulation
    if "fever" in extracted_info.key_symptoms and "cough" in extracted_info.key_symptoms:
        if extracted_info.relevant_lab_values.get("viral_load") == "high":
            differential_diagnoses.append("Influenza (Flu)")
            probabilities["Influenza (Flu)"] = 0.8
            recommended_tests.append("Rapid Flu Test")
            treatment_plan.append("Antiviral medication (e.g., Oseltamivir)")
        else:
            differential_diagnoses.append("Common Cold")
            probabilities["Common Cold"] = 0.7
            treatment_plan.append("Symptomatic relief (rest, fluids)")
    elif "sore throat" in extracted_info.key_symptoms:
        differential_diagnoses.append("Streptococcal Pharyngitis")
        probabilities["Streptococcal Pharyngitis"] = 0.6
        recommended_tests.append("Throat Swab Culture")
        treatment_plan.append("Antibiotics (e.g., Amoxicillin)")
    
    if not differential_diagnoses:
        differential_diagnoses.append("Undetermined viral infection")
        probabilities["Undetermined viral infection"] = 1.0
        treatment_plan.append("Symptomatic relief")

    # Mock constraint checking
    if "Penicillin" in extracted_info.allergies and "Amoxicillin" in treatment_plan:
        constraints.append("Patient has Penicillin allergy; avoid Amoxicillin. Consider alternative antibiotics.")
        treatment_plan.remove("Amoxicillin")
        treatment_plan.append("Antibiotics (e.g., Azithromycin)")

    return DiagnosisOutputModel(
        differential_diagnoses=differential_diagnoses,
        probabilities=probabilities,
        recommended_tests=recommended_tests,
        treatment_plan=treatment_plan,
        constraints=constraints
    )

# --- medical_assistant.py ---

class MedicalDiagnosticAssistant:
    def diagnose_patient(self, raw_data: Dict) -> DiagnosisOutputModel:
        patient_data = PatientDataModel(**raw_data)
        
        extracted_info = collect_and_extract_data(patient_data)
        
        diagnosis_output = formulate_diagnosis_and_plan(extracted_info)
        
        return diagnosis_output

if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()

    # Example raw patient data
    raw_patient_data_1 = {
        "ehr_data": "Patient has a history of penicillin allergy. No other significant medical history. Currently on no medications.",
        "lab_results": {"CRP": "normal", "WBC": "elevated", "viral_load": "high"},
        "symptoms": ["fever", "cough", "fatigue"]
    }

    raw_patient_data_2 = {
        "ehr_data": "Patient has mild hypertension, managed with Lisinopril. No known allergies.",
        "lab_results": {"CRP": "slightly elevated"},
        "symptoms": ["sore throat", "mild fever"]
    }

    print("\n--- Diagnosing Patient 1 ---")
    diagnosis_1 = assistant.diagnose_patient(raw_patient_data_1)
    print(diagnosis_1.model_dump_json(indent=2))

    print("\n--- Diagnosing Patient 2 ---")
    diagnosis_2 = assistant.diagnose_patient(raw_patient_data_2)
    print(diagnosis_2.model_dump_json(indent=2))
