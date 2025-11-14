from typing import Dict, List, Optional

class EHRSystem:
    """
    Simulates an Electronic Health Record (EHR) system.
    In a real system, this would connect to a database or EHR API.
    """
    def __init__(self):
        self.patient_records = {
            "P001": {
                "age": 45,
                "gender": "Male",
                "medical_history": ["Hypertension", "Type 2 Diabetes"],
                "lab_results": {"glucose": "high", "cholesterol": "elevated"},
                "imaging_reports": {},
                "ehr_status": "complete"
            },
            "P002": {
                "age": 30,
                "gender": "Female",
                "medical_history": ["Asthma"],
                "lab_results": {"CBC": "normal"},
                "imaging_reports": {},
                "ehr_status": "complete"
            },
            "P003": {
                "age": 60,
                "gender": "Male",
                "medical_history": ["Coronary Artery Disease"],
                "lab_results": {"ECG": "abnormal"},
                "imaging_reports": {"chest_xray": "clear"},
                "ehr_status": "complete"
            },
            "P004": {
                "age": 25,
                "gender": "Female",
                "medical_history": [],
                "lab_results": {},
                "imaging_reports": {},
                "ehr_status": "incomplete_ehr_data" # Simulate incomplete data
            }
        }

    def get_patient_ehr(self, patient_id: str) -> Optional[Dict]:
        """
        Retrieves simulated patient EHR data.
        """
        print(f"Retrieving EHR for patient: {patient_id}")
        return self.patient_records.get(patient_id)

class MedicalKnowledgeBase:
    """
    Simulates a medical knowledge base (e.g., PubMed, drug formularies).
    """
    def __init__(self):
        self.disease_info = {
            "Common Cold": {
                "symptoms": ["fever", "cough", "sore throat", "runny nose"],
                "treatments": ["symptomatic relief", "rest", "hydration"],
                "prevention": ["hand hygiene"],
                "evidence": ["WHO guidelines for common cold"]
            },
            "Influenza": {
                "symptoms": ["fever", "cough", "body aches", "fatigue", "sore throat"],
                "treatments": ["antivirals (if early)", "symptomatic relief", "rest"],
                "prevention": ["flu vaccine", "hand hygiene"],
                "evidence": ["CDC guidelines for influenza"]
            },
             "Hypertension": {
                "symptoms": ["headache", "chest pain", "vision changes"], # often asymptomatic
                "treatments": ["lifestyle changes", "medication (e.g., ACE inhibitors)"],
                "prevention": ["healthy diet", "exercise"],
                "evidence": ["AHA guidelines for hypertension"]
            }
        }
        self.drug_info = {
            "Acetaminophen": {
                "class": "Analgesic",
                "uses": ["pain relief", "fever reduction"],
                "side_effects": ["liver damage (high doses)"]
            },
            "Oseltamivir": {
                "class": "Antiviral",
                "uses": ["influenza treatment and prevention"],
                "side_effects": ["nausea", "vomiting"]
            },
            "Lisinopril": {
                "class": "ACE Inhibitor",
                "uses": ["hypertension", "heart failure"],
                "side_effects": ["cough", "dizziness"]
            }
        }

    def get_disease_details(self, disease_name: str) -> Optional[Dict]:
        """
        Retrieves simulated disease information.
        """
        print(f"Querying knowledge base for disease: {disease_name}")
        return self.disease_info.get(disease_name)

    def get_drug_details(self, drug_name: str) -> Optional[Dict]:
        """
        Retrieves simulated drug information.
        """
        print(f"Querying knowledge base for drug: {drug_name}")
        return self.drug_info.get(drug_name)

class ClinicalGuidelineEngine:
    """
    Simulates a clinical guideline engine.
    """
    def __init__(self):
        self.guidelines = {
            "Common Cold": {
                "diagnostic_pathway": ["clinical evaluation", "symptom assessment"],
                "treatment_protocol": ["rest", "fluids", "symptomatic medications"]
            },
            "Influenza": {
                "diagnostic_pathway": ["clinical evaluation", "rapid flu test (optional)"],
                "treatment_protocol": ["antivirals (within 48h of symptom onset)", "symptomatic care"]
            },
            "Hypertension": {
                "diagnostic_pathway": ["repeated blood pressure measurements", "risk factor assessment"],
                "treatment_protocol": ["lifestyle modifications", "pharmacotherapy based on stages"]
            }
        }

    def get_guidelines(self, condition: str) -> Optional[Dict]:
        """
        Retrieves simulated clinical guidelines for a given condition.
        """
        print(f"Consulting clinical guidelines for: {condition}")
        return self.guidelines.get(condition)
