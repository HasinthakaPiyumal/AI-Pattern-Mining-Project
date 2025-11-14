
class PatientData:
    def __init__(self, symptoms: list[str], lab_results: dict, genomic_profile: dict):
        self.symptoms = [s.lower() for s in symptoms]
        self.lab_results = lab_results
        self.genomic_profile = genomic_profile

class MedicalKnowledgeBase:
    def __init__(self):
        self.diseases = {
            "cystic fibrosis": {
                "symptoms": ["persistent cough", "frequent lung infections", "poor weight gain", "salty skin"],
                "genomic_markers": ["cf-marker-1", "cf-marker-2"],
                "lab_anomalies": {"sweat chloride test": {"threshold": 60, "direction": "above"}},
                "treatment": "mucus thinners, antibiotics, nutritional support"
            },
            "huntington's disease": {
                "symptoms": ["involuntary movements", "cognitive decline", "mood swings"],
                "genomic_markers": ["hd-marker-1"],
                "lab_anomalies": {},
                "treatment": "medications for symptoms, supportive care"
            },
            "sickle cell anemia": {
                "symptoms": ["fatigue", "pain crises", "swelling in hands and feet", "frequent infections"],
                "genomic_markers": ["sca-marker-1"],
                "lab_anomalies": {"hemoglobin": {"threshold": 10, "direction": "below"}},
                "treatment": "pain management, blood transfusions, hydroxyurea"
            },
            "tay-sachs disease": {
                "symptoms": ["loss of motor skills", "exaggerated startle reflex", "cherry-red spot in eyes"],
                "genomic_markers": ["tsd-marker-1"],
                "lab_anomalies": {"hexosaminidase a activity": {"threshold": 50, "direction": "below"}},
                "treatment": "supportive care, no cure"
            },
            "lupus": {
                "symptoms": ["fatigue", "joint pain", "skin rashes", "fever"],
                "genomic_markers": [],
                "lab_anomalies": {"ana test": {"value": "positive", "direction": "is"}},
                "treatment": "immunosuppressants, anti-inflammatories"
            }
        }

    def get_matching_diseases_by_symptoms(self, patient_symptoms: list[str]) -> list[str]:
        potential_diseases = []
        for disease, data in self.diseases.items():
            if any(symptom in data["symptoms"] for symptom in patient_symptoms):
                potential_diseases.append(disease)
        return potential_diseases

    def get_matching_diseases_by_genomic_markers(self, genomic_markers: list[str]) -> list[str]:
        potential_diseases = []
        for disease, data in self.diseases.items():
            if any(marker in data["genomic_markers"] for marker in genomic_markers):
                potential_diseases.append(disease)
        return potential_diseases

    def get_disease_details(self, disease_name: str) -> dict:
        return self.diseases.get(disease_name.lower(), {})

class LabResultsInterpreter:
    def interpret_results(self, lab_data: dict, disease_knowledge: dict) -> list[str]:
        anomalies = []
        for lab_test, expected_values in disease_knowledge.get("lab_anomalies", {}).items():
            patient_value = lab_data.get(lab_test.lower())
            if patient_value is not None:
                threshold = expected_values.get("threshold")
                direction = expected_values.get("direction")
                expected_value_str = expected_values.get("value") # For string comparisons like 