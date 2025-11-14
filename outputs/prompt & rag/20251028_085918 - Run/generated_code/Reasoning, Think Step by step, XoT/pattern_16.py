from typing import List, Dict
from data_models import ClinicalDiagnosisOutput, DifferentialDiagnosis

class VerificationModule:
    def __init__(self):
        """
        Initializes the VerificationModule. This would typically load medical guidelines,
        drug interaction databases, and potentially a medical knowledge graph.
        """
        # Simulate a small medical knowledge base and guidelines
        self.medical_guidelines = {
            "Acute Myocardial Infarction": {
                "symptoms_match": ["chest pain", "shortness of breath"],
                "required_tests": ["ECG", "Troponin"],
                "common_treatments": ["Aspirin", "Nitroglycerin"]
            },
            "Pulmonary Embolism": {
                "symptoms_match": ["shortness of breath", "chest pain"],
                "required_tests": ["CT Angiogram", "D-dimer"],
                "common_treatments": ["Anticoagulation"]
            },
            "Streptococcal Pharyngitis": {
                "symptoms_match": ["sore throat", "fever"],
                "required_tests": ["Rapid Strep Test"],
                "common_treatments": ["Antibiotics"]
            }
        }
        self.drug_interactions_db = {
            "Aspirin": ["Warfarin", "Ibuprofen"],
            "Nitroglycerin": ["PDE5 inhibitors"], # e.g., Sildenafil
        }
        print("[INFO] Initialized VerificationModule with simulated medical data.")

    def _check_guideline_adherence(self, diagnosis: DifferentialDiagnosis, patient_symptoms: List[str]) -> bool:
        """
        Checks if a diagnosis adheres to simulated medical guidelines based on symptoms and suggested tests.
        """
        guideline = self.medical_guidelines.get(diagnosis.diagnosis_name)
        if not guideline:
            return False # No guideline found for this diagnosis

        # Check if key symptoms are broadly consistent
        symptom_match = all(s in [s.lower() for s in patient_symptoms] for s in guideline["symptoms_match"])
        
        # Check if suggested tests include required tests from guidelines
        suggested_tests_lower = [t.lower() for t in diagnosis.suggested_tests]
        required_tests_present = all(rt.lower() in suggested_tests_lower for rt in guideline["required_tests"])
        
        return symptom_match and required_tests_present

    def _check_drug_interactions(self, suggested_treatments: List[str]) -> bool:
        """
        Simulates checking for potential drug-drug interactions among suggested treatments.
        """
        for i, drug1 in enumerate(suggested_treatments):
            for drug2 in suggested_treatments[i+1:]:
                if drug2 in self.drug_interactions_db.get(drug1, []):
                    print(f"[WARNING] Potential drug interaction detected: {drug1} and {drug2}")
                    return False
                if drug1 in self.drug_interactions_db.get(drug2, []):
                    print(f"[WARNING] Potential drug interaction detected: {drug2} and {drug1}")
                    return False
        return True

    def verify_diagnosis_output(self, 
                                differential_diagnoses: List[DifferentialDiagnosis],
                                patient_symptoms: List[str],
                                proposed_final_diagnosis: Optional[DifferentialDiagnosis] = None) -> Dict[str, bool]:
        """
        Performs various verification checks on the LLM's diagnostic output.
        Returns a dictionary indicating the status of each verification check.
        """
        verification_results = {
            "guideline_adherence_check": True, # Assume true initially
            "drug_interaction_check": True, # Assume true initially
            "consistency_with_patient_data": True # Assume true initially
        }

        # 1. Guideline Adherence for top diagnoses
        if differential_diagnoses:
            top_diagnosis = differential_diagnoses[0] # Take the highest confidence one for this check
            if not self._check_guideline_adherence(top_diagnosis, patient_symptoms):
                verification_results["guideline_adherence_check"] = False
                print(f"[ALERT] Guideline non-adherence for {top_diagnosis.diagnosis_name}.")
        else:
            verification_results["guideline_adherence_check"] = False # No diagnoses to check


        # 2. Drug Interaction Check (if a final diagnosis and treatments are available)
        if proposed_final_diagnosis and proposed_final_diagnosis.potential_treatments:
            if not self._check_drug_interactions(proposed_final_diagnosis.potential_treatments):
                verification_results["drug_interaction_check"] = False

        # 3. Simple consistency check with patient data (e.g., if a symptom is explicitly contradicted)
        # This is a placeholder and would be more sophisticated in a real system.
        # For example, if 'fever' is a symptom but diagnosis explanation says 'afebrile'.
        # For now, we assume consistency unless explicit contradiction logic is added.
        
        # Additional checks could include:
        # - Cross-referencing with a medical knowledge graph for factual accuracy
        # - Checking for rare disease probabilities based on demographics

        return verification_results
