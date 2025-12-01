import pandas as pd
import random

class MedicalDiagnosisAssistant:
    def __init__(self):
        self.known_symptoms = {
            "fever": ["influenza", "common cold", "bacterial infection"],
            "cough": ["influenza", "common cold", "bronchitis"],
            "headache": ["migraine", "tension headache", "influenza"],
            "sore throat": ["common cold", "strep throat"],
            "fatigue": ["influenza", "mononucleosis", "anemia"],
            "chest pain": ["heart attack", "angina", "anxiety"],
            "shortness of breath": ["asthma", "pneumonia", "heart failure"]
        }
        self.medical_literature = {
            "influenza": "Reference: CDC guidelines on influenza. Details: Viral infection with fever, cough, fatigue.",
            "common cold": "Reference: Mayo Clinic info on common cold. Details: Milder viral infection, no specific treatment.",
            "bacterial infection": "Reference: WHO guidelines on bacterial infections. Details: Often requires antibiotics.",
            "migraine": "Reference: National Headache Foundation. Details: Severe headache, often with aura.",
            "heart attack": "Reference: American Heart Association. Details: Medical emergency, chest pain, shortness of breath.",
            "strep throat": "Reference: NIH on Strep Throat. Details: Bacterial infection of throat, requires antibiotics.",
            "asthma": "Reference: GINA report on Asthma. Details: Chronic respiratory condition, causes shortness of breath."
        }
        self.ood_threshold = 0.6 # Placeholder for OOD detection probability

    def _preprocess_text(self, patient_narrative: str) -> str:
        return patient_narrative.lower()

    def _extract_symptoms(self, preprocessed_text: str) -> list:
        extracted_symptoms = []
        for symptom_keyword in self.known_symptoms.keys():
            if symptom_keyword in preprocessed_text:
                extracted_symptoms.append(symptom_keyword)
        return extracted_symptoms

    def _predict_diagnosis(self, symptoms: list) -> dict:
        potential_diagnoses = {}
        if not symptoms:
            return {"No specific symptoms detected": 0.5}

        for symptom in symptoms:
            for diagnosis in self.known_symptoms.get(symptom, []):
                potential_diagnoses[diagnosis] = potential_diagnoses.get(diagnosis, 0) + 1
        
        total_symptom_matches = sum(potential_diagnoses.values())
        if total_symptom_matches == 0:
             return {"Uncertain Diagnosis": 0.5}

        # Simple confidence scoring based on symptom count
        confidence_scores = {
            diag: count / total_symptom_matches for diag, count in potential_diagnoses.items()
        }
        return confidence_scores

    def _detect_ood(self, symptoms: list) -> bool:
        # Simple OOD detection: if a significant portion of symptoms are unknown
        known_symptom_count = sum(1 for s in symptoms if s in self.known_symptoms)
        if not symptoms: # No symptoms, could be OOD or benign
            return False
        if known_symptom_count / len(symptoms) < self.ood_threshold:
            return True
        return False

    def _retrieve_references(self, symptoms: list, diagnoses: dict) -> list:
        references = []
        for diagnosis in diagnoses.keys():
            if diagnosis in self.medical_literature:
                references.append(f"For {diagnosis}: {self.medical_literature[diagnosis]}")
        for symptom in symptoms:
            # Also retrieve references related to individual symptoms if available
            if symptom in self.medical_literature:
                 references.append(f"For {symptom}: {self.medical_literature[symptom]}")
        return list(set(references)) # Remove duplicates

    def _style_language(self, diagnosis_output: dict, confidence: float, is_ood: bool) -> str:
        if is_ood:
            return "Given the unusual nature of the symptoms, the system's ability to provide a definitive diagnosis is limited. Further medical evaluation is strongly recommended. Potential areas of concern include... " + diagnosis_output.get("explanation", "")

        if confidence < 0.6:
            prefix = "It appears that... "
        elif confidence < 0.8:
            prefix = "Based on the information provided, it is likely that... "
        else:
            prefix = "The most probable diagnosis is... "
        return prefix + diagnosis_output.get("explanation", "")

    def diagnose(self, patient_narrative: str) -> dict:
        preprocessed_text = self._preprocess_text(patient_narrative)
        extracted_symptoms = self._extract_symptoms(preprocessed_text)
        
        potential_diagnoses_with_confidence = self._predict_diagnosis(extracted_symptoms)
        
        top_diagnosis = max(potential_diagnoses_with_confidence, key=potential_diagnoses_with_confidence.get) if potential_diagnoses_with_confidence else "N/A"
        top_confidence = potential_diagnoses_with_confidence.get(top_diagnosis, 0.0)

        is_ood = self._detect_ood(extracted_symptoms)
        
        references = self._retrieve_references(extracted_symptoms, potential_diagnoses_with_confidence)

        explanation_parts = []
        if top_diagnosis != "N/A":
            explanation_parts.append(f"Top potential diagnosis: {top_diagnosis} (Confidence: {top_confidence:.2f})")
        if extracted_symptoms:
            explanation_parts.append(f"Symptoms identified: {', '.join(extracted_symptoms)}")
        if is_ood:
            explanation_parts.append("Warning: Symptoms may be outside the system's typical training data. Exercise caution.")

        diagnosis_output = {"explanation": ". ".join(explanation_parts)}

        final_explanation = self._style_language(diagnosis_output, top_confidence, is_ood)

        return {
            "patient_input": patient_narrative,
            "extracted_symptoms": extracted_symptoms,
            "potential_diagnoses": potential_diagnoses_with_confidence,
            "is_out_of_distribution": is_ood,
            "final_diagnosis_statement": final_explanation,
            "traceable_references": references
        }


if __name__ == '__main__':
    assistant = MedicalDiagnosisAssistant()

    print("\n--- Case 1: Common Cold Symptoms ---")
    patient_input_1 = "I have a runny nose, mild cough, and a slight sore throat. Feeling tired." 
    result_1 = assistant.diagnose(patient_input_1)
    for key, value in result_1.items():
        print(f"{key}: {value}")

    print("\n--- Case 2: Influenza Symptoms ---")
    patient_input_2 = "High fever, persistent cough, severe body aches, and fatigue." 
    result_2 = assistant.diagnose(patient_input_2)
    for key, value in result_2.items():
        print(f"{key}: {value}")

    print("\n--- Case 3: Out-of-Distribution Symptoms ---")
    patient_input_3 = "My skin is turning blue and I have severe ringing in my ears. I feel dizzy." 
    result_3 = assistant.diagnose(patient_input_3)
    for key, value in result_3.items():
        print(f"{key}: {value}")

    print("\n--- Case 4: High Confidence, Clear Symptoms ---")
    patient_input_4 = "I have a very bad headache, sensitive to light, and feel nauseous." 
    result_4 = assistant.diagnose(patient_input_4)
    for key, value in result_4.items():
        print(f"{key}: {value}")

    print("\n--- Case 5: No Clear Symptoms ---")
    patient_input_5 = "I just feel a bit off today." 
    result_5 = assistant.diagnose(patient_input_5)
    for key, value in result_5.items():
        print(f"{key}: {value}")