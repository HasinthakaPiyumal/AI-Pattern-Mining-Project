class MedicalKnowledgeBase:
    def __init__(self):
        self.disease_symptom_map = {
            "Influenza": {"fever", "cough", "fatigue", "sore throat", "body aches"},
            "Common Cold": {"cough", "sore throat", "runny nose", "sneezing"},
            "Bronchitis": {"cough", "mucus", "chest discomfort", "shortness of breath"},
            "Strep Throat": {"sore throat", "fever", "swollen lymph nodes", "headache"},
            "Pneumonia": {"cough", "fever", "chills", "shortness of breath", "chest pain"}
        }
        self.drug_interaction_rules = {
            ("warfarin", "aspirin"): "Increased bleeding risk",
            ("simvastatin", "grapefruit juice"): "Increased statin levels",
            ("antibiotic", "oral contraceptive"): "Reduced contraceptive effectiveness"
        }
        self.lab_result_interpretations = {
            "positive influenza test": {"diagnosis": "Influenza", "certainty": "high"},
            "elevated CRP": {"condition": "Inflammation", "certainty": "moderate"},
            "low hemoglobin": {"condition": "Anemia", "certainty": "high"}
        }

    def query_disease_symptom_database(self, symptoms):
        reasoning = f"Consulted Disease Symptom Database with symptoms: {', '.join(symptoms)}.\n"
        possible_diseases = []
        for disease, known_symptoms in self.disease_symptom_map.items():
            matched_symptoms = symptoms.intersection(known_symptoms)
            if len(matched_symptoms) > 0:
                match_percentage = len(matched_symptoms) / len(known_symptoms) * 100
                possible_diseases.append((disease, match_percentage))
        possible_diseases.sort(key=lambda x: x[1], reverse=True)
        
        tool_output = {"possible_diseases": possible_diseases}
        reasoning += f"Intermediate results: {tool_output}.\n"
        return tool_output, reasoning

    def check_drug_interactions(self, medications):
        reasoning = f"Consulted Drug Interaction Checker with medications: {', '.join(medications)}.\n"
        interactions = []
        for i in range(len(medications)):
            for j in range(i + 1, len(medications)):
                drug1, drug2 = sorted([medications[i], medications[j]])
                interaction_key = (drug1.lower(), drug2.lower())
                if interaction_key in self.drug_interaction_rules:
                    interactions.append(f"{drug1} and {drug2}: {self.drug_interaction_rules[interaction_key]}")
        
        tool_output = {"interactions": interactions}
        reasoning += f"Intermediate results: {tool_output}.\n"
        return tool_output, reasoning

    def interpret_lab_results(self, lab_results):
        reasoning = f"Consulted Lab Test Interpretation Tool with results: {', '.join(lab_results)}.\n"
        interpretations = []
        for result in lab_results:
            if result.lower() in self.lab_result_interpretations:
                interpretations.append({result: self.lab_result_interpretations[result.lower()]})
        
        tool_output = {"interpretations": interpretations}
        reasoning += f"Intermediate results: {tool_output}.\n"
        return tool_output, reasoning


class TransparentDiagnosisAssistant:
    def __init__(self):
        self.knowledge_base = MedicalKnowledgeBase()

    def diagnose_patient(self, patient_data):
        reasoning_log = []
        final_diagnosis_candidates = []
        
        symptoms = set(patient_data.get("symptoms", []))
        medications = patient_data.get("medications", [])
        lab_results = patient_data.get("lab_results", [])
        patient_history = patient_data.get("medical_history", "")

        reasoning_log.append("--- Starting Diagnosis Process ---")
        reasoning_log.append(f"Patient Data Received: Symptoms: {', '.join(symptoms) if symptoms else 'None'}, Medications: {', '.join(medications) if medications else 'None'}, Lab Results: {', '.join(lab_results) if lab_results else 'None'}, Medical History: {patient_history}")

        # Step 1: Query Disease Symptom Database
        reasoning_log.append("\n--- Step 1: Symptom Analysis ---")
        symptom_query_output, symptom_query_reasoning = self.knowledge_base.query_disease_symptom_database(symptoms)
        reasoning_log.append(symptom_query_reasoning)
        if symptom_query_output["possible_diseases"]:
            reasoning_log.append("Rationale for next step: Identified potential diseases based on symptoms to narrow down possibilities.")
            final_diagnosis_candidates.extend(symptom_query_output["possible_diseases"])
        else:
            reasoning_log.append("No direct disease matches found based on symptoms alone.")

        # Step 2: Check Drug Interactions
        if medications:
            reasoning_log.append("\n--- Step 2: Medication Review ---")
            drug_interaction_output, drug_interaction_reasoning = self.knowledge_base.check_drug_interactions(medications)
            reasoning_log.append(drug_interaction_reasoning)
            if drug_interaction_output["interactions"]:
                reasoning_log.append("Rationale for next step: Identified potential drug interactions that could influence diagnosis or treatment plan.")
                for interaction in drug_interaction_output["interactions"]:
                    reasoning_log.append(f"Significant Finding: {interaction}")
            else:
                reasoning_log.append("No significant drug interactions found.")
        else:
            reasoning_log.append("\n--- Step 2: Medication Review ---")
            reasoning_log.append("No medications provided for interaction check.")

        # Step 3: Interpret Lab Results
        if lab_results:
            reasoning_log.append("\n--- Step 3: Lab Results Interpretation ---")
            lab_interpretation_output, lab_interpretation_reasoning = self.knowledge_base.interpret_lab_results(lab_results)
            reasoning_log.append(lab_interpretation_reasoning)
            if lab_interpretation_output["interpretations"]:
                reasoning_log.append("Rationale for next step: Lab results provide objective evidence for confirming or ruling out conditions.")
                for interpretation_dict in lab_interpretation_output["interpretations"]:
                    for result, interpretation in interpretation_dict.items():
                        reasoning_log.append(f"Lab Result '{result}': Interpreted as '{interpretation['condition']}' with {interpretation['certainty']} certainty.")
                        if interpretation['diagnosis']:
                             final_diagnosis_candidates.append((interpretation['diagnosis'], 100)) # High certainty from lab test
            else:
                reasoning_log.append("No specific interpretations found for provided lab results.")
        else:
            reasoning_log.append("\n--- Step 3: Lab Results Interpretation ---")
            reasoning_log.append("No lab results provided for interpretation.")
        
        # Step 4: Final Synthesis and Diagnosis Selection
        reasoning_log.append("\n--- Step 4: Final Synthesis and Diagnosis Selection ---")
        final_diagnosis = "Undetermined"
        best_match_score = 0
        
        # Simple aggregation: prioritize direct lab diagnoses, then symptom matches
        for candidate, score in final_diagnosis_candidates:
            if score == 100: # Direct lab confirmation
                final_diagnosis = candidate
                reasoning_log.append(f"Final Diagnosis Rationale: {final_diagnosis} confirmed by highly certain lab results.")
                break
            elif score > best_match_score:
                best_match_score = score
                final_diagnosis = candidate
        
        if final_diagnosis == "Undetermined" and symptoms:
            # If no lab confirmation, pick highest symptom match
            if final_diagnosis_candidates:
                final_diagnosis = max(final_diagnosis_candidates, key=lambda item: item[1])[0]
                reasoning_log.append(f"Final Diagnosis Rationale: Best symptomatic match found, '{final_diagnosis}', with {best_match_score:.2f}% symptom correlation. Further tests may be needed for confirmation.")
            else:
                reasoning_log.append("Final Diagnosis Rationale: Unable to determine a specific diagnosis based on the provided information.")
        elif final_diagnosis == "Undetermined" and not symptoms and not lab_results:
             reasoning_log.append("Final Diagnosis Rationale: Insufficient information (no symptoms or lab results) to determine a specific diagnosis.")

        reasoning_log.append("--- Diagnosis Process Complete ---")
        return final_diagnosis, reasoning_log


def main():
    assistant = TransparentDiagnosisAssistant()

    # --- Example 1: Clear Case with Lab Result ---
    print("\n===================================================")
    print("Example 1: Clear Case with Lab Result (Influenza)")
    print("===================================================")
    patient_data_1 = {
        "symptoms": ["fever", "cough", "fatigue"],
        "medications": ["aspirin"],
        "lab_results": ["positive influenza test", "elevated CRP"],
        "medical_history": "Patient recently traveled internationally."
    }
    diagnosis_1, reasoning_1 = assistant.diagnose_patient(patient_data_1)
    print(f"\nFINAL DIAGNOSIS: {diagnosis_1}\n")
    print("DETAILED REASONING:")
    for step in reasoning_1:
        print(step)

    # --- Example 2: Symptom-based Diagnosis with Interactions ---
    print("\n=======================================================")
    print("Example 2: Symptom-based Diagnosis with Drug Interactions")
    print("=======================================================")
    patient_data_2 = {
        "symptoms": ["sore throat", "fever", "swollen lymph nodes"],
        "medications": ["warfarin", "aspirin"],
        "lab_results": [],
        "medical_history": "Patient has history of blood clots."
    }
    diagnosis_2, reasoning_2 = assistant.diagnose_patient(patient_data_2)
    print(f"\nFINAL DIAGNOSIS: {diagnosis_2}\n")
    print("DETAILED REASONING:")
    for step in reasoning_2:
        print(step)

    # --- Example 3: Insufficient Information ---
    print("\n========================================")
    print("Example 3: Insufficient Information")
    print("========================================")
    patient_data_3 = {
        "symptoms": [],
        "medications": ["simvastatin"],
        "lab_results": [],
        "medical_history": ""
    }
    diagnosis_3, reasoning_3 = assistant.diagnose_patient(patient_data_3)
    print(f"\nFINAL DIAGNOSIS: {diagnosis_3}\n")
    print("DETAILED REASONING:")
    for step in reasoning_3:
        print(step)

if __name__ == "__main__":
    main()