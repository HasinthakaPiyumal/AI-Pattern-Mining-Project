import pandas as pd
import time # To simulate processing

class MedicalDiagnosisAssistant:
    def __init__(self, medical_knowledge_base=None):
        # In a real app, this would be a sophisticated RAG system or a finely tuned LLM
        self.medical_knowledge_base = medical_knowledge_base
        self.history = []

    def _generate_potential_diagnoses(self, patient_data: dict, current_reasoning: list):
        """
        Simulates an LLM generating potential diagnostic steps or hypotheses.
        In a real scenario, this would involve prompting an LLM with patient data.
        """
        print(f"\n--- Generating potential diagnoses based on patient data: {patient_data.get('symptoms', 'N/A')} ---")
        time.sleep(1) # Simulate processing time

        if not current_reasoning:
            # Initial hypotheses
            if "fever" in patient_data.get("symptoms", []) and "cough" in patient_data.get("symptoms", []):
                 return [
                    {"diagnosis": "Common Cold", "confidence": 0.6, "reason": "Fever, cough, fatigue are common symptoms."},
                    {"diagnosis": "Influenza", "confidence": 0.5, "reason": "Similar to cold but often more severe."},
                    {"diagnosis": "Bacterial Infection (e.g., Bronchitis)", "confidence": 0.3, "reason": "Possible if symptoms persist or worsen significantly."},
                ]
            elif "fever" in patient_data.get("symptoms", []) and "rash" in patient_data.get("symptoms", []) and "headache" in patient_data.get("symptoms", []) and "no vaccinations" in patient_data.get("medical_history", []):
                 return [
                    {"diagnosis": "Measles", "confidence": 0.8, "reason": "Classic fever and rash presentation in an unvaccinated patient."},
                    {"diagnosis": "Rubella", "confidence": 0.7, "reason": "Similar but milder rash."},
                    {"diagnosis": "Drug Reaction", "confidence": 0.4, "reason": "Can present with fever and rash."},
                ]
            else:
                return [
                    {"diagnosis": "General Illness (undiagnosed)", "confidence": 0.4, "reason": "Initial broad hypotheses."},
                    {"diagnosis": "Consult Specialist", "confidence": 0.3, "reason": "Symptoms are not clear-cut."},
                ]
        else:
            # Refine based on current reasoning and new info
            last_accepted_or_suggested = current_reasoning[-1]
            if "shortness of breath" in patient_data.get("symptoms", []) and last_accepted_or_suggested.get("diagnosis") in ["Common Cold", "Influenza"]:
                 return [
                    {"diagnosis": "Bronchitis", "confidence": 0.7, "reason": "Cold/flu symptoms plus shortness of breath suggests lower respiratory involvement."},
                    {"diagnosis": "Pneumonia", "confidence": 0.6, "reason": "More severe respiratory infection, especially if fever is high and X-ray shows infiltrates."},
                    {"diagnosis": "Asthma Exacerbation", "confidence": 0.4, "reason": "If patient has history of asthma and current triggers."},
                ]
            elif "chest_xray_shows_infiltrates" in patient_data.get("test_results", []) and "shortness of breath" in patient_data.get("symptoms", []):
                 return [
                    {"diagnosis": "Pneumonia", "confidence": 0.9, "reason": "Chest X-ray infiltrates strongly suggest pneumonia with respiratory symptoms."},
                    {"diagnosis": "Acute Bronchitis (severe)", "confidence": 0.7, "reason": "Severe bronchitis can sometimes show infiltrates."},
                ]
            elif "persistent cough" in patient_data.get("new_symptom", "") and last_accepted_or_suggested.get("diagnosis") in ["Bronchitis"]:
                return [
                    {"diagnosis": "Chronic Bronchitis", "confidence": 0.75, "reason": "Persistent cough, especially in a smoker, points to chronic issues."},
                    {"diagnosis": "Asthma", "confidence": 0.5, "reason": "Persistent cough can also be an asthma symptom."},
                ]
            
            # Fallback for refinement if specific conditions not met
            return [
                {"diagnosis": f"Further investigation on {last_accepted_or_suggested.get('diagnosis', 'previous finding')}", "confidence": 0.6, "reason": "Refining based on past step."},
                {"diagnosis": "Re-evaluate all symptoms", "confidence": 0.4, "reason": "Considering other possibilities."},
            ]


    def _evaluate_and_accept_diagnosis(self, patient_data: dict, proposed_diagnoses: list, iteration: int):
        """
        Simulates an LLM evaluating proposed diagnoses.
        Decides which steps to accept, reject, or if more information is needed.
        """
        print(f"\n--- Evaluating proposed diagnoses (Iteration {iteration}) ---")
        time.sleep(1) # Simulate processing time

        best_diagnosis = None
        highest_confidence = 0

        for diag in proposed_diagnoses:
            print(f"  - Proposed: {diag['diagnosis']} (Confidence: {diag['confidence']:.2f}) - Reason: {diag['reason']}")
            # Complex evaluation logic would go here, potentially querying medical databases
            # For this simulation, we'll pick the one with highest confidence and apply some basic rules
            if diag['confidence'] > highest_confidence:
                highest_confidence = diag['confidence']
                best_diagnosis = diag

        if best_diagnosis and highest_confidence >= 0.75: # Threshold for acceptance
            print(f"  - Accepted diagnosis: {best_diagnosis['diagnosis']} (Confidence: {best_diagnosis['confidence']:.2f})")
            return {"status": "accepted", "diagnosis": best_diagnosis}
        else:
            print(f"  - No diagnosis met acceptance criteria or more information is needed.")
            # In a real scenario, this would also suggest what additional tests or questions are needed
            if "shortness of breath" in patient_data.get("symptoms", []) and iteration == 1:
                return {"status": "needs_more_info", "suggestion": "Consider chest X-ray or lung function tests."}
            if "persistent cough" in patient_data.get("new_symptom", "") and "smoker" in patient_data.get("medical_history", []) and iteration == 2:
                 return {"status": "needs_more_info", "suggestion": "Investigate for chronic respiratory conditions given persistent cough and smoking history."}
            return {"status": "needs_more_info", "suggestion": "Gather more patient history, perform additional tests, or consult a specialist."}

    def diagnose_patient(self, patient_data: dict, max_iterations=5):
        """
        Applies the Cumulative Reasoning pattern to diagnose a patient.
        """
        print(f"Starting diagnosis process for patient with symptoms: {patient_data.get('symptoms', 'N/A')}")
        current_reasoning_steps = []
        final_diagnosis = None

        for i in range(1, max_iterations + 1):
            print(f"\n=== Iteration {i} ===")

            # Step 1: Generate potential steps/diagnoses
            proposed_diagnoses = self._generate_potential_diagnoses(patient_data, current_reasoning_steps)

            # Step 2: Evaluate and decide
            evaluation_result = self._evaluate_and_accept_diagnosis(patient_data, proposed_diagnoses, i)

            if evaluation_result["status"] == "accepted":
                final_diagnosis = evaluation_result["diagnosis"]
                current_reasoning_steps.append(final_diagnosis)
                print(f"\nDiagnosis finalized after {i} iterations.")
                break
            elif evaluation_result["status"] == "needs_more_info":
                print(f"  - Assistant suggests: {evaluation_result['suggestion']}")
                # In a real application, this would prompt the user for more data or trigger external tests
                # For this demo, let's simulate adding more data if suggested
                if "chest X-ray" in evaluation_result['suggestion'] and "shortness of breath" in patient_data.get("symptoms", []) and "cough" in patient_data.get("symptoms", []) and "chest_xray_shows_infiltrates" not in patient_data.get("test_results", []) :
                    patient_data["test_results"] = patient_data.get("test_results", []) + ["chest_xray_shows_infiltrates"]
                    print("  - Simulating new information: Chest X-ray shows infiltrates.")
                if "Investigate for chronic respiratory conditions" in evaluation_result['suggestion'] and "new_symptom" not in patient_data:
                    patient_data["new_symptom"] = "persistent cough"
                    print("  - Simulating new information: Patient reports persistent cough.")
                current_reasoning_steps.append({"status": "refinement_needed", "suggestion": evaluation_result["suggestion"]})
            else:
                print("  - Unexpected evaluation result status.")
                break

        if final_diagnosis:
            print("\n--- Final Diagnosis ---")
            print(f"Diagnosis: {final_diagnosis['diagnosis']}")
            print(f"Confidence: {final_diagnosis['confidence']:.2f}")
            print(f"Reason: {final_diagnosis['reason']}")
        else:
            print(f"\nCould not reach a conclusive diagnosis after {max_iterations} iterations. Further human intervention may be required.")

        return final_diagnosis

if __name__ == "__main__":
    assistant = MedicalDiagnosisAssistant()

    # Example 1: Simple case - Common Cold Symptoms
    print("\n##### Patient Case 1: Common Cold Symptoms #####")
    patient_1_data = {
        "symptoms": ["fever", "cough", "fatigue", "sore throat"],
        "medical_history": ["no chronic conditions"],
        "test_results": []
    }
    assistant.diagnose_patient(patient_1_data)

    print("\n\n##### Patient Case 2: More Complex Symptoms (requiring refinement) #####")
    patient_2_data = {
        "symptoms": ["fever", "cough", "fatigue", "shortness of breath"],
        "medical_history": ["smoker"],
        "test_results": []
    }
    assistant.diagnose_patient(patient_2_data)

    print("\n\n##### Patient Case 3: Rash and Fever (requiring specific initial generation) #####")
    patient_3_data = {
        "symptoms": ["fever", "rash", "headache"],
        "medical_history": ["no vaccinations"],
        "test_results": []
    }
    assistant.diagnose_patient(patient_3_data)

    print("\n\n##### Patient Case 4: Undiagnosed case (demonstrates iteration without clear path) #####")
    patient_4_data = {
        "symptoms": ["abdominal pain", "nausea"],
        "medical_history": [],
        "test_results": []
    }
    assistant.diagnose_patient(patient_4_data)
