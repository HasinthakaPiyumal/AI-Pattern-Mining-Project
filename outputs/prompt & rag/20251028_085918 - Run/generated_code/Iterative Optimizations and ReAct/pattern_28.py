class PatientDataIngestion:
    def ingest(self, raw_data):
        structured_data = {
            "symptoms": raw_data.get("symptoms", []),
            "medical_history": raw_data.get("medical_history", []),
            "lab_results": raw_data.get("lab_results", {}),
            "imaging_reports": raw_data.get("imaging_reports", {})
        }
        return structured_data

class MedicalKnowledgeBase:
    def __init__(self):
        self.knowledge = {
            "flu": {
                "symptoms": ["fever", "cough", "sore throat", "fatigue"],
                "treatment": "Rest, fluids, antivirals"
            },
            "pneumonia": {
                "symptoms": ["cough", "fever", "shortness of breath", "chest pain"],
                "treatment": "Antibiotics, oxygen therapy"
            },
            "asthma": {
                "symptoms": ["wheezing", "shortness of breath", "chest tightness"],
                "treatment": "Inhalers, corticosteroids"
            },
            "migraine": {
                "symptoms": ["severe headache", "nausea", "light sensitivity"],
                "treatment": "Pain relievers, triptans"
            }
        }

    def query(self, topic, detail):
        if topic in self.knowledge:
            return self.knowledge[topic].get(detail)
        return None

class MockExternalTool:
    def perform_lab_test(self, test_name, patient_id):
        if test_name == "CRP" and patient_id == "P101":
            return {"CRP_level": "high"}
        return {"status": "test results pending"}

    def consult_specialist(self, condition, patient_data):
        if "fever" in patient_data.get("symptoms", []) and "cough" in patient_data.get("symptoms", []):
            return {"specialist_opinion": "Consider respiratory specialist for further evaluation."}
        return {"specialist_opinion": "No immediate specialist consultation recommended."}

class DiagnosticAgent:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.external_tool = MockExternalTool()

    def _reason(self, structured_data):
        symptoms = structured_data.get("symptoms", [])
        possible_diagnoses = []
        for condition, info in self.knowledge_base.knowledge.items():
            if all(s in symptoms for s in info["symptoms"]):
                possible_diagnoses.append(condition)
        return possible_diagnoses if possible_diagnoses else ["Undetermined"]

    def _self_correct(self, current_diagnosis, structured_data):
        print(f"Agent reflecting on current diagnosis: {current_diagnosis}")
        if "Undetermined" in current_diagnosis and structured_data.get("lab_results", {}).get("CRP_level") == "high":
            print("Agent noticing high CRP, suggesting inflammatory process.")
            return [d for d in current_diagnosis if d != "Undetermined"] + ["Infection/Inflammation"]
        return current_diagnosis

    def _process_feedback(self, diagnosis, feedback):
        print(f"Agent processing feedback: '{feedback}'")
        if "respiratory issues" in feedback.lower() and "pneumonia" not in diagnosis:
            print("Feedback suggests re-evaluating for respiratory conditions.")
            if self.knowledge_base.query("pneumonia", "symptoms") is not None:
                return diagnosis + ["Pneumonia (re-evaluation)"]
        return diagnosis

    def diagnose(self, raw_patient_data, max_iterations=3):
        print("--- Starting Diagnostic Process ---")
        ingestor = PatientDataIngestion()
        structured_data = ingestor.ingest(raw_patient_data)
        print(f"Initial structured data: {structured_data}")

        current_diagnosis = self._reason(structured_data)
        print(f"Initial reasoning result: {current_diagnosis}")

        for i in range(max_iterations):
            print(f"\n--- Iteration {i+1} ---")

            # Self-correction
            corrected_diagnosis = self._self_correct(current_diagnosis, structured_data)
            if corrected_diagnosis != current_diagnosis:
                print(f"Self-corrected diagnosis: {corrected_diagnosis}")
                current_diagnosis = corrected_diagnosis

            # Tool Manipulation (simulated)
            if "Undetermined" in current_diagnosis or "Infection/Inflammation" in current_diagnosis:
                print("Agent using external tool: Performing lab test (CRP)")
                lab_result = self.external_tool.perform_lab_test("CRP", raw_patient_data.get("patient_id"))
                structured_data["lab_results"].update(lab_result)
                print(f"Tool output (CRP): {lab_result}")
                # Re-evaluate after tool output
                current_diagnosis = self._reason(structured_data)
                current_diagnosis = self._self_correct(current_diagnosis, structured_data)
                print(f"Diagnosis after CRP test: {current_diagnosis}")

            # Simulate Clinician Feedback (manual for demonstration)
            if i == 0 and "Undetermined" in current_diagnosis:
                clinician_feedback = "Patient also reported increasing shortness of breath and chest discomfort."
                current_diagnosis = self._process_feedback(current_diagnosis, clinician_feedback)
                print(f"Diagnosis after clinician feedback: {current_diagnosis}")

            # Refine reasoning based on new data/feedback
            refined_diagnosis = self._reason(structured_data)
            if set(refined_diagnosis) != set(current_diagnosis):
                print(f"Refined diagnosis based on new information: {refined_diagnosis}")
                current_diagnosis = refined_diagnosis
            
            if "respiratory specialist" in self.external_tool.consult_specialist(current_diagnosis, structured_data).get("specialist_opinion", "").lower() and "specialist_consulted" not in structured_data:
                print("Agent using external tool: Consulting specialist.")
                specialist_recommendation = self.external_tool.consult_specialist(current_diagnosis, structured_data)
                print(f"Specialist recommendation: {specialist_recommendation['specialist_opinion']}")
                structured_data["specialist_consulted"] = True
                # This could trigger further data acquisition or reasoning in a real system

            if not any(d in ["Undetermined", "Infection/Inflammation"] for d in current_diagnosis) and len(current_diagnosis) == 1:
                print("Diagnosis seems stable and specific.")
                break

        print("--- Diagnostic Process Complete ---")
        return {"final_diagnosis": list(set(current_diagnosis)), "suggested_treatment": [self.knowledge_base.query(d.replace(' (re-evaluation)', ''), 'treatment') for d in current_diagnosis if self.knowledge_base.query(d.replace(' (re-evaluation)', ''), 'treatment')]}


if __name__ == "__main__":
    kb = MedicalKnowledgeBase()
    agent = DiagnosticAgent(kb)

    patient_data_1 = {
        "patient_id": "P101",
        "symptoms": ["fever", "cough", "fatigue"],
        "medical_history": ["seasonal allergies"]
    }

    patient_data_2 = {
        "patient_id": "P102",
        "symptoms": ["severe headache", "nausea"],
        "medical_history": []
    }

    print("\n*** Diagnosing Patient 1 ***")
    result_1 = agent.diagnose(patient_data_1)
    print(f"Final Diagnosis for P101: {result_1}")

    print("\n*** Diagnosing Patient 2 ***")
    result_2 = agent.diagnose(patient_data_2)
    print(f"Final Diagnosis for P102: {result_2}")
