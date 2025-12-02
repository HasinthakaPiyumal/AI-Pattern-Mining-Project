class MetacognitivePromptingEngine:
    def __init__(self):
        self.report = {}

    def clarify_question(self, patient_data):
        clarified_case = f"Patient presents with: {patient_data.get('symptoms', '')}. Relevant history: {patient_data.get('history', '')}. Test results: {patient_data.get('test_results', '')}.\nRephrased for clarity: The core issue appears to be {patient_data.get('symptoms', '')} in a patient with a history of {patient_data.get('history', '')}. We need to focus on potential conditions indicated by these findings."
        self.report["Clarified Question"] = clarified_case
        return clarified_case

    def preliminary_judgment(self, clarified_case):
        if "fever" in clarified_case.lower() and "cough" in clarified_case.lower():
            differential_diagnoses = [
                {"diagnosis": "Influenza", "reasoning": "Common viral infection with fever and cough."},
                {"diagnosis": "Bacterial Pneumonia", "reasoning": "Could also present with fever and cough, potentially more severe."},
                {"diagnosis": "Bronchitis", "reasoning": "Inflammation of bronchial tubes, often viral, causing cough."}
            ]
        elif "chest pain" in clarified_case.lower() and "shortness of breath" in clarified_case.lower():
             differential_diagnoses = [
                {"diagnosis": "Myocardial Infarction", "reasoning": "Classic symptoms of a heart attack, especially with acute onset."},
                {"diagnosis": "Pulmonary Embolism", "reasoning": "Blood clot in the lungs, can cause sudden chest pain and dyspnea."},
                {"diagnosis": "Anxiety Attack", "reasoning": "Can mimic cardiac symptoms, especially with stress."}
            ]
        else:
            differential_diagnoses = [
                {"diagnosis": "Generic Illness", "reasoning": "Insufficient information to provide specific differential diagnoses."}
            ]
        self.report["Preliminary Judgment"] = differential_diagnoses
        return differential_diagnoses

    def evaluate_response(self, differential_diagnoses):
        evaluation_report = []
        for diagnosis_entry in differential_diagnoses:
            diagnosis = diagnosis_entry["diagnosis"]
            reasoning = diagnosis_entry["reasoning"]
            evaluation = f"Evaluating {diagnosis}:\n"
            evaluation += f"- Consistency: The symptoms are generally consistent with {diagnosis}.\n"
            evaluation += f"- Confounding Factors: Consider patient's age and co-morbidities for {diagnosis}.\n"
            evaluation += f"- Strengths: {reasoning}.\n"
            evaluation += f"- Weaknesses: Further tests needed to confirm/rule out {diagnosis} conclusively."
            evaluation_report.append({"diagnosis": diagnosis, "evaluation": evaluation})
        self.report["Evaluation of Response"] = evaluation_report
        return evaluation_report

    def decision_confirmation(self, evaluation_report):
        if not evaluation_report:
            final_diagnosis = {"diagnosis": "Undetermined", "reasoning": "No diagnoses to evaluate."}
        else:
            best_diagnosis = evaluation_report[0]
            final_diagnosis = {
                "diagnosis": best_diagnosis["diagnosis"],
                "reasoning": f"Based on evaluation, {best_diagnosis['diagnosis']} appears to be the most probable. {best_diagnosis['evaluation']}"
            }
        self.report["Decision Confirmation"] = final_diagnosis
        return final_diagnosis

    def confidence_assessment(self, final_diagnosis):
        confidence_level = "Medium"
        uncertainties = "Further lab tests (e.g., blood work, imaging) are recommended to definitively confirm the diagnosis and rule out other possibilities. Patient history could be more detailed."
        recommendations = "Recommend ordering specific diagnostic tests based on the leading diagnosis and close monitoring of symptoms."

        if final_diagnosis["diagnosis"] == "Undetermined":
            confidence_level = "Low"
            uncertainties = "Significant lack of information. Cannot provide a confident diagnosis."
            recommendations = "Request more detailed patient information and comprehensive diagnostic workup."

        assessment = {
            "confidence_level": confidence_level,
            "uncertainties": uncertainties,
            "recommendations": recommendations
        }
        self.report["Confidence Assessment"] = assessment
        return assessment

    def diagnose(self, patient_data):
        print("--- Starting Metacognitive Diagnostic Process ---")
        
        clarified_case = self.clarify_question(patient_data)
        print(f"\n1. Clarifying the Question:\n{self.report['Clarified Question']}")
        
        preliminary_diagnoses = self.preliminary_judgment(clarified_case)
        print(f"\n2. Preliminary Judgment (Differential Diagnoses):")
        for diag in self.report['Preliminary Judgment']:
            print(f"- {diag['diagnosis']}: {diag['reasoning']}")

        evaluation = self.evaluate_response(preliminary_diagnoses)
        print(f"\n3. Evaluation of Response:")
        for eval_item in self.report['Evaluation of Response']:
            print(f"{eval_item['evaluation']}")

        final_diagnosis = self.decision_confirmation(evaluation)
        print(f"\n4. Decision Confirmation (Final Diagnosis):\nDiagnosis: {self.report['Decision Confirmation']['diagnosis']}\nReasoning: {self.report['Decision Confirmation']['reasoning']}")

        confidence = self.confidence_assessment(final_diagnosis)
        print(f"\n5. Confidence Assessment:\nConfidence Level: {self.report['Confidence Assessment']['confidence_level']}\nUncertainties: {self.report['Confidence Assessment']['uncertainties']}\nRecommendations: {self.report['Confidence Assessment']['recommendations']}")
        
        print("\n--- Metacognitive Diagnostic Process Complete ---")
        return self.report

if __name__ == "__main__":
    patient_case_1 = {
        "symptoms": "severe cough, fever, fatigue",
        "history": "smoker, recent travel abroad",
        "test_results": "no specific results available yet"
    }

    patient_case_2 = {
        "symptoms": "sudden chest pain, shortness of breath",
        "history": "no significant medical history",
        "test_results": "ECG pending"
    }

    patient_case_3 = {
        "symptoms": "mild headache",
        "history": "stressful week",
        "test_results": "none"
    }

    diagnostic_assistant = MetacognitivePromptingEngine()

    print("\n\n--- Diagnosing Patient Case 1 ---")
    report_1 = diagnostic_assistant.diagnose(patient_case_1)

    print("\n\n--- Diagnosing Patient Case 2 ---")
    diagnostic_assistant = MetacognitivePromptingEngine() # Reset for new case
    report_2 = diagnostic_assistant.diagnose(patient_case_2)

    print("\n\n--- Diagnosing Patient Case 3 ---")
    diagnostic_assistant = MetacognitivePromptingEngine() # Reset for new case
    report_3 = diagnostic_assistant.diagnose(patient_case_3)