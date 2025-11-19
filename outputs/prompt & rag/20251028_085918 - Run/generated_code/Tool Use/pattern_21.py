class MedicalDiagnosticAgent:
    def __init__(self):
        self.medical_knowledge_base = {
            "fever": "Common symptom of infection. Could be flu, common cold, or more serious conditions.",
            "cough": "Can indicate respiratory infection, allergies, or asthma.",
            "fatigue": "Non-specific symptom, could be due to lack of sleep, stress, or underlying illness.",
            "headache": "Migraine, tension headache, or symptom of other conditions.",
            "sore throat": "Often caused by viral infections like the common cold or flu."
        }

        self.drug_interaction_data = {
            ("warfarin", "aspirin"): "Increased risk of bleeding.",
            ("lisinopril", "ibuprofen"): "Reduced effectiveness of lisinopril, potential kidney issues.",
            ("paracetamol", "alcohol"): "Increased risk of liver damage."
        }

        self.lab_normal_ranges = {
            "hemoglobin": (12.0, 16.0), # g/dL for women, slightly higher for men
            "white_blood_cells": (4.0, 11.0), # 10^9/L
            "glucose": (70, 99) # mg/dL
        }

    def _get_medical_info(self, keyword):
        return self.medical_knowledge_base.get(keyword.lower(), "No specific information found for this keyword.")

    def _interpret_lab_results(self, lab_results):
        interpretation = {}
        anomalies = []
        for test, value in lab_results.items():
            if test in self.lab_normal_ranges:
                low, high = self.lab_normal_ranges[test]
                if value < low:
                    anomalies.append(f"{test} is low ({value} outside {low}-{high}).")
                    interpretation[test] = "Low"
                elif value > high:
                    anomalies.append(f"{test} is high ({value} outside {low}-{high}).")
                    interpretation[test] = "High"
                else:
                    interpretation[test] = "Normal"
            else:
                interpretation[test] = "Unknown test"
        return {"interpretation": interpretation, "anomalies": anomalies}

    def _check_drug_interactions(self, medications):
        interactions = []
        med_set = set(m.lower() for m in medications)
        for (drug1, drug2), effect in self.drug_interaction_data.items():
            if drug1 in med_set and drug2 in med_set:
                interactions.append(f"Potential interaction between {drug1} and {drug2}: {effect}")
        return interactions

    def _analyze_medical_image(self, image_data):
        return "Medical imaging analysis requires specialized AI models. Placeholder output: Image appears consistent with provided context, but further expert review is recommended."

    def _plan_tool_use(self, symptoms, medical_history, lab_results, medications):
        tools_to_use = []
        if symptoms:
            tools_to_use.append("medical_knowledge_base")
        if lab_results:
            tools_to_use.append("lab_result_interpreter")
        if medications:
            tools_to_use.append("drug_interaction_checker")
        # This tool is a placeholder for future integration
        # if medical_history and "imaging_requested" in medical_history: # Example condition
        #     tools_to_use.append("medical_imaging_analyzer")
        return tools_to_use

    def _execute_tool(self, tool_name, input_data):
        if tool_name == "medical_knowledge_base":
            return {symptom: self._get_medical_info(symptom) for symptom in input_data if symptom}
        elif tool_name == "lab_result_interpreter":
            return self._interpret_lab_results(input_data)
        elif tool_name == "drug_interaction_checker":
            return self._check_drug_interactions(input_data)
        elif tool_name == "medical_imaging_analyzer":
            return self._analyze_medical_image(input_data)
        return {}

    def _synthesize_diagnosis(self, tool_outputs, symptoms, medical_history):
        synthesis = []
        differential_diagnoses = []

        synthesis.append(f"Patient presented with symptoms: {', '.join(symptoms)}.")
        synthesis.append(f"Medical history: {medical_history if medical_history else 'None provided'}.")

        if "medical_knowledge_base" in tool_outputs:
            synthesis.append("--- Medical Knowledge Base Insights ---")
            for symptom, info in tool_outputs["medical_knowledge_base"].items():
                synthesis.append(f"  {symptom.capitalize()}: {info}")
                if "infection" in info.lower():
                    differential_diagnoses.append("Infectious disease")
                elif "respiratory" in info.lower():
                    differential_diagnoses.append("Respiratory condition")

        if "lab_result_interpreter" in tool_outputs:
            synthesis.append("--- Lab Results Interpretation ---")
            lab_interp = tool_outputs["lab_result_interpreter"]
            for test, status in lab_interp["interpretation"].items():
                synthesis.append(f"  {test.replace('_', ' ').capitalize()}: {status}")
            if lab_interp["anomalies"]:
                synthesis.append("  Anomalies detected: " + "; ".join(lab_interp["anomalies"]))
                if any("white_blood_cells is high" in a for a in lab_interp["anomalies"]):
                    differential_diagnoses.append("Bacterial infection")
                if any("glucose is high" in a for a in lab_interp["anomalies"]):
                    differential_diagnoses.append("Potential diabetes")

        if "drug_interaction_checker" in tool_outputs:
            synthesis.append("--- Drug Interaction Check ---")
            if tool_outputs["drug_interaction_checker"]:
                for interaction in tool_outputs["drug_interaction_checker"]:
                    synthesis.append(f"  {interaction}")
            else:
                synthesis.append("  No significant drug interactions detected.")

        # Add placeholder for imaging analysis
        if "medical_imaging_analyzer" in tool_outputs and tool_outputs["medical_imaging_analyzer"]:
            synthesis.append("--- Medical Imaging Analysis ---")
            synthesis.append(f"  {tool_outputs['medical_imaging_analyzer']}")

        return { "summary": "\n".join(synthesis), "differential_diagnoses": list(set(differential_diagnoses))}

    def _recommend_actions(self, synthesized_data):
        recommendations = []
        differential_diagnoses = synthesized_data.get("differential_diagnoses", [])
        summary = synthesized_data.get("summary", "")

        if differential_diagnoses:
            recommendations.append(f"Based on initial assessment, differential diagnoses include: {', '.join(differential_diagnoses)}.")
            if "Infectious disease" in differential_diagnoses or "Bacterial infection" in differential_diagnoses:
                recommendations.append("  - Consider further tests like bacterial cultures or a complete blood count (if not already done).")
                recommendations.append("  - Prescribe antibiotics if bacterial infection is confirmed and appropriate.")
            if "Potential diabetes" in differential_diagnoses:
                recommendations.append("  - Recommend a fasting blood glucose test or HbA1c to confirm diabetes.")
                recommendations.append("  - Advise on lifestyle changes and potential medication if diagnosed.")
        else:
            recommendations.append("Further investigation is needed to narrow down potential causes.")

        # General recommendations
        if "Anomalies detected" in summary:
            recommendations.append("  - Review specific lab anomalies with a specialist.")
        if "Potential interaction" in summary:
            recommendations.append("  - Re-evaluate current medications for potential interactions and adjust as necessary.")

        recommendations.append("  - Advise patient to monitor symptoms and seek immediate medical attention if symptoms worsen.")
        recommendations.append("  - Follow up with a primary care physician for comprehensive evaluation.")

        suggested_tests = []
        if "Infectious disease" in differential_diagnoses or "Bacterial infection" in differential_diagnoses:
            suggested_tests.append("Complete Blood Count (CBC)")
            suggested_tests.append("Bacterial culture (if specific infection suspected)")
        if "Potential diabetes" in differential_diagnoses:
            suggested_tests.append("Fasting Blood Glucose")
            suggested_tests.append("HbA1c")

        treatment_plans = []
        if "Infectious disease" in differential_diagnoses or "Bacterial infection" in differential_diagnoses:
            treatment_plans.append("Antibiotics (if bacterial)")
            treatment_plans.append("Symptomatic relief (e.g., fever reducers, cough suppressants)")
        if "Potential diabetes" in differential_diagnoses:
            treatment_plans.append("Dietary and lifestyle modifications")
            treatment_plans.append("Antidiabetic medications (if confirmed)")

        return {
            "overall_recommendations": "\n".join(recommendations),
            "suggested_tests": list(set(suggested_tests)),
            "recommended_treatment_plans": list(set(treatment_plans))
        }

    def diagnose_patient(self, symptoms, medical_history, lab_results=None, medications=None):
        tool_outputs = {}
        tools_to_use = self._plan_tool_use(symptoms, medical_history, lab_results, medications)

        if "medical_knowledge_base" in tools_to_use and symptoms:
            tool_outputs["medical_knowledge_base"] = self._execute_tool("medical_knowledge_base", symptoms)
        if "lab_result_interpreter" in tools_to_use and lab_results:
            tool_outputs["lab_result_interpreter"] = self._execute_tool("lab_result_interpreter", lab_results)
        if "drug_interaction_checker" in tools_to_use and medications:
            tool_outputs["drug_interaction_checker"] = self._execute_tool("drug_interaction_checker", medications)
        # Placeholder for imaging tool
        # if "medical_imaging_analyzer" in tools_to_use:
        #     tool_outputs["medical_imaging_analyzer"] = self._execute_tool("medical_imaging_analyzer", "patient_image_data")

        synthesized_data = self._synthesize_diagnosis(tool_outputs, symptoms, medical_history)
        recommendations_and_plans = self._recommend_actions(synthesized_data)

        final_output = {
            "patient_summary": synthesized_data["summary"],
            "differential_diagnoses": synthesized_data["differential_diagnoses"],
            "recommendations": recommendations_and_plans["overall_recommendations"],
            "suggested_further_tests": recommendations_and_plans["suggested_tests"],
            "recommended_treatment_plans": recommendations_and_plans["recommended_treatment_plans"]
        }

        return final_output


if __name__ == "__main__":
    agent = MedicalDiagnosticAgent()

    # Example 1: Basic symptoms
    print("\n--- Example 1: Basic symptoms ---")
    patient_1_symptoms = ["fever", "cough", "fatigue"]
    patient_1_history = "No significant medical history. Recent travel."
    diagnosis_1 = agent.diagnose_patient(
        symptoms=patient_1_symptoms,
        medical_history=patient_1_history
    )
    for key, value in diagnosis_1.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")

    # Example 2: Symptoms with lab results and medications
    print("\n--- Example 2: Symptoms with lab results and medications ---")
    patient_2_symptoms = ["headache", "sore throat"]
    patient_2_history = "Hypertension, currently on Warfarin."
    patient_2_lab_results = {
        "hemoglobin": 13.5,
        "white_blood_cells": 13.0,
        "glucose": 110
    }
    patient_2_medications = ["Warfarin", "Lisinopril", "Aspirin"]

    diagnosis_2 = agent.diagnose_patient(
        symptoms=patient_2_symptoms,
        medical_history=patient_2_history,
        lab_results=patient_2_lab_results,
        medications=patient_2_medications
    )
    for key, value in diagnosis_2.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")

    # Example 3: Only lab results and medications
    print("\n--- Example 3: Only lab results and medications ---")
    patient_3_symptoms = []
    patient_3_history = "History of elevated cholesterol."
    patient_3_lab_results = {
        "hemoglobin": 15.0,
        "white_blood_cells": 8.0,
        "glucose": 125
    }
    patient_3_medications = ["Simvastatin", "Amlodipine"]

    diagnosis_3 = agent.diagnose_patient(
        symptoms=patient_3_symptoms,
        medical_history=patient_3_history,
        lab_results=patient_3_lab_results,
        medications=patient_3_medications
    )
    for key, value in diagnosis_3.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")