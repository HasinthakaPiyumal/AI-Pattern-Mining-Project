import numpy as np

class MedicalDiagnosticAssistant:
    def __init__(self):
        pass

    def _call_llm(self, query, tool_outputs=None):
        llm_response = f"LLM processing query: '{query}'."
        if tool_outputs:
            llm_response += " Based on tool outputs: "
            for tool, output in tool_outputs.items():
                llm_response += f"[{tool}: {output}] "
        return llm_response

    def _medical_knowledge_retrieval(self, topic):
        if "diabetes guidelines" in topic.lower():
            return "Retrieved: Latest ADA guidelines recommend Metformin as first-line for Type 2 Diabetes."
        elif "hypertension research" in topic.lower():
            return "Retrieved: Recent studies show ARBs effective in blood pressure management."
        return f"Retrieved general medical facts about {topic}."

    def _medical_calculator(self, calculation_type, values):
        if calculation_type == "drug_dosage":
            if len(values) == 2:
                weight_kg = values[0]
                dose_mg_per_kg = values[1]
                total_dose = np.round(weight_kg * dose_mg_per_kg, 2)
                return f"Calculated drug dosage: {total_dose} mg."
            return "Invalid values for drug dosage calculation."
        elif calculation_type == "risk_score":
            if len(values) == 3:
                age, cholesterol, bp = values
                risk = np.round((age * 0.1) + (cholesterol * 0.05) + (bp * 0.02), 2)
                return f"Calculated cardiovascular risk score: {risk}."
            return "Invalid values for risk score calculation."
        return f"Performed a calculation of type {calculation_type} with values {values}."

    def _medical_image_analysis(self, image_description):
        if "chest x-ray" in image_description.lower():
            return "Image Analysis: Chest X-ray shows no acute cardiopulmonary abnormalities."
        elif "mri brain" in image_description.lower():
            return "Image Analysis: MRI Brain indicates small ischemic changes in white matter."
        return f"Image Analysis report for {image_description}: No specific findings detected."

    def _drug_interaction_database(self, drugs):
        drugs_lower = [d.lower() for d in drugs]
        if "warfarin" in drugs_lower and "ibuprofen" in drugs_lower:
            return "Warning: Potential increased risk of bleeding with Warfarin and Ibuprofen."
        elif "simvastatin" in drugs_lower and "grapefruit" in drugs_lower:
            return "Warning: Grapefruit can increase Simvastatin levels, increasing risk of side effects."
        return f"No significant drug interactions found for {', '.join(drugs)}."

    def diagnose_patient(self, patient_query):
        print(f"\nDoctor's Query: {patient_query}")

        # Step 1: LLM initial processing
        llm_initial_thought = self._call_llm(patient_query)
        print(f"LLM Initial Thought: {llm_initial_thought}")

        tool_outputs = {}

        # Step 2: Orchestrate tool calls based on keywords
        if any(keyword in patient_query.lower() for keyword in ["guidelines", "research", "disease facts"]):
            retrieval_output = self._medical_knowledge_retrieval(patient_query)
            tool_outputs["MedicalKnowledgeRetrieval"] = retrieval_output
            print(f"Tool Call: Medical Knowledge Retrieval -> {retrieval_output}")

        if any(keyword in patient_query.lower() for keyword in ["calculate dosage", "risk score", "gfr", "numerical"]):
            if "calculate dosage" in patient_query.lower():
                # Mocking values for demonstration
                calc_output = self._medical_calculator("drug_dosage", [70, 0.5]) # 70kg, 0.5mg/kg
            elif "risk score" in patient_query.lower():
                # Mocking values for demonstration: age, cholesterol, bp
                calc_output = self._medical_calculator("risk_score", [55, 200, 130])
            else:
                calc_output = self._medical_calculator("general_calculation", [10, 20, 30])
            tool_outputs["MedicalCalculator"] = calc_output
            print(f"Tool Call: Medical Calculator -> {calc_output}")

        if any(keyword in patient_query.lower() for keyword in ["x-ray", "mri", "ct scan", "image analysis"]):
            image_output = self._medical_image_analysis(patient_query)
            tool_outputs["MedicalImageAnalysis"] = image_output
            print(f"Tool Call: Medical Image Analysis -> {image_output}")

        if any(keyword in patient_query.lower() for keyword in ["drug interaction", "medication"]):
            # Mocking drug names for demonstration
            if "warfarin" in patient_query.lower() and "ibuprofen" in patient_query.lower():
                drug_output = self._drug_interaction_database(["Warfarin", "Ibuprofen"])
            elif "simvastatin" in patient_query.lower() and "grapefruit" in patient_query.lower():
                drug_output = self._drug_interaction_database(["Simvastatin", "Grapefruit"])
            else:
                drug_output = self._drug_interaction_database(["DrugA", "DrugB"])
            tool_outputs["DrugInteractionDatabase"] = drug_output
            print(f"Tool Call: Drug Interaction Database -> {drug_output}")

        # Step 3: LLM final response with tool outputs
        final_diagnosis = self._call_llm(patient_query, tool_outputs)
        print(f"\nLLM Final Diagnosis/Recommendation: {final_diagnosis}")
        return final_diagnosis

if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()

    print("--- Scenario 1: Factual Query ---")
    assistant.diagnose_patient("What are the latest guidelines for managing Type 2 Diabetes?")

    print("\n--- Scenario 2: Numerical Calculation and Medical Image ---")
    assistant.diagnose_patient("Patient with a chest X-ray result, please calculate a drug dosage for a 70kg patient (0.5mg/kg).")

    print("\n--- Scenario 3: Drug Interaction Check and Risk Score ---")
    assistant.diagnose_patient("Patient is on Warfarin and might take Ibuprofen. Also, calculate their cardiovascular risk score (age 60, cholesterol 220, BP 140).")

    print("\n--- Scenario 4: General Query ---")
    assistant.diagnose_patient("Patient presents with headache and fatigue. What are potential differential diagnoses?")

    print("\n--- Scenario 5: Specific Drug-Food Interaction ---")
    assistant.diagnose_patient("Are there any drug interactions if a patient on Simvastatin consumes grapefruit?")
