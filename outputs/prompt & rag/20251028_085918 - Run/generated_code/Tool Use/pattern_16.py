class MedicalKnowledgeBase:
    """Simulates a medical knowledge base for condition lookup."""
    def __init__(self):
        self.knowledge = {
            "fever": ["influenza", "common cold", "bacterial infection"],
            "cough": ["bronchitis", "common cold", "asthma"],
            "headache": ["migraine", "tension headache", "sinusitis"],
            "stomach pain": ["gastritis", "appendicitis", "food poisoning"]
        }

    def get_possible_conditions(self, symptoms: list) -> list:
        found_conditions = set()
        for symptom in symptoms:
            if symptom in self.knowledge:
                found_conditions.update(self.knowledge[symptom])
        return sorted(list(found_conditions))

class DrugInteractionChecker:
    """Simulates a drug interaction database."""
    def __init__(self):
        self.interactions = {
            ("ibuprofen", "warfarin"): "Increased bleeding risk",
            ("paracetamol", "alcohol"): "Increased liver damage risk"
        }

    def check_interactions(self, drugs: list) -> dict:
        results = {}
        if len(drugs) < 2:
            return results
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                drug1, drug2 = sorted([drugs[i], drugs[j]])  # Ensure consistent order
                if (drug1, drug2) in self.interactions:
                    results[f"{drug1} + {drug2}"] = self.interactions[(drug1, drug2)]
        return results

class HallucinationDetector:
    """Placeholder for a sophisticated AST-based hallucination detection.
    In a real system, this would analyze the output for logical consistency
    and medical accuracy using structured knowledge and AST parsing."""
    def detect(self, text: str, context: dict) -> bool:
        print(f"[HallucinationDetector] Analyzing output for hallucinations...")
        # Simple heuristic: if a diagnosis is too vague without supporting symptoms
        if "unknown" in text.lower() and not context.get("symptoms"): # Example simple check
            return True
        # More complex logic involving AST parsing and medical ontologies would go here
        return False

class LLMController:
    """ Orchestrates medical tools based on patient input, simulating an LLM agent."""
    def __init__(self):
        self.knowledge_base = MedicalKnowledgeBase()
        self.drug_checker = DrugInteractionChecker()
        self.hallucination_detector = HallucinationDetector()
        self.tools = {
            "medical_knowledge_base": self.knowledge_base.get_possible_conditions,
            "drug_interaction_checker": self.drug_checker.check_interactions
        }

    def _decide_tools(self, patient_input: dict) -> list:
        """Simulates the LLM's decision-making process for tool selection."""
        tools_to_use = []
        if "symptoms" in patient_input and patient_input["symptoms"]:
            tools_to_use.append("medical_knowledge_base")
        if "medications" in patient_input and patient_input["medications"] and len(patient_input["medications"]) > 1:
            tools_to_use.append("drug_interaction_checker")
        return tools_to_use

    def diagnose(self, patient_input: dict) -> dict:
        """Processes patient input using orchestrated tools and provides a diagnosis/recommendation."""
        print(f"[LLMController] Receiving patient input: {patient_input}")
        diagnosis_report = {"patient_input": patient_input, "findings": []}

        # Tool Understanding & In-Context Learning for Tool Use (simulated by _decide_tools)
        selected_tools = self._decide_tools(patient_input)
        print(f"[LLMController] Selected tools: {selected_tools}")

        if "medical_knowledge_base" in selected_tools:
            symptoms = patient_input.get("symptoms", [])
            if symptoms:
                print(f"[LLMController] Using MedicalKnowledgeBase for symptoms: {symptoms}")
                conditions = self.tools["medical_knowledge_base"](symptoms)
                diagnosis_report["findings"].append({"type": "possible_conditions", "data": conditions})
            else:
                print("[LLMController] No symptoms provided for medical knowledge base tool.")

        if "drug_interaction_checker" in selected_tools:
            medications = patient_input.get("medications", [])
            if medications:
                print(f"[LLMController] Using DrugInteractionChecker for medications: {medications}")
                interactions = self.tools["drug_interaction_checker"](medications)
                if interactions:
                    diagnosis_report["findings"].append({"type": "drug_interactions", "data": interactions})
                else:
                    diagnosis_report["findings"].append({"type": "drug_interactions", "data": "No significant interactions found."})
            else:
                print("[LLMController] No medications provided for drug interaction checker tool.")

        # Simulate Formalism-Enhanced Reasoning & Tool-Integrated Reasoning Loop
        # In a real system, the LLM would analyze findings and refine diagnosis iteratively.
        final_diagnosis = "Based on the provided information and tool outputs: "
        if any(f["type"] == "possible_conditions" for f in diagnosis_report["findings"]):
            cond_data = [f["data"] for f in diagnosis_report["findings"] if f["type"] == "possible_conditions"][0]
            if cond_data:
                final_diagnosis += f"Possible conditions include: {', '.join(cond_data)}. "
            else:
                final_diagnosis += "No specific conditions identified based on symptoms. "

        if any(f["type"] == "drug_interactions" for f in diagnosis_report["findings"]):
            interaction_data = [f["data"] for f in diagnosis_report["findings"] if f["type"] == "drug_interactions"][0]
            if interaction_data and interaction_data != "No significant interactions found.":
                final_diagnosis += f"Important drug interactions: {interaction_data}. "
            else:
                final_diagnosis += "No significant drug interactions detected. "
        
        if not diagnosis_report["findings"]:
            final_diagnosis += "Insufficient information to provide a diagnosis or recommendation."

        diagnosis_report["summary_diagnosis"] = final_diagnosis

        # AST-based Hallucination Detection
        if self.hallucination_detector.detect(final_diagnosis, patient_input):
            diagnosis_report["alert"] = "Potential hallucination detected in summary. Review required."
        
        return diagnosis_report

if __name__ == "__main__":
    controller = LLMController()

    print("\n--- Scenario 1: Basic Symptom Diagnosis ---")
    patient_data_1 = {
        "symptoms": ["fever", "cough"],
        "patient_id": "P001"
    }
    result_1 = controller.diagnose(patient_data_1)
    print("\nDiagnosis Report 1:")
    print(result_1)

    print("\n--- Scenario 2: Drug Interaction Check ---")
    patient_data_2 = {
        "medications": ["ibuprofen", "warfarin"],
        "patient_id": "P002"
    }
    result_2 = controller.diagnose(patient_data_2)
    print("\nDiagnosis Report 2:")
    print(result_2)

    print("\n--- Scenario 3: Symptoms and Medications ---")
    patient_data_3 = {
        "symptoms": ["headache", "fever"],
        "medications": ["paracetamol", "ibuprofen"],
        "patient_id": "P003"
    }
    result_3 = controller.diagnose(patient_data_3)
    print("\nDiagnosis Report 3:")
    print(result_3)

    print("\n--- Scenario 4: No Relevant Input ---")
    patient_data_4 = {
        "patient_id": "P004"
    }
    result_4 = controller.diagnose(patient_data_4)
    print("\nDiagnosis Report 4:")
    print(result_4)

    print("\n--- Scenario 5: Detecting a simple 'hallucination' (conceptual) ---")
    patient_data_5 = {
        "symptoms": [], # No symptoms to base a diagnosis on
        "patient_id": "P005"
    }
    # Manually override the summary to trigger simple hallucination check for demo
    controller_for_hallucination = LLMController()
    result_5 = controller_for_hallucination.diagnose(patient_data_5)
    # Simulate a potentially hallucinated summary for demonstration
    if not result_5["findings"]:
        result_5["summary_diagnosis"] = "Based on limited information, the patient likely has an unknown rare condition that requires immediate surgery."
        if controller_for_hallucination.hallucination_detector.detect(result_5["summary_diagnosis"], patient_data_5):
            result_5["alert"] = "Potential hallucination detected in summary. Review required."
    
    print("\nDiagnosis Report 5 (Hallucination Demo):")
    print(result_5)
