import re

class MedicalKnowledgeDB:
    def __init__(self):
        self.knowledge = {
            "fever": "A common symptom of infection or inflammation. Often accompanied by chills, sweating, and headache.",
            "cough": "A reflex action to clear the airways of mucus or irritants. Can be dry or productive.",
            "headache": "Pain in the head or face. Can be mild or severe, and caused by various factors.",
            "influenza": "A viral infection that attacks the respiratory system (nose, throat, lungs). Common symptoms include fever, cough, sore throat, muscle aches, and fatigue.",
            "pneumonia": "An infection that inflames air sacs in one or both lungs, which may fill with fluid or pus. Symptoms include cough with phlegm, fever, chills, and difficulty breathing.",
            "paracetamol": "A common pain reliever and fever reducer.",
            "amoxicillin": "An antibiotic used to treat bacterial infections.",
            "sore throat": "Pain or irritation of the throat that often worsens with swallowing. Common causes include viral infections like the common cold or flu."
        }

    def retrieve_info(self, query):
        query = query.lower()
        if query in self.knowledge:
            return self.knowledge[query]
        else:
            return "No specific information found for that query in the medical knowledge database."

class LabResultsInterpreter:
    def analyze_results(self, lab_data):
        anomalies = []
        # Simulate checking for abnormal values (e.g., simplified ranges)
        if "white blood cell count" in lab_data and lab_data["white blood cell count"] > 10.0:
            anomalies.append("High white blood cell count (potential infection).")
        if "hemoglobin" in lab_data and lab_data["hemoglobin"] < 12.0:
            anomalies.append("Low hemoglobin (potential anemia).")
        if "glucose" in lab_data and lab_data["glucose"] > 120:
            anomalies.append("High glucose (potential hyperglycemia/diabetes risk).")

        if anomalies:
            return f"Lab results analysis: Anomalies detected: {', '.join(anomalies)}"
        else:
            return "Lab results analysis: No significant anomalies detected."

class DrugInteractionChecker:
    def __init__(self):
        self.interactions = {
            ("amoxicillin", "warfarin"): "Increased risk of bleeding due to enhanced anticoagulant effect.",
            ("ibuprofen", "lisinopril"): "Reduced effectiveness of lisinopril and increased risk of kidney problems.",
            ("paracetamol", "alcohol"): "Increased risk of liver damage, especially with chronic heavy alcohol use."
        }

    def check_interactions(self, drug1, drug2):
        drug1 = drug1.lower()
        drug2 = drug2.lower()
        if (drug1, drug2) in self.interactions:
            return f"Drug interaction found between {drug1} and {drug2}: {self.interactions[(drug1, drug2)]}"
        elif (drug2, drug1) in self.interactions:
            return f"Drug interaction found between {drug2} and {drug1}: {self.interactions[(drug2, drug1)]}"
        else:
            return f"No known significant interaction between {drug1} and {drug2} found."

class MedicalDiagnosisAssistant:
    def __init__(self):
        self.medical_db = MedicalKnowledgeDB()
        self.lab_interpreter = LabResultsInterpreter()
        self.drug_checker = DrugInteractionChecker()

    def _call_tool(self, tool_name, *args, **kwargs):
        if tool_name == "medical_knowledge":
            return self.medical_db.retrieve_info(*args, **kwargs)
        elif tool_name == "lab_interpreter":
            return self.lab_interpreter.analyze_results(*args, **kwargs)
        elif tool_name == "drug_checker":
            return self.drug_checker.check_interactions(*args, **kwargs)
        else:
            return "Unknown tool requested."

    def diagnose(self, patient_query):
        patient_query = patient_query.lower()

        # --- Tool Orchestration Logic (Simulated LLM Decision Making) ---

        # Check for medical knowledge queries
        if re.search(r"what is|tell me about|info on|symptoms of", patient_query):
            match = re.search(r"what is (.*?)(?:\?|$)|tell me about (.*?)(?:\?|$)|info on (.*?)(?:\?|$)|symptoms of (.*?)(?:\?|$)", patient_query)
            if match:
                query_term = next((g for g in match.groups() if g), None)
                if query_term:
                    print(f"\nAssistant: Routing to Medical Knowledge Database for '{query_term}'.")
                    return self._call_tool("medical_knowledge", query_term.strip())

        # Check for lab results interpretation queries
        if "lab results" in patient_query or "blood test" in patient_query or "analyze results" in patient_query:
            print("\nAssistant: Routing to Lab Results Interpreter.")
            # In a real scenario, this would extract structured lab data from the query or prompt for it.
            # For this simulation, we'll use a sample data structure.
            sample_lab_data = {
                "white blood cell count": 12.5,  # High
                "hemoglobin": 11.0,           # Low
                "platelet count": 250,        # Normal
                "glucose": 135                # High
            }
            return self._call_tool("lab_interpreter", sample_lab_data)

        # Check for drug interaction queries
        if "drug interaction" in patient_query or "interact with" in patient_query:
            match = re.search(r"(?:drug interaction between|interact with|does) (\w+)(?: and |,) (\w+)", patient_query)
            if match:
                drug1, drug2 = match.groups()
                print(f"\nAssistant: Routing to Drug Interaction Checker for '{drug1}' and '{drug2}'.")
                return self._call_tool("drug_checker", drug1, drug2)

        # Default LLM response if no specific tool is triggered
        print("\nAssistant: Processing with core LLM (no specific tool triggered).")
        return "I can provide information on medical conditions, interpret lab results, or check drug interactions. Please specify your query."

# --- Example Usage ---
if __name__ == "__main__":
    assistant = MedicalDiagnosisAssistant()

    print("--- Medical Diagnosis Assistant ---\n")

    # Example 1: Medical Knowledge Query
    print("Patient: What are the symptoms of influenza?")
    response = assistant.diagnose("What are the symptoms of influenza?")
    print(f"Assistant Response: {response}")

    print("\nPatient: Tell me about fever.")
    response = assistant.diagnose("Tell me about fever.")
    print(f"Assistant Response: {response}")

    print("\nPatient: Info on pneumonia.")
    response = assistant.diagnose("Info on pneumonia.")
    print(f"Assistant Response: {response}")

    # Example 2: Lab Results Interpretation
    print("\nPatient: Can you analyze these lab results?")
    response = assistant.diagnose("Can you analyze these lab results?")
    print(f"Assistant Response: {response}")

    # Example 3: Drug Interaction Check
    print("\nPatient: Check drug interaction between amoxicillin and warfarin.")
    response = assistant.diagnose("Check drug interaction between amoxicillin and warfarin.")
    print(f"Assistant Response: {response}")

    print("\nPatient: Does ibuprofen interact with lisinopril?")
    response = assistant.diagnose("Does ibuprofen interact with lisinopril?")
    print(f"Assistant Response: {response}")

    # Example 4: Query not triggering a specific tool (LLM default)
    print("\nPatient: How can I stay healthy?")
    response = assistant.diagnose("How can I stay healthy?")
    print(f"Assistant Response: {response}")