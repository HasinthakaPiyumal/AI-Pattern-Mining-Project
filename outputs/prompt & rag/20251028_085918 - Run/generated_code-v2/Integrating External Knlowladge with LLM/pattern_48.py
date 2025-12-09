import json

# Part 1: Medical Knowledge Graph (MKG) - Data Model and Storage
medical_knowledge_graph = {
    "diseases": {
        "Common Cold": {
            "symptoms": ["runny nose", "sore throat", "cough", "sneezing"],
            "treatments": ["rest", "fluids", "over-the-counter medication"]
        },
        "Influenza": {
            "symptoms": ["fever", "body aches", "fatigue", "cough", "sore throat"],
            "treatments": ["antiviral drugs", "rest", "fluids"]
        },
        "Diabetes Type 2": {
            "symptoms": ["frequent urination", "increased thirst", "fatigue", "blurred vision"],
            "treatments": ["diet control", "exercise", "medication", "insulin"]
        },
        "Hypertension": {
            "symptoms": ["headaches", "shortness of breath", "nosebleeds"], # Often asymptomatic
            "treatments": ["lifestyle changes", "antihypertensive medication"]
        },
        "Migraine": {
            "symptoms": ["severe headache", "nausea", "sensitivity to light", "sensitivity to sound"],
            "treatments": ["pain relievers", "triptans", "lifestyle adjustments"]
        }
    },
    "drugs": {
        "Ibuprofen": {
            "side_effects": ["stomach upset", "heartburn"],
            "interactions": {"Warfarin": "increased bleeding risk"}
        },
        "Paracetamol": {
            "side_effects": ["liver damage (high doses)"]
        },
        "Amoxicillin": {
            "side_effects": ["nausea", "diarrhea", "rash"],
            "interactions": {"Methotrexate": "increased methotrexate toxicity"}
        },
        "Metformin": {
            "side_effects": ["nausea", "diarrhea", "lactic acidosis"],
            "interactions": {"Iodinated contrast": "acute kidney injury"}
        },
        "Lisinopril": {
            "side_effects": ["cough", "dizziness", "fatigue"],
            "interactions": {"Potassium supplements": "hyperkalemia"}
        }
    },
    "symptoms_to_diseases": {
        "runny nose": ["Common Cold"],
        "sore throat": ["Common Cold", "Influenza"],
        "cough": ["Common Cold", "Influenza"],
        "sneezing": ["Common Cold"],
        "fever": ["Influenza"],
        "body aches": ["Influenza"],
        "fatigue": ["Influenza", "Diabetes Type 2", "Hypertension", "Lisinopril side effect"],
        "frequent urination": ["Diabetes Type 2"],
        "increased thirst": ["Diabetes Type 2"],
        "blurred vision": ["Diabetes Type 2"],
        "headaches": ["Hypertension", "Migraine"],
        "shortness of breath": ["Hypertension"],
        "nosebleeds": ["Hypertension"],
        "severe headache": ["Migraine"],
        "nausea": ["Migraine", "Amoxicillin side effect", "Metformin side effect"],
        "sensitivity to light": ["Migraine"],
        "sensitivity to sound": ["Migraine"]
    }
}

# Part 2: KG Interaction Layer (Tools)
class KGTools:
    @staticmethod
    def get_symptoms_for_disease(disease: str) -> str:
        disease_info = medical_knowledge_graph["diseases"].get(disease)
        if disease_info and "symptoms" in disease_info:
            return f"Symptoms for {disease}: {', '.join(disease_info['symptoms'])}"
        return f"No symptom information found for {disease}."

    @staticmethod
    def get_treatments_for_disease(disease: str) -> str:
        disease_info = medical_knowledge_graph["diseases"].get(disease)
        if disease_info and "treatments" in disease_info:
            return f"Treatments for {disease}: {', '.join(disease_info['treatments'])}"
        return f"No treatment information found for {disease}."

    @staticmethod
    def get_side_effects_for_drug(drug: str) -> str:
        drug_info = medical_knowledge_graph["drugs"].get(drug)
        if drug_info and "side_effects" in drug_info:
            return f"Side effects for {drug}: {', '.join(drug_info['side_effects'])}"
        return f"No side effect information found for {drug}."

    @staticmethod
    def get_drug_interactions(drug1: str, drug2: str) -> str:
        drug1_info = medical_knowledge_graph["drugs"].get(drug1)
        if drug1_info and "interactions" in drug1_info:
            interaction = drug1_info["interactions"].get(drug2)
            if interaction:
                return f"Interaction between {drug1} and {drug2}: {interaction}"
        drug2_info = medical_knowledge_graph["drugs"].get(drug2)
        if drug2_info and "interactions" in drug2_info:
            interaction = drug2_info["interactions"].get(drug1)
            if interaction:
                return f"Interaction between {drug2} and {drug1}: {interaction}"
        return f"No known interaction found between {drug1} and {drug2}."

    @staticmethod
    def find_diseases_by_symptom(symptom: str) -> str:
        symptom = symptom.lower()
        diseases = medical_knowledge_graph["symptoms_to_diseases"].get(symptom)
        if diseases:
            return f"Diseases associated with {symptom}: {', '.join(diseases)}"
        return f"No diseases found for symptom: {symptom}."

# Part 3: LLM Agent (Simulated for demonstration)
class LLMAgent:
    def __init__(self, tools):
        self.tools = tools

    def _parse_query_for_tool(self, query: str) -> tuple:
        query = query.lower()
        if "symptoms of" in query:
            disease = query.split("symptoms of ", 1)[1].strip().title()
            return "get_symptoms_for_disease", [disease]
        elif "treatments for" in query:
            disease = query.split("treatments for ", 1)[1].strip().title()
            return "get_treatments_for_disease", [disease]
        elif "side effects of" in query:
            drug = query.split("side effects of ", 1)[1].strip().title()
            return "get_side_effects_for_drug", [drug]
        elif "interaction between" in query:
            parts = query.split("interaction between ", 1)[1].split(" and ")
            if len(parts) == 2:
                drug1 = parts[0].strip().title()
                drug2 = parts[1].strip().title()
                return "get_drug_interactions", [drug1, drug2]
        elif "diseases associated with" in query or "what diseases cause" in query:
            symptom_phrase = query.split("with ", 1)[1].strip() if "with " in query else query.split("cause ", 1)[1].strip()
            symptom = symptom_phrase.replace("the symptom ", "").strip()
            return "find_diseases_by_symptom", [symptom]
        return None, []

    def run(self, query: str) -> str:
        tool_name, args = self._parse_query_for_tool(query)
        if tool_name and hasattr(self.tools, tool_name):
            tool_func = getattr(self.tools, tool_name)
            try:
                result = tool_func(*args)
                return f"Agent Response: {result}"
            except Exception as e:
                return f"Agent Error: Failed to execute tool {tool_name} with arguments {args}. Error: {e}"
        return "Agent Response: I couldn't understand your query or find a suitable tool to answer it. Please rephrase."

# Part 4: Main Application Flow - Medical Diagnosis Assistant
if __name__ == "__main__":
    print("Medical Diagnosis Assistant - How can I help you today?")
    kg_tools = KGTools()
    agent = LLMAgent(tools=kg_tools)

    while True:
        user_query = input("\nYour query (e.g., 'symptoms of Common Cold', 'treatments for Influenza', 'side effects of Ibuprofen', 'interaction between Ibuprofen and Warfarin', 'diseases associated with runny nose', or 'exit'): ")
        if user_query.lower() == 'exit':
            print("Exiting Medical Diagnosis Assistant. Goodbye!")
            break

        response = agent.run(user_query)
        print(response)
