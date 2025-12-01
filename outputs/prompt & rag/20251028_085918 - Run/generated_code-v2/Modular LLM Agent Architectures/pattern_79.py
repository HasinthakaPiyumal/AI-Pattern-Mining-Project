class MedicalKnowledgeModule:
    def __init__(self):
        self.medical_knowledge = {
            "fever": {"symptoms": ["high temperature", "chills", "headache"], "diagnoses": [{"name": "Flu", "likelihood": 0.7}, {"name": "Common Cold", "likelihood": 0.3}], "facts": "Fever is a temporary increase in your body temperature, often due to an illness.", "next_steps": ["Rest", "Drink fluids", "Consult a doctor if severe"]},
            "cough": {"symptoms": ["sore throat", "hoarseness"], "diagnoses": [{"name": "Common Cold", "likelihood": 0.6}, {"name": "Bronchitis", "likelihood": 0.4}], "facts": "A cough is a reflex that helps clear your airways of irritants and mucus.", "next_steps": ["Cough medicine", "Hydration", "Avoid irritants"]},
            "headache": {"symptoms": ["pain in head", "nausea"], "diagnoses": [{"name": "Tension Headache", "likelihood": 0.8}, {"name": "Migraine", "likelihood": 0.2}], "facts": "Headaches are a common condition that most people will experience many times in their lives.", "next_steps": ["Pain relievers", "Rest in a dark room"]}
        }

    def get_diagnoses(self, symptoms: list) -> dict:
        potential_diagnoses = {}
        for symptom in symptoms:
            for condition, data in self.medical_knowledge.items():
                if symptom in data["symptoms"]:
                    for diagnosis_info in data["diagnoses"]:
                        diag_name = diagnosis_info["name"]
                        if diag_name not in potential_diagnoses:
                            potential_diagnoses[diag_name] = {"likelihood": 0, "symptoms": []}
                        potential_diagnoses[diag_name]["likelihood"] += diagnosis_info["likelihood"]
                        potential_diagnoses[diag_name]["symptoms"].append(symptom)
        return potential_diagnoses

    def get_medical_facts(self, query: str) -> str:
        for condition, data in self.medical_knowledge.items():
            if condition in query.lower() or any(s in query.lower() for s in data["symptoms"]):
                return data["facts"]
        return "No specific medical facts found for your query."

    def suggest_next_steps(self, diagnosis: str) -> list:
        for condition, data in self.medical_knowledge.items():
            for diag_info in data["diagnoses"]:
                if diag_info["name"].lower() == diagnosis.lower():
                    return data["next_steps"]
        return ["Consult a medical professional for advice."]

def simulate_llm_response(augmented_context: str) -> str:
    return f"Based on the medical information, here is a detailed response:\n{augmented_context}\n\nPlease remember this is a simulated response and not a substitute for professional medical advice."

class MedicalDiagnosisSystem:
    def __init__(self, medical_module_instance: MedicalKnowledgeModule):
        self.medical_module = medical_module_instance

    def process_user_query(self, user_input: str) -> str:
        # Simple parsing for demonstration
        symptoms = []
        if "symptoms" in user_input.lower():
            # This is a very basic parsing. In a real system, NLP would be used.
            parts = user_input.lower().split("symptoms:")
            if len(parts) > 1:
                symptom_str = parts[1].split(", what could it be?")[0].strip()
                symptoms = [s.strip() for s in symptom_str.split(" and ")]
        else:
             # Fallback for simpler queries, extract keywords
            keywords = ["fever", "cough", "headache"]
            symptoms = [kw for kw in keywords if kw in user_input.lower()]

        general_query = user_input

        # Interact with Medical Module
        diagnoses = self.medical_module.get_diagnoses(symptoms)
        medical_facts = self.medical_module.get_medical_facts(general_query)
        
        next_steps_suggestions = []
        for diag_name in diagnoses.keys():
            steps = self.medical_module.suggest_next_steps(diag_name)
            if steps:
                next_steps_suggestions.append(f"For {diag_name}: {', '.join(steps)}")

        # Augment Context
        augmented_context = f"User Query: {user_input}\n\n"
        if symptoms:
            augmented_context += f"Identified Symptoms: {', '.join(symptoms)}\n"
        if diagnoses:
            augmented_context += "Potential Diagnoses (with likelihood scores and associated symptoms):\n"
            for diag_name, info in diagnoses.items():
                augmented_context += f"- {diag_name} (Likelihood Score: {info['likelihood']:.2f}, Related Symptoms: {', '.join(info['symptoms'])})\n"
        if medical_facts:
            augmented_context += f"Relevant Medical Facts: {medical_facts}\n"
        if next_steps_suggestions:
            augmented_context += "Suggested Next Steps:\n"
            for step_str in next_steps_suggestions:
                augmented_context += f"- {step_str}\n"

        # Simulate LLM Response
        llm_response = simulate_llm_response(augmented_context)
        return llm_response

if __name__ == "__main__":
    # Initialize the Medical Knowledge Module
    medical_module = MedicalKnowledgeModule()

    # Initialize the Medical Diagnosis System with the medical module
    diagnosis_system = MedicalDiagnosisSystem(medical_module)

    # Example User Queries
    query1 = "I have symptoms: fever and headache, what could it be?"
    response1 = diagnosis_system.process_user_query(query1)
    print(f"\n--- User Query 1 ---\n{query1}\n")
    print(f"--- System Response 1 ---\n{response1}\n")

    query2 = "I have a persistent cough. What medical facts are relevant?"
    response2 = diagnosis_system.process_user_query(query2)
    print(f"\n--- User Query 2 ---\n{query2}\n")
    print(f"--- System Response 2 ---\n{response2}\n")

    query3 = "What are the next steps for a common cold?"
    response3 = diagnosis_system.process_user_query(query3)
    print(f"\n--- User Query 3 ---\n{query3}\n")
    print(f"--- System Response 3 ---\n{response3}\n")