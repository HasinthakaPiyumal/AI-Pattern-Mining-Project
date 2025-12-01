class SymptomToDiseaseKGRRetrievalModule:
    def __init__(self):
        self.knowledge_graph = {
            "Common Cold": {
                "symptoms": ["runny nose", "sore throat", "cough", "fever"],
                "description": "A viral infection of your nose and throat (upper respiratory tract).",
                "treatment": "Rest, fluids, over-the-counter cold medicines."
            },
            "Influenza": {
                "symptoms": ["fever", "body aches", "chills", "fatigue", "cough", "sore throat"],
                "description": "A contagious respiratory illness caused by influenza viruses.",
                "treatment": "Antiviral drugs, rest, fluids."
            },
            "Migraine": {
                "symptoms": ["severe headache", "nausea", "sensitivity to light", "sensitivity to sound"],
                "description": "A type of headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head.",
                "treatment": "Pain relievers, triptans, CGRP inhibitors."
            },
            "Allergies": {
                "symptoms": ["sneezing", "itchy eyes", "runny nose", "skin rash"],
                "description": "A condition in which the immune system reacts abnormally to a foreign substance.",
                "treatment": "Antihistamines, decongestants, immunotherapy."
            }
        }

    def retrieve_knowledge(self, symptoms: list) -> str:
        found_diseases = []
        for disease, info in self.knowledge_graph.items():
            if any(symptom.lower() in [s.lower() for s in info["symptoms"]] for symptom in symptoms):
                found_diseases.append(disease)
        
        if found_diseases:
            knowledge_str = "Retrieved Medical Knowledge:\n"
            for disease_name in found_diseases:
                info = self.knowledge_graph[disease_name]
                knowledge_str += f"  Disease: {disease_name}\n"
                knowledge_str += f"    Symptoms: {', '.join(info['symptoms'])}\n"
                knowledge_str += f"    Description: {info['description']}\n"
                knowledge_str += f"    Treatment: {info['treatment']}\n"
            return knowledge_str
        else:
            return "No direct medical knowledge found for the given symptoms in our database."

class MockLLM:
    def generate_response(self, prompt: str) -> str:
        response = f"LLM Processed Prompt:\n--------------------\n{prompt}\n--------------------\nLLM Diagnosis Suggestion: Based on the provided information and symptoms, further medical consultation is recommended for a definitive diagnosis. However, considering the context, potential conditions include... (This is a simulated LLM response based on the input prompt and retrieved knowledge.)"
        return response

class MedicalDiagnosticAssistant:
    def __init__(self, kg_module: SymptomToDiseaseKGRRetrievalModule, llm_client: MockLLM):
        self.kg_module = kg_module
        self.llm_client = llm_client

    def diagnose(self, patient_symptoms: list) -> str:
        retrieved_knowledge = self.kg_module.retrieve_knowledge(patient_symptoms)

        prompt = f"Patient Symptoms: {', '.join(patient_symptoms)}\n\n{retrieved_knowledge}\n\nBased on the patient's symptoms and the retrieved medical knowledge, provide a diagnostic suggestion and potential next steps."

        llm_response = self.llm_client.generate_response(prompt)
        return llm_response

if __name__ == "__main__":
    # Initialize the Plug-and-Play KGR Module
    kg_retrieval_module = SymptomToDiseaseKGRRetrievalModule()

    # Initialize the Simulated LLM
    mock_llm = MockLLM()

    # Initialize the Medical Diagnostic Assistant with the KGR module and LLM
    assistant = MedicalDiagnosticAssistant(kg_retrieval_module, mock_llm)

    # Example 1: Symptoms with a direct match
    print("\n--- Patient 1: Common Cold Symptoms ---")
    patient_symptoms_1 = ["runny nose", "sore throat", "cough"]
    diagnosis_1 = assistant.diagnose(patient_symptoms_1)
    print(diagnosis_1)

    # Example 2: Symptoms with another direct match
    print("\n--- Patient 2: Migraine Symptoms ---")
    patient_symptoms_2 = ["severe headache", "nausea", "sensitivity to light"]
    diagnosis_2 = assistant.diagnose(patient_symptoms_2)
    print(diagnosis_2)

    # Example 3: Symptoms with no direct match in the simplified KG
    print("\n--- Patient 3: Uncommon Symptoms ---")
    patient_symptoms_3 = ["abdominal pain", "jaundice"]
    diagnosis_3 = assistant.diagnose(patient_symptoms_3)
    print(diagnosis_3)

    # Example 4: Symptoms that could hint at multiple conditions (simulated multi-match)
    print("\n--- Patient 4: Fever and Cough ---")
    patient_symptoms_4 = ["fever", "cough"]
    diagnosis_4 = assistant.diagnose(patient_symptoms_4)
    print(diagnosis_4)