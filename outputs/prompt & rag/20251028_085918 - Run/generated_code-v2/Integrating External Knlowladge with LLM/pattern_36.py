import abc

class LLMInterface(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class KGInterface(abc.ABC):
    @abc.abstractmethod
    def query_knowledge(self, query: str) -> dict:
        pass

class ChatGPTAdapter(LLMInterface):
    def __init__(self, api_key: str = "dummy_api_key"):
        self._api_key = api_key

    def generate_response(self, prompt: str) -> str:
        # Simulate an OpenAI API call
        return f"ChatGPT's response to: {prompt}. (Using key: {self._api_key[:5]}...)"

class Llama2Adapter(LLMInterface):
    def __init__(self, model_path: str = "./llama2_model"):
        self._model_path = model_path

    def generate_response(self, prompt: str) -> str:
        # Simulate a Llama2 model inference
        return f"Llama2's response to: {prompt}. (Model from: {self._model_path})"

class MedicalKnowledgeGraphAdapter(KGInterface):
    def __init__(self):
        self._knowledge_base = {
            "symptoms": {
                "fever": ["influenza", "common cold", "malaria"],
                "cough": ["bronchitis", "pneumonia", "common cold"],
                "headache": ["migraine", "tension headache", "flu"],
                "fatigue": ["anemia", "chronic fatigue syndrome", "influenza"]
            },
            "diseases": {
                "influenza": {"symptoms": ["fever", "cough", "fatigue"], "treatment": "Antivirals, rest, fluids"},
                "common cold": {"symptoms": ["cough", "fever"], "treatment": "Rest, fluids, symptom relief"},
                "bronchitis": {"symptoms": ["cough"], "treatment": "Antibiotics (if bacterial), bronchodilators"},
                "migraine": {"symptoms": ["headache"], "treatment": "Pain relievers, triptans"}
            },
            "patients": {
                "P001": {"name": "Alice Smith", "age": 45, "conditions": ["migraine"], "medications": ["sumatriptan"]}
            }
        }

    def query_knowledge(self, query: str) -> dict:
        query_lower = query.lower()
        if "symptoms of" in query_lower:
            disease = query_lower.replace("symptoms of ", "").strip()
            return self._knowledge_base["diseases"].get(disease, {}).get("symptoms", [])
        elif "treatment for" in query_lower:
            condition = query_lower.replace("treatment for ", "").strip()
            return self._knowledge_base["diseases"].get(condition, {}).get("treatment", "No specific treatment found.")
        elif "diseases with symptom" in query_lower:
            symptom = query_lower.replace("diseases with symptom ", "").strip()
            return self._knowledge_base["symptoms"].get(symptom, [])
        elif "patient info for" in query_lower:
            patient_id = query_lower.replace("patient info for ", "").strip().upper()
            return self._knowledge_base["patients"].get(patient_id, {})
        return {}

class SmartMedicalAssistantFramework:
    def __init__(self, llm_adapter: LLMInterface, kg_adapter: KGInterface):
        self._llm = llm_adapter
        self._kg = kg_adapter

    def diagnose(self, symptoms: str) -> str:
        kg_query_result = self._kg.query_knowledge(f"diseases with symptom {symptoms.lower()}")
        if kg_query_result:
            kg_info = f"\nBased on knowledge graph, common diseases for {symptoms} are: {', '.join(kg_query_result)}."
        else:
            kg_info = f"\nKnowledge graph has no specific diseases listed for {symptoms}."
        
        llm_prompt = f"Given the symptoms: {symptoms}. What is a differential diagnosis? {kg_info} Consider the knowledge graph information in your diagnosis."
        llm_response = self._llm.generate_response(llm_prompt)
        return f"Diagnosis Assistant: {llm_response}"

    def recommend_treatment(self, condition: str) -> str:
        kg_treatment = self._kg.query_knowledge(f"treatment for {condition.lower()}")
        
        llm_prompt = f"For the medical condition: {condition}, what are the recommended treatments? Knowledge graph suggests: {kg_treatment}. Consider this in your recommendations."
        llm_response = self._llm.generate_response(llm_prompt)
        return f"Treatment Assistant: {llm_response}"

    def retrieve_patient_info(self, patient_id: str) -> str:
        patient_data = self._kg.query_knowledge(f"patient info for {patient_id}")
        if patient_data:
            info_str = f"Patient ID: {patient_id}\nName: {patient_data.get('name')}\nAge: {patient_data.get('age')}\nConditions: {', '.join(patient_data.get('conditions', []))}\nMedications: {', '.join(patient_data.get('medications', []))}"
        else:
            info_str = f"No patient found with ID: {patient_id}."
        
        llm_prompt = f"Provide a summary of the following patient information: {info_str}. Also, suggest any immediate considerations or questions a doctor might have based on this data."
        llm_response = self._llm.generate_response(llm_prompt)
        return f"Patient Info Assistant: {llm_response}"

# Example Usage:

# 1. Initialize KG Adapter
medical_kg = MedicalKnowledgeGraphAdapter()

# 2. Initialize LLM Adapters (can swap them out)
chatgpt_llm = ChatGPTAdapter(api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
llama2_llm = Llama2Adapter(model_path="/models/llama2-7b-chat")

# 3. Initialize the Smart Medical Assistant with different LLMs
assistant_with_chatgpt = SmartMedicalAssistantFramework(llm_adapter=chatgpt_llm, kg_adapter=medical_kg)
assistant_with_llama2 = SmartMedicalAssistantFramework(llm_adapter=llama2_llm, kg_adapter=medical_kg)

print("--- Using ChatGPT Adapter ---")
print(assistant_with_chatgpt.diagnose("fever and cough"))
print(assistant_with_chatgpt.recommend_treatment("influenza"))
print(assistant_with_chatgpt.retrieve_patient_info("P001"))

print("\n--- Using Llama2 Adapter ---")
print(assistant_with_llama2.diagnose("headache"))
print(assistant_with_llama2.recommend_treatment("migraine"))
print(assistant_with_llama2.retrieve_patient_info("P002"))
