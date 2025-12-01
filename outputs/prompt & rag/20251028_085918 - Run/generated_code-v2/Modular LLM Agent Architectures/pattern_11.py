import os
from typing import Dict, Any, List, Optional

# Mocking external libraries for a self-contained snippet
# In a real application, these would be imported from actual packages

class MockBaseLanguageModel:
    def invoke(self, prompt: str) -> str:
        # Simulate LLM response
        if "diagnosis" in prompt.lower():
            return "Based on the symptoms, a likely diagnosis is Viral Gastroenteritis. Consider recommending rehydration and rest."
        elif "treatment" in prompt.lower():
            return "For Viral Gastroenteritis, treatment typically involves supportive care: oral rehydration, antiemetics if severe nausea, and avoidance of solid foods for a short period."
        else:
            return f"LLM processed: {prompt}"

class MockChatOpenAI(MockBaseLanguageModel):
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4"):
        self.openai_api_key = openai_api_key
        self.model_name = model_name

class MockSentenceTransformerEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Simulate embeddings for text
        return [[0.1 * i for i in range(10)] for _ in texts] # Dummy embeddings

    def embed_query(self, text: str) -> List[float]:
        # Simulate embedding for a query
        return [0.05 * i for i in range(10)] # Dummy embedding

class MockChroma:
    def __init__(self):
        self.store = {}
        self.id_counter = 0

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        if metadatas is None:
            metadatas = [{} for _ in documents]
        for doc, meta in zip(documents, metadatas):
            self.store[self.id_counter] = {"document": doc, "metadata": meta, "embedding": MockSentenceTransformerEmbeddings().embed_documents([doc])[0]}
            self.id_counter += 1

    def similarity_search(self, query: str, k: int = 4) -> List[str]:
        # Simple similarity search based on keywords for demonstration
        query_lower = query.lower()
        results = []
        for doc_id, data in self.store.items():
            if query_lower in data["document"].lower():
                results.append(data["document"])
        return results[:k]


# --- LLM Interface Module ---
class LLMInterface:
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        self.llm = MockChatOpenAI(openai_api_key=api_key, model_name=model_name)

    def generate_response(self, prompt: str) -> str:
        return self.llm.invoke(prompt)


# --- Working Memory Module ---
class WorkingMemory:
    def __init__(self):
        self.patient_context: Dict[str, Any] = {}
        self.session_state: Dict[str, Any] = {}
        self.retrieved_facts: List[str] = []
        self.llm_thoughts: List[str] = []

    def update_patient_context(self, context: Dict[str, Any]):
        self.patient_context.update(context)

    def update_session_state(self, state: Dict[str, Any]):
        self.session_state.update(state)

    def add_retrieved_facts(self, facts: List[str]):
        self.retrieved_facts.extend(facts)

    def add_llm_thought(self, thought: str):
        self.llm_thoughts.append(thought)

    def get_full_context(self) -> str:
        context_str = f"Patient Context: {self.patient_context}\n"
        context_str += f"Session State: {self.session_state}\n"
        context_str += f"Retrieved Medical Facts: {'; '.join(self.retrieved_facts)}\n"
        context_str += f"LLM Internal Thoughts: {'; '.join(self.llm_thoughts)}\n"
        return context_str.strip()


# --- Medical Knowledge Base (Vector Store) ---
class MedicalKnowledgeBase:
    def __init__(self):
        self.vector_store = MockChroma()
        # Pre-populate with some mock medical data
        self.vector_store.add_documents([
            "Viral Gastroenteritis is an inflammation of the stomach and intestines caused by a virus.",
            "Common symptoms of Viral Gastroenteritis include nausea, vomiting, diarrhea, and abdominal cramps.",
            "Treatment for Viral Gastroenteritis is typically supportive, focusing on rehydration.",
            "Appendicitis is an inflammation of the appendix, often requiring surgical removal.",
            "Symptoms of Appendicitis include sudden pain in the lower right abdomen, nausea, vomiting, and fever.",
            "Type 2 Diabetes Mellitus is a chronic condition that affects the way the body processes blood sugar (glucose).",
            "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems."
        ])

    def retrieve_facts(self, query: str, k: int = 4) -> List[str]:
        return self.vector_store.similarity_search(query, k=k)


# --- CDSS Agent (Policy Module & Orchestrator) ---
class CDSSAgent:
    def __init__(self, llm_api_key: str):
        self.llm_interface = LLMInterface(api_key=llm_api_key)
        self.working_memory = WorkingMemory()
        self.knowledge_base = MedicalKnowledgeBase()

    def process_patient_query(self, query: str, patient_info: Dict[str, Any]) -> str:
        # 1. Update working memory with patient context and query
        self.working_memory.update_patient_context(patient_info)
        self.working_memory.update_session_state({"current_query": query})

        # 2. Retrieve relevant facts from the knowledge base
        relevant_facts = self.knowledge_base.retrieve_facts(query, k=5)
        self.working_memory.add_retrieved_facts(relevant_facts)

        # 3. Construct a detailed prompt for the LLM
        full_context = self.working_memory.get_full_context()
        prompt = f"""
        You are a highly experienced medical AI assistant. Your task is to provide evidence-based medical advice and diagnoses based on the provided patient information and medical facts. Avoid making up information.

        {full_context}

        Based on the above patient information and retrieved medical facts, what is a likely diagnosis and what treatment recommendations would you suggest?
        """

        # 4. Call the LLM and get its response
        llm_response = self.llm_interface.generate_response(prompt)
        self.working_memory.add_llm_thought(llm_response)

        # 5. Return the LLM's final recommendation/diagnosis
        return llm_response


# --- Main Application Logic (Example Usage) ---
if __name__ == "__main__":
    # Replace with your actual OpenAI API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY") 

    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("WARNING: Please set your OPENAI_API_KEY environment variable or replace 'YOUR_OPENAI_API_KEY' with your actual key for full functionality.")

    cdss_agent = CDSSAgent(llm_api_key=OPENAI_API_KEY)

    patient_data_1 = {
        "name": "John Doe",
        "age": 35,
        "gender": "Male",
        "symptoms": "Nausea, vomiting, diarrhea for 24 hours, abdominal cramps."
    }
    query_1 = "What is the most likely diagnosis and initial treatment for a patient with acute gastroenteritis symptoms?"
    print(f"--- Patient 1 Query ---\nQuery: {query_1}\nPatient Info: {patient_data_1}")
    response_1 = cdss_agent.process_patient_query(query_1, patient_data_1)
    print(f"CDSS Recommendation: {response_1}\n")

    patient_data_2 = {
        "name": "Jane Smith",
        "age": 50,
        "gender": "Female",
        "symptoms": "Sudden onset severe lower right abdominal pain, fever, loss of appetite."
    }
    query_2 = "Could this be appendicitis? What are the next steps?"
    print(f"--- Patient 2 Query ---\nQuery: {query_2}\nPatient Info: {patient_data_2}")
    response_2 = cdss_agent.process_patient_query(query_2, patient_data_2)
    print(f"CDSS Recommendation: {response_2}\n")