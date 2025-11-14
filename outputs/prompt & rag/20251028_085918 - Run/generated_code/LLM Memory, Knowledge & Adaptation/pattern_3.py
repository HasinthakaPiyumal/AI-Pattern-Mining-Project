import uuid
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

# --- LLM Service (Mock) ---
class LLMService:
    def __init__(self, model_name="mock_llm"):
        self.model_name = model_name
        # In a real application, you would load your LLM here, e.g., using transformers
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, prompt: str, complexity_level: str) -> str:
        """
        Generates a response from the LLM.
        In a real scenario, complexity_level could influence model selection,
        decoding strategies, or prompt engineering.
        """
        if complexity_level == "simple":
            return f"LLM (Simple Mode) processed: {prompt}\n\nProviding a straightforward answer based on the context."
        elif complexity_level == "complex":
            return f"LLM (Complex Mode) processed: {prompt}\n\nConducting an in-depth analysis based on the context."
        else:
            return f"LLM (Default Mode) processed: {prompt}\n\nStandard response based on the context."

# --- Query Classifier (Mock) ---
class QueryClassifier:
    def __init__(self):
        # In a real application, load a pre-trained model (e.g., scikit-learn or a small BERT)
        pass

    def classify_query(self, query: str) -> str:
        """
        Classifies the complexity of a given query.
        For demonstration, we'll use a simple keyword-based classification.
        """
        query_lower = query.lower()
        if "differential diagnosis" in query_lower or "complex case" in query_lower or "unusual symptoms" in query_lower or "explain mechanism" in query_lower:
            return "complex"
        elif "what is" in query_lower or "symptoms of" in query_lower or "simple explanation" in query_lower or "definition of" in query_lower:
            return "simple"
        else:
            return "moderate"

# --- Memory Manager ---
class ShortTermMemory:
    def __init__(self, max_interactions: int = 5):
        self.memory: List[Dict[str, str]] = []
        self.max_interactions = max_interactions

    def add_interaction(self, user_query: str, assistant_response: str):
        self.memory.append({"user": user_query, "assistant": assistant_response})
        if len(self.memory) > self.max_interactions:
            self.memory.pop(0) # Remove oldest interaction

    def get_context(self) -> str:
        context = []
        for interaction in self.memory:
            context.append(f"User: {interaction['user']}")
            context.append(f"Assistant: {interaction['assistant']}")
        return "\n".join(context)

    def clear(self):
        self.memory = []

class LongTermMemory:
    def __init__(self, vector_db_client=None):
        # In a real application, `vector_db_client` would be an initialized Weaviate/Pinecone client
        # For this example, we'll simulate a simple in-memory store
        self.vector_db_client = vector_db_client
        self.knowledge_base: Dict[str, str] = {} # Mock: Maps ID to content for retrieval
        self.embeddings: Dict[str, List[float]] = {} # Mock: Maps ID to embedding (not actually used for similarity here)

        # Mock some medical knowledge
        self._add_mock_knowledge()

    def _generate_embedding(self, text: str) -> List[float]:
        # In a real application, use sentence-transformers or similar
        # For now, a simple mock, not used for actual similarity search in this example
        return [hash(text) % 1000 / 1000.0] * 768 # Placeholder embedding

    def _add_mock_knowledge(self):
        medical_facts = [
            "Symptoms of diabetes include frequent urination, increased thirst, and unexplained weight loss. It is a metabolic disorder.",
            "Hypertension, or high blood pressure, is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Management often involves lifestyle changes and medication.",
            "A differential diagnosis is a systematic diagnostic method used to identify the presence of a disease where multiple alternatives are possible, based on a patient's symptoms and signs.",
            "Common antibiotics include penicillin, amoxicillin, and azithromycin. They are used to treat bacterial infections and are ineffective against viral infections.",
            "MRI (Magnetic Resonance Imaging) uses a powerful magnetic field, radio waves and a computer to produce detailed pictures of organs, soft tissues, bone and virtually all other internal body structures, without using ionizing radiation.",
            "Type 2 diabetes is often managed with diet, exercise, and oral medications. Insulin may be required in some cases."
        ]
        for fact in medical_facts:
            item_id = str(uuid.uuid4())
            self.knowledge_base[item_id] = fact
            self.embeddings[item_id] = self._generate_embedding(fact) # Store but not directly used for search

    def retrieve_relevant_knowledge(self, query: str, top_k: int = 2) -> List[str]:
        """
        Retrieves relevant knowledge from the long-term memory based on the query.
        In a real application, this would query the vector database using embeddings.
        For this mock, we use keyword matching.
        """
        query_lower = query.lower()
        relevant_items = []

        # Simple keyword matching for mock retrieval
        for item_id, fact in self.knowledge_base.items():
            fact_lower = fact.lower()
            score = 0
            for word in query_lower.split():
                if word in fact_lower:
                    score += 1
            if score > 0:
                relevant_items.append((score, fact))
        
        relevant_items.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in relevant_items[:top_k]]

# --- FastAPI Application ---
app = FastAPI(title="Smart Medical Assistant API")

# Initialize components
llm_service = LLMService()
query_classifier = QueryClassifier()
short_term_memory = ShortTermMemory(max_interactions=5)
long_term_memory = LongTermMemory() # Pass actual client if using Weaviate/Pinecone

class MedicalQuery(BaseModel):
    query: str
    patient_id: str = "anonymous" # Placeholder for future patient-specific LTM

class AssistantResponse(BaseModel):
    response: str
    context_used: str
    memory_type_accessed: List[str]
    query_complexity: str

@app.post("/diagnose", response_model=AssistantResponse)
async def diagnose_patient(medical_query: MedicalQuery):
    user_query = medical_query.query

    # 1. Classify Query Complexity
    complexity_level = query_classifier.classify_query(user_query)

    # 2. Retrieve from Long-Term Memory (RAG)
    # This step is crucial for factual consistency and external knowledge
    relevant_long_term_knowledge = long_term_memory.retrieve_relevant_knowledge(user_query, top_k=3)
    ltm_context = ""
    memory_types_accessed_current = []
    if relevant_long_term_knowledge:
        ltm_context = "Relevant Medical Knowledge:\n" + "\n".join(relevant_long_term_knowledge)
        memory_types_accessed_current.append("Long-Term Memory")

    # 3. Retrieve from Short-Term Memory (Conversation History)
    stm_context = short_term_memory.get_context()
    if stm_context:
        memory_types_accessed_current.append("Short-Term Memory")

    # Combine all contexts for the LLM
    full_context_for_llm = ""
    if ltm_context:
        full_context_for_llm += ltm_context + "\n\n"
    if stm_context:
        full_context_for_llm += "Conversation History:\n" + stm_context + "\n\n"
    
    # Construct the final prompt for the LLM
    # The actual prompt engineering would be more sophisticated based on complexity_level
    llm_prompt = f"Given the following context:\n---\n{full_context_for_llm.strip()}\n---\nUser's current medical query: {user_query}\n\nBased on the context and your medical knowledge, provide a comprehensive and accurate response. Focus on patient safety and accurate information. "

    # 4. Generate LLM Response
    llm_response = llm_service.generate_response(llm_prompt, complexity_level)

    # 5. Update Short-Term Memory
    short_term_memory.add_interaction(user_query, llm_response)

    return AssistantResponse(
        response=llm_response,
        context_used=full_context_for_llm.strip(),
        memory_type_accessed=memory_types_accessed_current,
        query_complexity=complexity_level
    )

@app.post("/clear_session")
async def clear_session():
    short_term_memory.clear()
    return {"message": "Short-term memory cleared for the current session."}

# To run this application:
# 1. Save the code as smart_medical_assistant.py
# 2. Install dependencies: pip install fastapi uvicorn pydantic
# 3. Run from your terminal: uvicorn smart_medical_assistant:app --reload
# 4. Access the API documentation at http://127.0.0.1:8000/docs