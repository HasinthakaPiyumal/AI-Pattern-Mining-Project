from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import uuid

# Mocking sentence_transformers and chromadb for standalone execution
class MockSentenceTransformer:
    def encode(self, sentences, convert_to_tensor=False):
        return [list(range(768)) for _ in sentences] # Mock a 768-dim embedding

class MockChromaClient:
    def get_or_create_collection(self, name):
        return MockChromaCollection(name)

class MockChromaCollection:
    def __init__(self, name):
        self.name = name
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.embeddings = []

    def add(self, documents, metadatas, ids, embeddings=None):
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        if embeddings:
            self.embeddings.extend(embeddings)
        else:
            # Generate mock embeddings if not provided (for simplicity)
            self.embeddings.extend([list(range(768)) for _ in documents])

    def query(self, query_embeddings=None, query_texts=None, n_results=1, where=None, **kwargs):
        if where and 'customer_id' in where:
            customer_id = where['customer_id']
            filtered_docs = []
            filtered_metadatas = []
            for i, meta in enumerate(self.metadatas):
                if meta and meta.get('customer_id') == customer_id:
                    filtered_docs.append(self.documents[i])
                    filtered_metadatas.append(meta)
            return {"documents": [filtered_docs[:n_results]], "metadatas": [filtered_metadatas[:n_results]]}
        # Simple mock for now, returns first n_results for any query
        return {"documents": [self.documents[:n_results]], "metadatas": [self.metadatas[:n_results]]}

# 1. Customer Profile Manager
class CustomerProfileManager:
    def __init__(self):
        self.customer_data: Dict[str, Dict[str, Any]] = {}

    def add_interaction(self, customer_id: str, interaction_text: str, timestamp: str):
        if customer_id not in self.customer_data:
            self.customer_data[customer_id] = {"interactions": [], "profile_facts": []}
        self.customer_data[customer_id]["interactions"].append({
            "text": interaction_text,
            "timestamp": timestamp
        })

    def update_profile_fact(self, customer_id: str, fact_text: str):
        if customer_id not in self.customer_data:
            self.customer_data[customer_id] = {"interactions": [], "profile_facts": []}
        if fact_text not in self.customer_data[customer_id]["profile_facts"]:
            self.customer_data[customer_id]["profile_facts"].append(fact_text)

    def get_customer_history(self, customer_id: str) -> List[Dict[str, Any]]:
        return self.customer_data.get(customer_id, {}).get("interactions", [])

    def get_customer_profile_facts(self, customer_id: str) -> List[str]:
        return self.customer_data.get(customer_id, {}).get("profile_facts", [])

# 2. Embedding Generator
class EmbeddingGenerator:
    def __init__(self):
        self.model = MockSentenceTransformer() # Using mock model

    def get_embedding(self, text: str) -> List[float]:
        return self.model.encode([text])[0]

# 3. LLM Service (Mocked)
class LLMService:
    def generate_response(self, prompt: str) -> str:
        # Mock LLM response for demonstration
        if "previous orders" in prompt.lower() or "past purchases" in prompt.lower():
            return f"Based on your history, you recently purchased a 'Smartwatch Pro' and 'Wireless Earbuds'. How can I help with those items, or something new?" + "\nPrompt received: " + prompt
        elif "shipping address" in prompt.lower() or "delivery details" in prompt.lower():
            return f"Your primary shipping address is 123 Main St, Anytown, USA. Do you want to confirm or update this?" + "\nPrompt received: " + prompt
        elif "profile" in prompt.lower() and "preferred" in prompt.lower():
            return f"Your preferred product categories include Electronics and Home Goods. Is there anything specific you are looking for in these categories?" + "\nPrompt received: " + prompt
        return f"Hello! I'm a customer support agent. How can I assist you today?" + "\nPrompt received: " + prompt

# 4. Context Manager
class ContextManager:
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        self.chroma_client = MockChromaClient()
        self.history_collection = self.chroma_client.get_or_create_collection(name="customer_history")
        self.profile_collection = self.chroma_client.get_or_create_collection(name="customer_profiles")

    def store_interaction_data(self, customer_id: str, interaction_text: str, timestamp: str):
        embedding = self.embedding_generator.get_embedding(interaction_text)
        self.history_collection.add(
            documents=[interaction_text],
            metadatas=[{"customer_id": customer_id, "timestamp": timestamp}],
            ids=[str(uuid.uuid4())],
            embeddings=[embedding]
        )

    def store_profile_fact(self, customer_id: str, fact_text: str):
        embedding = self.embedding_generator.get_embedding(fact_text)
        self.profile_collection.add(
            documents=[fact_text],
            metadatas=[{"customer_id": customer_id}],
            ids=[str(uuid.uuid4())],
            embeddings=[embedding]
        )

    def retrieve_relevant_context(self, customer_id: str, current_query: str, num_history_items: int = 3, num_profile_facts: int = 2) -> Dict[str, List[str]]:
        query_embedding = self.embedding_generator.get_embedding(current_query)

        # Retrieve relevant history
        history_results = self.history_collection.query(
            query_embeddings=[query_embedding],
            n_results=num_history_items,
            where={"customer_id": customer_id}
        )
        retrieved_history = history_results["documents"][0] if history_results["documents"] else []

        # Retrieve relevant profile facts
        profile_results = self.profile_collection.query(
            query_embeddings=[query_embedding],
            n_results=num_profile_facts,
            where={"customer_id": customer_id}
        )
        retrieved_profile_facts = profile_results["documents"][0] if profile_results["documents"] else []

        return {
            "history": retrieved_history,
            "profile_facts": retrieved_profile_facts
        }

    def summarize_text(self, text: str, max_length: int = 200) -> str:
        # Simple summarization: truncate to max_length words
        words = text.split()
        if len(words) > max_length:
            return " ".join(words[:max_length]) + "..."
        return text

    def build_prompt(self, current_query: str, retrieved_history: List[str], retrieved_profile_facts: List[str]) -> str:
        prompt_parts = [
            "You are an AI customer support agent for an e-commerce platform.",
            "Provide helpful and concise answers based on the provided context.",
            f"Customer Query: {current_query}"
        ]

        if retrieved_history:
            summarized_history = [self.summarize_text(h) for h in retrieved_history]
            prompt_parts.append("\n--- Relevant Past Interactions ---")
            prompt_parts.extend(summarized_history)

        if retrieved_profile_facts:
            prompt_parts.append("\n--- Customer Profile Facts ---")
            prompt_parts.extend(retrieved_profile_facts)

        return "\n".join(prompt_parts)

# FastAPI Application
app = FastAPI()

# Data Models
class CustomerQueryRequest(BaseModel):
    customer_id: str
    query: str

class AgentResponse(BaseModel):
    response_message: str
    context_used: Dict[str, List[str]]
    full_prompt_sent_to_llm: str

# Initialize Components
customer_profile_manager = CustomerProfileManager()
embedding_generator = EmbeddingGenerator()
context_manager = ContextManager(embedding_generator=embedding_generator)
llm_service = LLMService()

@app.post("/ask_agent", response_model=AgentResponse)
async def ask_agent(request: CustomerQueryRequest):
    timestamp = datetime.now().isoformat()

    # 1. Record the new interaction
    customer_profile_manager.add_interaction(request.customer_id, request.query, timestamp)
    context_manager.store_interaction_data(request.customer_id, request.query, timestamp)

    # Simulate adding a profile fact for a new customer or if conditions met
    if request.customer_id == "user123" and len(customer_profile_manager.get_customer_history("user123")) == 1:
        customer_profile_manager.update_profile_fact(request.customer_id, "Preferred shipping method is express.")
        context_manager.store_profile_fact(request.customer_id, "Preferred shipping method is express.")

    # 2. Retrieve relevant context
    relevant_context = context_manager.retrieve_relevant_context(
        request.customer_id, request.query
    )

    # 3. Build the LLM prompt
    full_prompt = context_manager.build_prompt(
        request.query, relevant_context["history"], relevant_context["profile_facts"]
    )

    # 4. Generate response using LLM
    agent_reply = llm_service.generate_response(full_prompt)

    return AgentResponse(
        response_message=agent_reply,
        context_used=relevant_context,
        full_prompt_sent_to_llm=full_prompt
    )

# To run this application:
# 1. Save the code as main.py
# 2. Install FastAPI and uvicorn: pip install fastapi uvicorn pydantic
# 3. Run from your terminal: uvicorn main:app --reload
# 4. Access the API documentation at http://127.0.0.1:8000/docs