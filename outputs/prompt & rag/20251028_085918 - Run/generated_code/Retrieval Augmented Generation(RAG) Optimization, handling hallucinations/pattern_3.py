
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from functools import lru_cache
from loguru import logger
import asyncio
import time

# --- 1. Configuration and Logging ---
logger.add("file_{time}.log", rotation="1 day", level="INFO")

# --- 2. Dummy Models and Knowledge Base ---

class DummyEmbeddingModel:
    """Simulates a sentence-transformers embedding model."""
    def encode(self, texts):
        logger.info(f"Encoding text: {texts}")
        # In a real scenario, this would return actual embeddings
        # For simplicity, we return a hash as a pseudo-embedding
        return [hash(text) % (10**9) for text in texts]

class DummyVectorDB:
    """Simulates a ChromaDB-like vector store for medical documents."""
    def __init__(self):
        self.documents = {
            "doc1": {"text": "Fever is a common symptom of many illnesses, including the flu and common cold.", "metadata": {"source": "WHO"}},
            "doc2": {"text": "Headaches can be caused by stress, dehydration, or more serious conditions like migraines.", "metadata": {"source": "Mayo Clinic"}},
            "doc3": {"text": "The flu vaccine is recommended annually to protect against influenza viruses.", "metadata": {"source": "CDC"}},
            "doc4": {"text": "Diabetes mellitus is a chronic condition that affects how your body turns food into energy.", "metadata": {"source": "NIH"}},
            "doc5": {"text": "For mild pain, over-the-counter pain relievers like ibuprofen or acetaminophen are often effective.", "metadata": {"source": "WebMD"}},
        }
        self.embeddings = {hash(doc['text']) % (10**9): doc['text'] for doc in self.documents.values()}
        logger.info("DummyVectorDB initialized with sample medical documents.")

    def search(self, query_embedding, top_k=2):
        """Simulates searching for top_k relevant documents based on a query embedding."""
        logger.info(f"Searching DummyVectorDB for embedding: {query_embedding}")
        # In a real vector DB, this would involve complex similarity search.
        # Here, we'll do a very basic keyword-based 'relevance' simulation.
        query_text = self.embeddings.get(query_embedding, "") # Reverse lookup for demo
        
        relevant_docs = []
        for doc_id, doc_data in self.documents.items():
            if any(word.lower() in doc_data['text'].lower() for word in query_text.split() if len(word) > 2): # Simple keyword match
                relevant_docs.append(doc_data['text'])
        
        # Sort to make it somewhat deterministic for demo
        return sorted(list(set(relevant_docs)))[:top_k]

class DummyLLM:
    """Simulates a Language Model for generating responses."""
    def generate(self, context):
        logger.info(f"DummyLLM generating response for context: {context[:100]}...")
        # Simulate a delay for LLM processing
        time.sleep(0.5)
        if "Fever" in context and "flu" in context:
            return f"Based on the information, a fever could be a symptom of the flu. Always consult a healthcare professional for diagnosis. (Context: {context})"
        elif "headache" in context:
            return f"Headaches can have various causes. If severe, seek medical advice. (Context: {context})"
        elif "diabetes" in context:
            return f"Diabetes is a complex condition requiring medical management. Please consult a doctor. (Context: {context})"
        elif "pain" in context:
             return f"Over-the-counter pain relievers might help with mild pain. For persistent pain, consult a doctor. (Context: {context})"
        else:
            return f"I can provide general medical information. For specific concerns, please consult a healthcare professional. (Context: {context})"

# --- 3. Core RAG System Components ---

class Retriever:
    """Responsible for intelligently querying the Knowledge Base."""
    def __init__(self, embedding_model, vector_db):
        self.embedding_model = embedding_model
        self.vector_db = vector_db

    async def retrieve(self, query: str, top_k: int = 2) -> list[str]:
        """Retrieves relevant documents based on the query."""
        logger.info(f"Retriever initiated for query: {query}")
        # Simulate asynchronous embedding and search
        query_embedding = await asyncio.to_thread(self.embedding_model.encode, [query])
        retrieved_docs = await asyncio.to_thread(self.vector_db.search, query_embedding[0], top_k)
        logger.info(f"Retrieved {len(retrieved_docs)} documents.")
        return retrieved_docs

class ContextConditioner:
    """Combines query and retrieved documents into a coherent context."""
    def condition_context(self, query: str, retrieved_documents: list[str]) -> str:
        logger.info("Conditioning context...")
        if retrieved_documents:
            docs_str = "\n".join([f"Document: {doc}" for doc in retrieved_documents])
            context = f"User Query: {query}\nRelevant Medical Information:\n{docs_str}\nBased on the above, please provide a comprehensive and accurate answer."
        else:
            context = f"User Query: {query}\nNo specific relevant documents found. Please provide a general answer if possible."
        logger.info(f"Context conditioned: {context[:200]}...")
        return context

class AdaptiveDecisionMakingModule:
    """Decides when to retrieve information or abstain."""
    def should_retrieve(self, query: str) -> bool:
        # Simple rule-based decision for demonstration
        keywords_for_retrieval = ["medical", "symptom", "disease", "condition", "treatment", "medication", "health", "what is", "how to treat"]
        if any(keyword in query.lower() for keyword in keywords_for_retrieval):
            logger.info(f"Decision: Retrieve for query '{query}'")
            return True
        logger.info(f"Decision: No retrieval needed for query '{query}'")
        return False
    
    def should_abstain(self, query: str, confidence_score: float = 0.5) -> bool:
        # Simulate abstention for very vague or non-medical queries, or low confidence
        non_medical_keywords = ["weather", "news", "tell me a joke", "what is the capital"]
        if any(keyword in query.lower() for keyword in non_medical_keywords):
            logger.warning(f"Decision: Abstaining for non-medical query '{query}'")
            return True
        # In a real system, confidence_score would come from LLM or a separate classifier
        if confidence_score < 0.3: # Example threshold
            logger.warning(f"Decision: Abstaining due to low confidence for query '{query}'")
            return True
        return False

# --- 4. MedChat RAG System Integration ---

class MedChatRAGSystem:
    def __init__(self):
        self.embedding_model = DummyEmbeddingModel()
        self.vector_db = DummyVectorDB()
        self.llm = DummyLLM()
        self.retriever = Retriever(self.embedding_model, self.vector_db)
        self.context_conditioner = ContextConditioner()
        self.adaptive_module = AdaptiveDecisionMakingModule()
        logger.info("MedChatRAGSystem initialized.")

    @lru_cache(maxsize=128) # Caching frequently asked questions/responses
    async def generate_response(self, query: str) -> str:
        logger.info(f"Processing query with MedChatRAGSystem: {query}")

        if self.adaptive_module.should_abstain(query, confidence_score=0.6): # Dummy confidence
            return "I am designed to provide medical information. Please ask a relevant medical question."

        retrieved_documents = []
        if self.adaptive_module.should_retrieve(query):
            retrieved_documents = await self.retriever.retrieve(query)

        context = self.context_conditioner.condition_context(query, retrieved_documents)
        response = await asyncio.to_thread(self.llm.generate, context)
        logger.info(f"Generated response: {response[:100]}...")
        return response

# --- 5. FastAPI Application ---

app = FastAPI(title="MedChat AI RAG System")
medchat_rag = MedChatRAGSystem()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    timestamp: float
    cached: bool = False

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    original_query = request.message
    
    # Check if response is in cache before calling the RAG system
    cache_key = original_query
    cached_result = medchat_rag.generate_response.cache_info()
    
    # Functools lru_cache doesn't directly expose if an item *will be* a hit before calling.
    # We'll rely on the actual call and then check cache_info after to reflect hit/miss.
    
    response_text = await medchat_rag.generate_response(original_query)
    
    end_time = time.time()
    
    # After calling, check if it was a hit by comparing before/after stats (a bit hacky for demo)
    # A more robust solution for 'cached' flag would involve a custom cache decorator or wrapping lru_cache.
    current_cache_info = medchat_rag.generate_response.cache_info()
    was_cached = current_cache_info.hits > cached_result.hits
    
    logger.info(f"Chat request processed. Query: '{original_query}', Response: '{response_text[:50]}...', Cached: {was_cached}")
    
    return ChatResponse(
        response=response_text,
        timestamp=end_time,
        cached=was_cached
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "MedChat AI is running"}

@app.get("/cache_info")
async def get_cache_info():
    return medchat_rag.generate_response.cache_info()


if __name__ == "__main__":
    logger.info("Starting MedChat AI FastAPI application...")
    # You can run this file using: uvicorn medchat_ai:app --reload
    # For direct execution within this script, uncomment the following line:
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("To run, use 'uvicorn medchat_ai:app --reload' in your terminal.")
    logger.info("Access the API at http://127.0.0.1:8000/docs for Swagger UI.")

