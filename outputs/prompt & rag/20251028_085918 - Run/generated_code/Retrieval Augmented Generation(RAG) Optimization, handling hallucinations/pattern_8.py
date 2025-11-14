"""
medical_assistant_ralm.py

This script implements an Intelligent Medical Assistant for Clinical Decision Support
using an Adaptive and Optimized Retrieval-Augmented Language Model (RALM) architecture.
It leverages sentence-transformers for embeddings, FAISS for vector search,
and FastAPI for the API backend.

NOTE: This is a simplified implementation for demonstration purposes.
  - The 'LLM' is simulated using string formatting.
  - 'Query Understanding' and 'Re-ranking' modules are placeholder functions.
  - Medical data is dummy data.
  - FAISS index is in-memory and not persistent.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from loguru import logger

# Configure Loguru as the default logger
logger.add(
    "medical_assistant.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO"
)

# Suppress default uvicorn loggers to avoid duplicate messages
logging.getLogger("uvicorn.access").handlers = []
logging.getLogger("uvicorn.error").handlers = []

logger.info("Starting Medical Assistant RALM application...")

class MedicalKnowledgeBase:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Initializing Embedding Model: {embedding_model_name}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance for similarity
            self.documents = []  # Stores original document texts
            logger.info(f"Embedding model loaded. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model or initialize FAISS: {e}")
            raise

    def _create_dummy_data(self) -> List[Dict[str, str]]:
        """Generates a small set of dummy medical documents for demonstration."""
        logger.info("Creating dummy medical data...")
        return [
            {
                "id": "doc1",
                "text": "Type 2 diabetes mellitus is a chronic metabolic disorder characterized by high blood sugar, insulin resistance, and relative lack of insulin. Treatment often includes lifestyle changes, metformin, and other oral hypoglycemic agents."
            },
            {
                "id": "doc2",
                "text": "Hypertension, or high blood pressure, significantly increases the risk of heart disease, stroke, and kidney failure. Common treatments involve ACE inhibitors, ARBs, diuretics, and beta-blockers, alongside dietary modifications."
            },
            {
                "id": "doc3",
                "text": "Symptoms of a common cold typically include a runny nose, sore throat, cough, and congestion. It is caused by viruses, mainly rhinoviruses, and usually resolves within 7-10 days. Rest and hydration are key."
            },
            {
                "id": "doc4",
                "text": "A myocardial infarction, commonly known as a heart attack, occurs when blood flow to a part of the heart is blocked, causing heart muscle damage. Emergency treatment involves restoring blood flow, often with angioplasty or thrombolytics."
            },
            {
                "id": "doc5",
                "text": "Asthma is a chronic respiratory condition where airways narrow and swell and produce extra mucus, making breathing difficult. Inhalers containing bronchodilators and corticosteroids are primary treatments."
            },
             {
                "id": "doc6",
                "text": "Metformin is a first-line medication for type 2 diabetes. It works by decreasing glucose production in the liver and improving insulin sensitivity. Side effects can include gastrointestinal issues."
            }
        ]

    def index_documents(self, docs: List[Dict[str, str]]):
        """Embeds and indexes documents into the FAISS index."""
        if not docs:
            logger.warning("No documents provided for indexing.")
            return

        logger.info(f"Indexing {len(docs)} documents...")
        try:
            texts = [doc["text"] for doc in docs]
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            self.index.add(embeddings)
            self.documents.extend(docs) # Store the original documents with IDs
            logger.info(f"Successfully indexed {len(docs)} documents. Total in index: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Error during document indexing: {e}")
            raise

    def retrieve_documents(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves top-k most similar documents from the FAISS index."""
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty. No documents to retrieve.")
            return []

        logger.info(f"Retrieving top {k} documents...")
        try:
            D, I = self.index.search(np.array([query_embedding]), k)  # D: distances, I: indices
            retrieved_docs = []
            for i, doc_idx in enumerate(I[0]):
                if doc_idx != -1: # -1 indicates no document found for that slot
                    doc = self.documents[doc_idx]
                    retrieved_docs.append({"document": doc["text"], "score": float(D[0][i]), "id": doc["id"]})
            logger.info(f"Retrieved {len(retrieved_docs)} documents.")
            return retrieved_docs
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            return []


class MedicalRALMAssistant:
    def __init__(self, knowledge_base: MedicalKnowledgeBase, llm_model_name: str = "Simulated LLM"):
        self.knowledge_base = knowledge_base
        self.llm_model_name = llm_model_name # Placeholder for a real LLM integration
        self.cache = {}
        logger.info(f"Medical RALM Assistant initialized with {llm_model_name}.")

    def _query_understanding(self, query: str) -> Dict[str, Any]:
        """Placeholder for advanced query understanding (NER, intent, complexity)."""
        logger.debug(f"Performing query understanding for: '{query}'")
        # In a real scenario, this would use NLP models (e.g., Spacy, NLTK, or a fine-tuned model)
        # to extract medical entities, determine query type (diagnosis, treatment, drug interaction), etc.
        return {"original_query": query, "keywords": query.split(), "complexity": "medium"}

    def _dynamic_retrieval(self, query_info: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Adapts retrieval based on query information and context sufficiency."""
        logger.debug(f"Initiating dynamic retrieval for query_info: {query_info}")
        query = query_info["original_query"]
        query_embedding = self.knowledge_base.embedding_model.encode([query], convert_to_numpy=True)[0]

        # Simulate adaptive strategy (e.g., retrieve more documents for complex queries)
        adjusted_k = k
        if query_info["complexity"] == "high": # Placeholder for real complexity assessment
            adjusted_k = k + 2
            logger.info(f"Query complexity is high, adjusting retrieval k to {adjusted_k}")

        retrieved = self.knowledge_base.retrieve_documents(query_embedding, k=adjusted_k)

        # Simulate context assessment (e.g., if initial retrieval is not diverse enough)
        if not retrieved or len(retrieved) < adjusted_k / 2: # Very basic check
            logger.warning("Initial retrieval seems insufficient, attempting broader search.")
            # In a real system, this might involve a different search strategy or re-querying
            pass # For this demo, we'll stick to the initial retrieved for simplicity

        return retrieved

    def _re_rank_documents(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Placeholder for re-ranking retrieved documents using a cross-encoder or similar."""
        logger.debug(f"Re-ranking {len(retrieved_docs)} documents for query: '{query}'")
        # In a real scenario, a cross-encoder model (e.g., from sentence-transformers)
        # would score each document-query pair for more precise relevance.
        # For now, we return them as is, assuming FAISS already provided decent ranking.
        return sorted(retrieved_docs, key=lambda x: x["score"], reverse=False) # L2 distance: lower is better

    def _generate_response_with_llm(self, query: str, context: List[str]) -> str:
        """Simulates LLM response generation using the provided context."""
        logger.info(f"Simulating LLM response for query: '{query}' with {len(context)} context pieces.")
        if not context:
            return f"I'm sorry, I couldn't find specific medical information relevant to '{query}' in my knowledge base. Please consult a healthcare professional."

        combined_context = " ".join([f"[Source {i+1}] {doc}" for i, doc in enumerate(context)])
        
        # Simple rule-based generation, mimicking an LLM's structure
        response_template = (
            f"Based on the available medical information related to '{query}', here's what I found:\n\n"
            f"Contextual details: {combined_context}\n\n"
            f"Please note: This information is for educational purposes and should not replace professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment."
        )
        return response_template

    def process_query(self, query: str) -> Dict[str, Any]:
        """Main RAG pipeline for processing a medical query."""
        logger.info(f"Processing query: '{query}'")

        # 1. Check cache for frequently accessed facts/queries (basic caching)
        if query in self.cache:
            logger.info("Query found in cache, returning cached response.")
            return self.cache[query]

        query_info = self._query_understanding(query)
        retrieved_docs_with_scores = self._dynamic_retrieval(query_info)
        
        if not retrieved_docs_with_scores:
            response = self._generate_response_with_llm(query, []) # No context
            result = {"query": query, "response": response, "sources": []}
            self.cache[query] = result
            return result

        re_ranked_docs = self._re_rank_documents(query, retrieved_docs_with_scores)
        context_texts = [doc["document"] for doc in re_ranked_docs]
        
        response = self._generate_response_with_llm(query, context_texts)
        sources = [{
            "id": doc["id"],
            "text_snippet": doc["document"][:150] + "..." if len(doc["document"]) > 150 else doc["document"],
            "relevance_score": doc["score"]
        } for doc in re_ranked_docs]
        
        result = {"query": query, "response": response, "sources": sources}
        self.cache[query] = result # Store in cache
        return result


# FastAPI Application Setup
app = FastAPI(
    title="Intelligent Medical Assistant API",
    description="API for a Retrieval-Augmented Language Model (RALM) medical assistant."
)

# Global instances (initialized on startup)
medical_knowledge_base: MedicalKnowledgeBase = None
medical_ralm_assistant: MedicalRALMAssistant = None

class QueryRequest(BaseModel):
    query: str

class SourceInfo(BaseModel):
    id: str
    text_snippet: str
    relevance_score: float

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[SourceInfo]


@app.on_event("startup")
async def startup_event():
    """Initialize the knowledge base and RALM assistant on application startup."""
    global medical_knowledge_base, medical_ralm_assistant
    logger.info("Application startup: Initializing MedicalKnowledgeBase and MedicalRALMAssistant...")
    try:
        medical_knowledge_base = MedicalKnowledgeBase()
        dummy_docs = medical_knowledge_base._create_dummy_data()
        medical_knowledge_base.index_documents(dummy_docs)
        medical_ralm_assistant = MedicalRALMAssistant(medical_knowledge_base)
        logger.info("Medical Assistant initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize application components: {e}")
        # In a production setting, you might want to exit or provide a degraded service
        raise RuntimeError(f"Failed to start application: {e}")


@app.post("/query", response_model=QueryResponse)
async def process_medical_query(request: QueryRequest):
    """Endpoint to process medical queries using the RALM assistant."""
    logger.info(f"Received query request: '{request.query}'")
    try:
        result = medical_ralm_assistant.process_query(request.query)
        logger.info("Query processed successfully.")
        return QueryResponse(**result)
    except Exception as e:
        logger.exception(f"Error processing query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


if __name__ == "__main__":
    # To run this, save it as a Python file (e.g., medical_assistant_ralm.py)
    # and execute from your terminal:
    # uvicorn medical_assistant_ralm:app --host 0.0.0.0 --port 8000 --reload
    logger.info("Application ready. To run, use 'uvicorn medical_assistant_ralm:app --host 0.0.0.0 --port 8000'.")
    # For direct execution within a script for testing, you might uncomment the below
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)

