import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import collections

# --- 1. RAGCacheManager (Simplified Custom Implementation) ---
class RAGCacheManager:
    def __init__(self, cache_limit: int = 100):
        self.cache_limit = cache_limit
        self.cache = collections.OrderedDict() # Acts as an LRU cache

    def get(self, document_id: str) -> Any:
        if document_id in self.cache:
            # Move to the end to signify recent use (LRU)
            self.cache.move_to_end(document_id)
            return self.cache[document_id]
        return None

    def put(self, document_id: str, kv_tensors: Any):
        if document_id in self.cache:
            self.cache.move_to_end(document_id)
        self.cache[document_id] = kv_tensors
        if len(self.cache) > self.cache_limit:
            # Evict the least recently used item
            self.cache.popitem(last=False)
        # In a real RAGCache, this would involve more complex management:
        # - 'knowledge tree' to organize order-sensitive KV tensors.
        # - 'PGDSF replacement policy' for efficient multi-level cache management.
        # - Managing GPU/host memory explicitly.

    def contains(self, document_id: str) -> bool:
        return document_id in self.cache

# --- 2. FastAPI Application Setup ---
app = FastAPI(
    title="Medical Research Q&A Assistant",
    description="A RAG-powered assistant for medical queries with dynamic knowledge caching."
)

# --- 3. Embedding Model ---
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Please ensure you have internet access or the model is downloaded locally.")
    # Fallback or raise error
    raise

# --- 4. Vector Store (Faiss) and Dummy Data ---
# Simulate medical documents
medical_documents = {
    "doc_001": "Recent study on the efficacy of Remdesivir for COVID-19 treatment. Key findings indicate a reduction in recovery time for hospitalized patients with severe COVID-19.",
    "doc_002": "Mechanism of action of monoclonal antibodies in autoimmune diseases. Focus on TNF-alpha inhibitors like Adalimumab and Infliximab.",
    "doc_003": "Clinical trials for a new Alzheimer's drug targeting amyloid-beta plaques. Early results show promise in slowing cognitive decline.",
    "doc_004": "Understanding the side effects and contraindications of Metformin in type 2 diabetes management. Special considerations for renal impairment.",
    "doc_005": "Guidelines for pediatric vaccination schedules: MMR, DTaP, Polio. Importance of timely administration.",
    "doc_006": "Impact of lifestyle interventions on hypertension control. Diet, exercise, and stress reduction strategies.",
    "doc_007": "Diagnosis and management of sepsis in critically ill patients. Importance of early recognition and broad-spectrum antibiotics.",
    "doc_008": "Drug interactions between Warfarin and common antibiotics, focusing on increased bleeding risk.",
    "doc_009": "Role of genetic testing in personalized oncology. BRCA mutations in breast cancer and companion diagnostics.",
    "doc_010": "New advancements in CRISPR-Cas9 technology for genetic disorders. Ethical considerations and therapeutic potential."
}

document_ids = list(medical_documents.keys())
document_texts = list(medical_documents.values())

# Generate embeddings for the documents
document_embeddings = embedding_model.encode(document_texts)
embedding_dimension = document_embeddings.shape[1]

# Create a Faiss index
faiss_index = faiss.IndexFlatL2(embedding_dimension)
faiss_index.add(document_embeddings)

# Mapping from Faiss index ID to our document_id
faiss_id_to_doc_id = {i: doc_id for i, doc_id in enumerate(document_ids)}

# --- 5. RAGCache Manager Initialization ---
rag_cache_manager = RAGCacheManager(cache_limit=50) # Cache up to 50 document KV states

# --- 6. Large Language Model (LLM) Inference (vLLM Placeholder) ---
async def call_llm_inference(prompt: str) -> str:
    # This function simulates calling a vLLM service.
    # In a real application, this would involve an HTTP call to a vLLM endpoint
    # or direct integration if vLLM is run in-process.
    # For demonstration, we'll return a static or simple generative response.
    print(f"Simulating LLM inference for prompt:\n{prompt[:200]}...") # Log part of the prompt
    if "Remdesivir" in prompt and "COVID-19" in prompt:
        return "Remdesivir has shown to reduce recovery time for hospitalized patients with severe COVID-19, according to recent studies."
    elif "Adalimumab" in prompt and "autoimmune" in prompt:
        return "Adalimumab is a monoclonal antibody that inhibits TNF-alpha, commonly used in the treatment of various autoimmune diseases."
    elif "Alzheimer" in prompt and "amyloid-beta" in prompt:
        return "New drugs targeting amyloid-beta plaques show promise in slowing cognitive decline in Alzheimer's patients in early clinical trials."
    else:
        return "Based on the provided medical context, the relevant information is: [LLM generated answer based on context]."

# --- Request Model for FastAPI ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3 # Number of top documents to retrieve

class QueryResponse(BaseModel):
    answer: str
    retrieved_documents: List[str]
    cached_documents_count: int

# --- 7. FastAPI Endpoint ---
@app.post("/query", response_model=QueryResponse)
async def query_medical_assistant(request: QueryRequest):
    user_query = request.query
    top_k = request.top_k

    # Step 1: Embed the user query
    query_embedding = embedding_model.encode([user_query])

    # Step 2: Retrieve relevant documents from Faiss
    distances, indices = faiss_index.search(query_embedding, top_k)
    retrieved_doc_ids = [faiss_id_to_doc_id[idx] for idx in indices[0]]

    # Step 3: Process retrieved documents and leverage RAGCache
    context_parts = []
    cached_docs_count = 0
    for doc_id in retrieved_doc_ids:
        # Check if KV tensors for this document are in cache
        cached_kv_tensors = rag_cache_manager.get(doc_id)

        if cached_kv_tensors:
            # If cached, use the cached "KV tensors" (here, just the text content)
            context_parts.append(cached_kv_tensors)
            cached_docs_count += 1
            print(f"Cache Hit for document: {doc_id}")
        else:
            # If not cached, retrieve the full document content
            doc_content = medical_documents.get(doc_id, "")
            if doc_content:
                context_parts.append(doc_content)
                # Simulate generating KV tensors and cache them
                # In a real RAGCache, this would involve running the LLM's
                # initial layers on the document to produce KV tensors.
                rag_cache_manager.put(doc_id, doc_content) # Store doc content as simulated KV
                print(f"Cache Miss for document: {doc_id}, added to cache.")

    # Step 4: Construct the prompt for the LLM
    context = "\n\n".join(context_parts)
    prompt = f"Given the following medical context, answer the question accurately and concisely:\n\nContext:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"

    # Step 5: Call the LLM inference
    llm_answer = await call_llm_inference(prompt)

    return QueryResponse(
        answer=llm_answer,
        retrieved_documents=retrieved_doc_ids,
        cached_documents_count=cached_docs_count
    )

if __name__ == "__main__":
    print("Starting Medical Research Q&A Assistant API...")
    print("To run, execute: uvicorn medical_rag_assistant:app --reload")
