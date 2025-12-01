import time
import uuid
import heapq
from collections import deque
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
import uvicorn
import threading


class PGCacheNode:
    def __init__(self, kv_tensor_id: str, kv_tensor_data: Any, size: int, recomputation_cost: float):
        self.kv_tensor_id = kv_tensor_id
        self.kv_tensor_data = kv_tensor_data
        self.size = size
        self.frequency = 1  # Initial frequency
        self.last_access_clock = 0
        self.recomputation_cost = recomputation_cost

    def calculate_priority(self, current_clock: int) -> float:
        # Priority = Clock * Frequency * Cost / Size
        # Use (current_clock - last_access_clock + 1) to represent recency and avoid division by zero if clock is 0
        recency_factor = current_clock - self.last_access_clock + 1
        if self.size == 0:
            return float("inf") # Avoid division by zero, prioritize non-zero size
        return (recency_factor * self.frequency * self.recomputation_cost) / self.size

    def __lt__(self, other): # For min-heap based on priority
        return self.calculate_priority(self.last_access_clock) < other.calculate_priority(other.last_access_clock)


class PGDSFCache:
    def __init__(self, max_size: int):
        self.max_size = max_size  # Max total size of KV tensors (e.g., in tokens)
        self.current_size = 0
        self.cache: Dict[str, PGCacheNode] = {}
        self.priority_queue: List[Tuple[float, str]] = []  # Min-heap (priority, kv_tensor_id)
        self.current_clock = 0

    def _update_priority_queue(self, kv_tensor_id: str):
        # Rebuild priority queue or update specific node's priority
        # For simplicity, we'll rebuild or remove and re-add in this example.
        # A more efficient way would be a custom heap or using a library that supports priority updates.
        node = self.cache[kv_tensor_id]
        priority = node.calculate_priority(self.current_clock)
        # Remove old entry if it exists (inefficient for large caches)
        # For practical implementation, consider a doubly linked list + dict for O(1) removals.
        # Here, we'll just add the new one; the old one might linger but won't be picked due to higher priority of updated node.
        heapq.heappush(self.priority_queue, (priority, kv_tensor_id))

    def get(self, kv_tensor_id: str) -> Any:
        self.current_clock += 1
        if kv_tensor_id in self.cache:
            node = self.cache[kv_tensor_id]
            node.frequency += 1
            node.last_access_clock = self.current_clock
            self._update_priority_queue(kv_tensor_id)
            return node.kv_tensor_data
        return None

    def put(self, node: PGCacheNode) -> PGCacheNode or None: # Returns evicted node if any
        self.current_clock += 1
        node.last_access_clock = self.current_clock
        node.frequency = 1 # Reset frequency for new node

        if node.kv_tensor_id in self.cache:
            # Update existing node
            existing_node = self.cache[node.kv_tensor_id]
            self.current_size -= existing_node.size
            self.cache[node.kv_tensor_id] = node
            self.current_size += node.size
            self._update_priority_queue(node.kv_tensor_id)
            return None

        evicted_node = None
        while self.current_size + node.size > self.max_size:
            if not self.priority_queue:
                raise Exception("Cache full and cannot evict any item.")
            
            # Pop the lowest priority item
            _, evicted_id = heapq.heappop(self.priority_queue)
            if evicted_id not in self.cache: # Node might have been updated/re-added, so this entry is stale
                continue
            
            evicted_node = self.cache.pop(evicted_id)
            self.current_size -= evicted_node.size

        self.cache[node.kv_tensor_id] = node
        self.current_size += node.size
        self._update_priority_queue(node.kv_tensor_id)
        return evicted_node

    def evict(self) -> PGCacheNode or None:
        if not self.cache:
            return None
        
        while self.priority_queue:
            _, evicted_id = heapq.heappop(self.priority_queue)
            if evicted_id in self.cache:
                evicted_node = self.cache.pop(evicted_id)
                self.current_size -= evicted_node.size
                return evicted_node
        return None


class HostMemory:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.current_size = 0
        self.memory: Dict[str, PGCacheNode] = {}
        self.lru_queue: deque = deque() # For basic LRU eviction in host memory

    def get(self, kv_tensor_id: str) -> Any:
        if kv_tensor_id in self.memory:
            # Update recency
            node = self.memory[kv_tensor_id]
            self.lru_queue.remove(kv_tensor_id)
            self.lru_queue.append(kv_tensor_id)
            return node.kv_tensor_data
        return None

    def put(self, node: PGCacheNode):
        if node.kv_tensor_id in self.memory:
            self.lru_queue.remove(node.kv_tensor_id)
            self.current_size -= self.memory[node.kv_tensor_id].size

        while self.current_size + node.size > self.max_size:
            if not self.lru_queue:
                # Should not happen if max_size is > 0 and a single item fits.
                # If an item is too large, it won't be cached.
                break 
            lru_id = self.lru_queue.popleft()
            if lru_id in self.memory: # Check if it's still there
                evicted_node = self.memory.pop(lru_id)
                self.current_size -= evicted_node.size
        
        if self.current_size + node.size <= self.max_size: # Only add if it fits after evictions
            self.memory[node.kv_tensor_id] = node
            self.current_size += node.size
            self.lru_queue.append(node.kv_tensor_id)
        else:
            pass # Node too large for host memory or host memory is full after evictions

    def remove(self, kv_tensor_id: str) -> PGCacheNode or None:
        if kv_tensor_id in self.memory:
            node = self.memory.pop(kv_tensor_id)
            self.current_size -= node.size
            self.lru_queue.remove(kv_tensor_id)
            return node
        return None


class CostEstimator:
    def __init__(self):
        self.cost_profile: Dict[str, float] = {
            "doc1_chunk0": 0.1,
            "doc1_chunk1": 0.12,
            "doc2_chunk0": 0.08,
            "doc3_chunk0": 0.15,
            "doc3_chunk1": 0.17,
            "doc3_chunk2": 0.14,
        }
        self.default_cost = 0.1

    def estimate_cost(self, document_chunk_id: str, prefix_length: int = 0) -> float:
        # Simplified: A real system would use a more complex model (e.g., bilinear interpolation)
        # based on document context, prefix length, and historical recomputation times.
        # For this example, we'll use a lookup table and a default.
        return self.cost_profile.get(document_chunk_id, self.default_cost)


class KnowledgeBase:
    def __init__(self, embedding_model):
        self.documents: Dict[str, str] = {}
        self.document_chunks: Dict[str, str] = {}
        self.chunk_embeddings: Dict[str, np.ndarray] = {}
        self.embedding_model = embedding_model
        self.chunk_id_to_doc_id: Dict[str, str] = {}

    def add_document_chunk(self, doc_id: str, chunk_id: str, content: str):
        self.document_chunks[chunk_id] = content
        self.chunk_id_to_doc_id[chunk_id] = doc_id
        embedding = self.embedding_model.encode(content, convert_to_tensor=False)
        self.chunk_embeddings[chunk_id] = embedding

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
        
        similarities = []
        for chunk_id, embedding in self.chunk_embeddings.items():
            sim = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((sim, chunk_id))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk_id for _, chunk_id in similarities[:top_k]]


def mock_llm_inference(prompt: str) -> str:
    # Simulate LLM response based on prompt content
    if "how do I reset my password" in prompt.lower():
        return "To reset your password, please visit our website's login page and click on 'Forgot Password'."
    elif "product warranty" in prompt.lower():
        return "Our products come with a one-year limited warranty. Please check your product documentation for details."
    return "I'm sorry, I don't have enough information to answer that. Could you please rephrase your question?"


# FastAPI App Initialization
app = FastAPI()

# Global instances
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
knowledge_base = KnowledgeBase(embedding_model)
gpu_cache = PGDSFCache(max_size=500)  # Max 500 tokens in GPU cache
host_memory = HostMemory(max_size=1000) # Max 1000 tokens in Host memory
cost_estimator = CostEstimator()

# Dummy data for knowledge base
def initialize_knowledge_base():
    documents_data = [
        ("doc1", "How to reset password. Go to settings, security, then reset password. Make sure to use a strong, unique password."),
        ("doc1", "Common login issues. If you cannot log in, check your internet connection and ensure caps lock is off."),
        ("doc2", "Product A features. Product A has X, Y, and Z features, designed for enhanced user experience."),
        ("doc3", "Warranty information for Product B. Product B comes with a 1-year warranty covering manufacturing defects."),
        ("doc3", "Customer support contact. For further assistance, contact our support team at support@example.com or call 1-800-XXX-XXXX."),
    ]
    for doc_id, content in documents_data:
        # Simulate splitting into chunks for KV tensors
        chunk_size = 50 # tokens
        words = content.split()
        for i in range(0, len(words), chunk_size):
            chunk_content = " ".join(words[i:i+chunk_size])
            chunk_id = f"{doc_id}_chunk{i//chunk_size}"
            knowledge_base.add_document_chunk(doc_id, chunk_id, chunk_content)

# Pydantic models for API requests
class IngestDocumentRequest(BaseModel):
    document_id: str
    content: str

class QueryRequest(BaseModel):
    query: str

class RAGResponse(BaseModel):
    query: str
    response: str
    cached_chunks: List[str]
    recomputed_chunks: List[str]
    cache_status: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    initialize_knowledge_base()


@app.post("/ingest", response_model=Dict[str, str])
async def ingest_document(request: IngestDocumentRequest):
    doc_id = request.document_id
    content = request.content
    
    chunk_size = 50 # tokens
    words = content.split()
    ingested_chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_content = " ".join(words[i:i+chunk_size])
        chunk_id = f"{doc_id}_chunk{i//chunk_size}"
        knowledge_base.add_document_chunk(doc_id, chunk_id, chunk_content)
        ingested_chunks.append(chunk_id)
    return {"status": "success", "ingested_chunks": ", ".join(ingested_chunks)}


@app.post("/query", response_model=RAGResponse)
async def process_query(request: QueryRequest):
    query = request.query
    relevant_chunk_ids = knowledge_base.retrieve_relevant_chunks(query)

    context_chunks = []
    cached_chunks_info = []
    recomputed_chunks_info = []

    for chunk_id in relevant_chunk_ids:
        # Try GPU cache
        kv_tensor_data = gpu_cache.get(chunk_id)
        if kv_tensor_data:
            context_chunks.append(kv_tensor_data)
            cached_chunks_info.append(chunk_id + " (GPU)")
        else:
            # Try Host Memory
            kv_tensor_data = host_memory.get(chunk_id)
            if kv_tensor_data:
                host_memory.remove(chunk_id) # Remove from host to promote to GPU
                size = len(knowledge_base.document_chunks[chunk_id].split())
                cost = cost_estimator.estimate_cost(chunk_id)
                node = PGCacheNode(chunk_id, kv_tensor_data, size, cost)
                evicted = gpu_cache.put(node)
                if evicted:
                    host_memory.put(evicted) # Move evicted GPU node to host
                context_chunks.append(kv_tensor_data)
                cached_chunks_info.append(chunk_id + " (Host->GPU)")
            else:
                # Recompute (simulate fetching from original source/KB)
                original_content = knowledge_base.document_chunks.get(chunk_id, "")
                if original_content:
                    size = len(original_content.split())
                    cost = cost_estimator.estimate_cost(chunk_id)
                    kv_tensor_data = f"KV_TENSOR_FOR_{chunk_id}: {original_content}"
                    node = PGCacheNode(chunk_id, kv_tensor_data, size, cost)
                    evicted = gpu_cache.put(node)
                    if evicted:
                        host_memory.put(evicted) # Move evicted GPU node to host
                    context_chunks.append(kv_tensor_data)
                    recomputed_chunks_info.append(chunk_id)
                else:
                    pass # Should not happen if chunk_id comes from KB

    prompt = f"Context: {' '.join(context_chunks)}\n\nQuestion: {query}\nAnswer:"
    llm_response = mock_llm_inference(prompt)

    return RAGResponse(
        query=query,
        response=llm_response,
        cached_chunks=cached_chunks_info,
        recomputed_chunks=recomputed_chunks_info,
        cache_status={
            "gpu_cache_size": gpu_cache.current_size,
            "gpu_cache_max_size": gpu_cache.max_size,
            "host_memory_size": host_memory.current_size,
            "host_memory_max_size": host_memory.max_size,
            "gpu_cache_items": list(gpu_cache.cache.keys()),
            "host_memory_items": list(host_memory.memory.keys()),
        }
    )


def run_fastapi_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)


def run_streamlit_app():
    st.title("Intelligent Customer Support RAG System")

    st.header("Knowledge Base Ingestion")
    ingest_doc_id = st.text_input("Document ID (e.g., product_manual_v1)", key="ingest_id")
    ingest_content = st.text_area("Document Content", height=150, key="ingest_content")
    if st.button("Ingest Document"):
        if ingest_doc_id and ingest_content:
            try:
                response = requests.post("http://localhost:8000/ingest", json={
                    "document_id": ingest_doc_id,
                    "content": ingest_content
                })
                if response.status_code == 200:
                    st.success(f"Document ingested: {response.json().get('ingested_chunks')}")
                else:
                    st.error(f"Error ingesting document: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI backend. Please ensure it is running.")
        else:
            st.warning("Please provide both document ID and content.")

    st.header("Customer Query")
    user_query = st.text_input("Enter your query:", "How do I reset my password?")

    if st.button("Get Answer"):
        if user_query:
            try:
                response = requests.post("http://localhost:8000/query", json={"query": user_query})
                if response.status_code == 200:
                    rag_response = RAGResponse(**response.json())
                    st.subheader("RAG System Response:")
                    st.write(rag_response.response)

                    st.subheader("Cache Status:")
                    st.json(rag_response.cache_status)
                    st.write(f"Cached Chunks: {', '.join(rag_response.cached_chunks) if rag_response.cached_chunks else 'None'}")
                    st.write(f"Recomputed Chunks: {', '.join(rag_response.recomputed_chunks) if rag_response.recomputed_chunks else 'None'}")
                else:
                    st.error(f"Error processing query: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI backend. Please ensure it is running.")
        else:
            st.warning("Please enter a query.")


if __name__ == "__main__":
    # Note: To run both FastAPI and Streamlit concurrently in a single script for demonstration,
    # you would typically use multiprocessing or two separate terminals.
    # For this submission, the code provides both components. 
    # Instructions to run them separately are provided in the explanation.
    # To simplify this single file, we'll make Streamlit the primary entry point 
    # and assume the FastAPI server is running or will be started manually.
    # For Streamlit to make requests, 'requests' library is needed.
    import requests 
    run_streamlit_app()

    # If you want to run FastAPI in a separate thread (for development/testing, not production):
    # fastapi_thread = threading.Thread(target=run_fastapi_server)
    # fastapi_thread.daemon = True
    # fastapi_thread.start()
    # time.sleep(1) # Give server a moment to start
    # run_streamlit_app()

