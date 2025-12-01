"""
This script implements a conceptual intelligent customer support chatbot featuring a "Swap-Out-Only-Once Cache Strategy" for LLM Key-Value (KV) cache management. It demonstrates how to optimize memory transfers between fast GPU memory and slower host memory, reducing bandwidth consumption for frequently accessed conversational states and knowledge base articles.

Key Components:
- FastAPI for the web API.
- Pydantic for data validation.
- A custom KVCacheManager implementing the "Swap-Out-Only-Once" strategy.
- Mock LLMService and KnowledgeBaseService to simulate their functionalities.
- ChatbotService to orchestrate the overall conversational flow.

To run this code:
1. Ensure you have PyTorch, transformers, sentence-transformers, faiss-gpu (or faiss-cpu), fastapi, uvicorn, pydantic, and loguru installed.
   `pip install torch transformers sentence-transformers faiss-cpu fastapi "uvicorn[standard]" pydantic loguru`
   (Use `faiss-gpu` if you have a CUDA-enabled GPU and appropriate drivers).
2. Save the code as `customer_support_chatbot.py`.
3. Run the FastAPI application:
   `uvicorn customer_support_chatbot:app --reload`
4. Access the API documentation at `http://127.0.0.1:8000/docs` to test the `/chat` endpoint.
"""

import os
import time
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- 1. Pydantic Models ---
class UserQuery(BaseModel):
    session_id: str
    text: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    cached_kv_hits: int = 0
    cached_kv_misses: int = 0

class KVCacheEntry:
    node_id: str
    # gpu_tensors: Optional[List[torch.Tensor]] = None
    host_tensors: Optional[List[torch.Tensor]] = None
    is_host_resident: bool = False
    last_accessed: float = 0.0
    size_bytes: int = 0

    def __init__(self, node_id: str, size_bytes: int):
        self.node_id = node_id
        self.size_bytes = size_bytes
        self.last_accessed = time.time()

# --- 2. KV Cache Manager (Swap-Out-Only-Once Strategy) ---
class KVCacheManager:
    def __init__(self, gpu_cache_capacity_mb: int = 200):
        self.gpu_cache_capacity_bytes = gpu_cache_capacity_mb * 1024 * 1024
        self.gpu_cache: Dict[str, List[torch.Tensor]] = {}
        self.host_cache: Dict[str, List[torch.Tensor]] = {}
        self.node_metadata: Dict[str, KVCacheEntry] = {}
        self.current_gpu_usage_bytes = 0
        logger.info(f"KV Cache Manager initialized with GPU capacity: {gpu_cache_capacity_mb}MB")

    def _get_tensors_size(self, tensors: List[torch.Tensor]) -> int:
        return sum(t.element_size() * t.nelement() for t in tensors)

    def _evict_from_gpu(self, node_id_to_evict: str):
        if node_id_to_evict not in self.gpu_cache:
            return

        kv_tensors_gpu = self.gpu_cache[node_id_to_evict]
        kv_entry = self.node_metadata[node_id_to_evict]
        tensors_size = self._get_tensors_size(kv_tensors_gpu)

        if not kv_entry.is_host_resident:
            # First eviction: Copy to host memory
            kv_tensors_cpu = [t.to("cpu") for t in kv_tensors_gpu]
            self.host_cache[node_id_to_evict] = kv_tensors_cpu
            kv_entry.is_host_resident = True
            logger.debug(f"Evicted '{node_id_to_evict}' from GPU to host (first time, size: {tensors_size/1024/1024:.2f}MB)")
        else:
            # Subsequent eviction: Host already has a copy, just free GPU memory
            logger.debug(f"Evicted '{node_id_to_evict}' from GPU (already on host, size: {tensors_size/1024/1024:.2f}MB)")

        # Free GPU memory by removing reference
        del self.gpu_cache[node_id_to_evict]
        self.current_gpu_usage_bytes -= tensors_size
        torch.cuda.empty_cache() # Aggressively clear CUDA cache

    def add_to_cache(self, node_id: str, kv_tensors: List[torch.Tensor]):
        tensors_size = self._get_tensors_size(kv_tensors)

        if node_id in self.gpu_cache:
            # Already in GPU cache, update access time
            self.node_metadata[node_id].last_accessed = time.time()
            return

        # Evict if capacity exceeded
        while self.current_gpu_usage_bytes + tensors_size > self.gpu_cache_capacity_bytes:
            if not self.gpu_cache:
                logger.warning(f"Cannot add {node_id} (size {tensors_size/1024/1024:.2f}MB), GPU cache is full and empty. Increase capacity.")
                # If the item itself is larger than capacity, handle it (e.g., store only on host)
                if not self.node_metadata.get(node_id) or not self.node_metadata[node_id].is_host_resident:
                     # Store on host if too large for GPU initially
                    kv_tensors_cpu = [t.to("cpu") for t in kv_tensors]
                    self.host_cache[node_id] = kv_tensors_cpu
                    self.node_metadata[node_id] = KVCacheEntry(node_id, tensors_size)
                    self.node_metadata[node_id].is_host_resident = True
                    logger.warning(f"Node '{node_id}' too large for GPU cache, storing directly on host.")
                return

            # Simple LRU eviction policy
            lru_node_id = min(self.node_metadata, key=lambda nid: self.node_metadata[nid].last_accessed if nid in self.gpu_cache else float('inf'))
            if lru_node_id not in self.gpu_cache: # All GPU items too new or no GPU items
                break # Should not happen if self.gpu_cache is not empty

            self._evict_from_gpu(lru_node_id)

        # Add to GPU cache
        self.gpu_cache[node_id] = kv_tensors
        self.current_gpu_usage_bytes += tensors_size
        if node_id not in self.node_metadata:
            self.node_metadata[node_id] = KVCacheEntry(node_id, tensors_size)
        else:
            self.node_metadata[node_id].last_accessed = time.time()
        logger.debug(f"Added '{node_id}' to GPU cache (size: {tensors_size/1024/1024:.2f}MB). Current GPU usage: {self.current_gpu_usage_bytes/1024/1024:.2f}MB/{self.gpu_cache_capacity_bytes/1024/1024:.2f}MB")

    def retrieve_from_cache(self, node_id: str) -> Optional[List[torch.Tensor]]:
        if node_id in self.gpu_cache:
            self.node_metadata[node_id].last_accessed = time.time()
            logger.debug(f"Cache hit: '{node_id}' found in GPU cache.")
            return self.gpu_cache[node_id]

        if node_id in self.host_cache:
            kv_tensors_cpu = self.host_cache[node_id]
            kv_entry = self.node_metadata[node_id]
            tensors_size = self._get_tensors_size(kv_tensors_cpu)

            # Try to move back to GPU (promotion)
            # First, check if there's enough space, evict if necessary
            while self.current_gpu_usage_bytes + tensors_size > self.gpu_cache_capacity_bytes:
                if not self.gpu_cache:
                    logger.warning(f"Cannot promote '{node_id}' from host, GPU cache full and empty. Will use CPU copy for this operation.")
                    self.node_metadata[node_id].last_accessed = time.time()
                    return kv_tensors_cpu # Return CPU copy if GPU is too full to promote

                lru_node_id = min(self.node_metadata, key=lambda nid: self.node_metadata[nid].last_accessed if nid in self.gpu_cache else float('inf'))
                if lru_node_id == node_id: # The item we want to promote is the only one in GPU (or newest) and it doesn't fit
                    logger.warning(f"Cannot promote '{node_id}' from host, GPU cache full and it's the newest GPU item. Will use CPU copy.")
                    self.node_metadata[node_id].last_accessed = time.time()
                    return kv_tensors_cpu
                self._evict_from_gpu(lru_node_id)

            kv_tensors_gpu = [t.to("cuda") for t in kv_tensors_cpu]
            self.gpu_cache[node_id] = kv_tensors_gpu
            self.current_gpu_usage_bytes += tensors_size
            self.node_metadata[node_id].last_accessed = time.time()
            logger.debug(f"Cache hit: '{node_id}' promoted from host to GPU (size: {tensors_size/1024/1024:.2f}MB). Current GPU usage: {self.current_gpu_usage_bytes/1024/1024:.2f}MB/{self.gpu_cache_capacity_bytes/1024/1024:.2f}MB")
            return kv_tensors_gpu

        logger.debug(f"Cache miss: '{node_id}' not found in any cache level.")
        return None

    def clear_node_from_cache(self, node_id: str):
        if node_id in self.gpu_cache:
            tensors_size = self._get_tensors_size(self.gpu_cache[node_id])
            del self.gpu_cache[node_id]
            self.current_gpu_usage_bytes -= tensors_size
            torch.cuda.empty_cache()
            logger.debug(f"Cleared '{node_id}' from GPU cache.")

        if node_id in self.host_cache:
            del self.host_cache[node_id]
            logger.debug(f"Cleared '{node_id}' from host cache.")

        if node_id in self.node_metadata:
            del self.node_metadata[node_id]
            logger.debug(f"Removed metadata for '{node_id}'.")

    def get_cache_stats(self):
        return {
            "gpu_usage_bytes": self.current_gpu_usage_bytes,
            "gpu_capacity_bytes": self.gpu_cache_capacity_bytes,
            "gpu_items": len(self.gpu_cache),
            "host_items": len(self.host_cache),
            "total_nodes_tracked": len(self.node_metadata)
        }

# --- 3. Mock LLM Service ---
class LLMService:
    def __init__(self, model_name: str = "distilbert-base-uncased"): # Placeholder
        # In a real scenario, load an actual LLM like Llama or Mistral
        # self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"LLMService initialized with mock model: {model_name}")

    def generate_response(self, prompt: str, kv_cache_tensors: Optional[List[torch.Tensor]] = None) -> str:
        # Simulate LLM inference with a delay
        time.sleep(0.5) 
        base_response = f"Hello! I'm a chatbot. You said: '{prompt[:50]}...'"
        if kv_cache_tensors:
            base_response += f" (Used {len(kv_cache_tensors)} KV cache tensors)"
        return base_response

# --- 4. Knowledge Base Service ---
class KnowledgeBaseService:
    def __init__(self, documents: List[str], embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.documents = documents
        self.embedder = SentenceTransformer(embedding_model_name)
        self.embeddings = self.embedder.encode(documents, convert_to_tensor=True, show_progress_bar=False).cpu().numpy()
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)
        logger.info(f"KnowledgeBaseService initialized with {len(documents)} documents. Faiss index created.")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.encode(query, convert_to_tensor=True, show_progress_bar=False).cpu().numpy()

    def retrieve_documents(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        distances, indices = self.index.search(np.expand_dims(query_embedding, axis=0), top_k)
        retrieved = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        logger.debug(f"Retrieved {len(retrieved)} documents from KB.")
        return retrieved

# --- 5. Chatbot Service ---
class ChatbotService:
    def __init__(self):
        self.llm_service = LLMService()
        self.kv_cache_manager = KVCacheManager(gpu_cache_capacity_mb=100) # 100MB GPU cache

        # Dummy knowledge base documents
        kb_docs = [
            "Our product features include real-time analytics, cloud storage, and secure authentication.",
            "Troubleshooting guide: If your device is not powering on, please check the power cable and try a different outlet.",
            "FAQs: How do I reset my password? Go to settings -> security -> reset password.",
            "Pricing plans start at $9.99/month for the basic package, with premium options available.",
            "Contact support at support@example.com or call 1-800-CHATBOT."
        ]
        self.knowledge_base_service = KnowledgeBaseService(documents=kb_docs)

        self.session_history: Dict[str, List[str]] = {}
        self.session_kv_node_ids: Dict[str, List[str]] = {}
        logger.info("ChatbotService initialized.")

    def process_query(self, session_id: str, user_text: str) -> ChatResponse:
        logger.info(f"Processing query for session {session_id}: '{user_text}'")
        start_time = time.time()

        self.session_history.setdefault(session_id, []).append(f"User: {user_text}")
        self.session_kv_node_ids.setdefault(session_id, [])

        # 1. Embed user query
        query_embedding = self.knowledge_base_service.embed_query(user_text)

        # 2. Knowledge Base Retrieval (RAG)
        retrieved_docs = self.knowledge_base_service.retrieve_documents(query_embedding)
        kb_context = "\n".join(retrieved_docs)

        # 3. Conversational History & KV Cache Management
        conversation_context = "\n".join(self.session_history[session_id])

        # Simulate KV cache for each turn/node in history
        kv_cache_tensors_for_llm: List[torch.Tensor] = []
        kv_hits = 0
        kv_misses = 0

        # Simulate generating or retrieving KV states for current turn and historical turns
        # For demonstration, let's treat each historical message and the current input as potential KV nodes.
        # In a real LLM, KV states are generated per token or per layer.

        # First, try to retrieve KV states for previous turns
        for node_id in self.session_kv_node_ids[session_id]:
            retrieved_kv = self.kv_cache_manager.retrieve_from_cache(node_id)
            if retrieved_kv:
                kv_cache_tensors_for_llm.extend(retrieved_kv)
                kv_hits += 1
            else:
                kv_misses += 1

        # Simulate generating new KV states for the current user query and context
        # For simplicity, create dummy KV tensors. In a real LLM, these would be actual key/value tensors.
        new_kv_node_id = f"{session_id}_turn_{len(self.session_history[session_id])}"
        # Simulate 2 tensors (key and value) of some size, e.g., 64x128 for a layer
        dummy_key_tensor = torch.rand(64, 128, device="cuda") if torch.cuda.is_available() else torch.rand(64, 128)
        dummy_value_tensor = torch.rand(64, 128, device="cuda") if torch.cuda.is_available() else torch.rand(64, 128)
        new_kv_tensors = [dummy_key_tensor, dummy_value_tensor]

        self.kv_cache_manager.add_to_cache(new_kv_node_id, new_kv_tensors)
        self.session_kv_node_ids[session_id].append(new_kv_node_id)
        kv_cache_tensors_for_llm.extend(new_kv_tensors)

        # 4. LLM Input Preparation
        llm_prompt = f"Context from knowledge base:\n{kb_context}\n\nConversation history:\n{conversation_context}\n\nUser: {user_text}\nChatbot:"

        # 5. LLM Inference
        llm_response = self.llm_service.generate_response(llm_prompt, kv_cache_tensors_for_llm)

        # 6. Store Chatbot response in history
        self.session_history[session_id].append(f"Chatbot: {llm_response}")

        end_time = time.time()
        logger.info(f"Query for session {session_id} processed in {end_time - start_time:.2f}s. KV Hits: {kv_hits}, KV Misses: {kv_misses}")
        logger.debug(f"KV Cache Stats: {self.kv_cache_manager.get_cache_stats()}")

        return ChatResponse(
            session_id=session_id,
            response=llm_response,
            cached_kv_hits=kv_hits,
            cached_kv_misses=kv_misses
        )

# --- FastAPI Application ---
app = FastAPI(title="Customer Support Chatbot with Swap-Out-Only-Once KV Cache")
chatbot_service = ChatbotService()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(query: UserQuery):
    return chatbot_service.process_query(query.session_id, query.text)

@app.get("/health")
async def health_check():
    return {"status": "ok", "kv_cache_stats": chatbot_service.kv_cache_manager.get_cache_stats()}
