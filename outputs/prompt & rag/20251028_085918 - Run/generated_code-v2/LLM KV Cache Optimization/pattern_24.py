
import torch
import time
import collections
import threading
from loguru import logger
from typing import Dict, Any, Optional

# --- 1. Cache Entry Metadata ---

class CacheEntryMetadata:
    def __init__(
        self, 
        kv_node_id: str, 
        size: int, 
        gpu_ref: Optional[torch.Tensor] = None,
        host_ref: Optional[torch.Tensor] = None,
        is_host_resident: bool = False
    ):
        self.kv_node_id = kv_node_id
        self.size = size
        self.gpu_ref = gpu_ref
        self.host_ref = host_ref
        self.is_host_resident = is_host_resident
        self.last_accessed = time.time()

    def update_access_time(self):
        self.last_accessed = time.time()


# --- 2. Hierarchical KV Cache Manager ---

class HierarchicalKVManager:
    def __init__(self, gpu_cache_capacity: int, host_cache_capacity: int):
        self.gpu_cache_capacity = gpu_cache_capacity  # in bytes (simulated)
        self.host_cache_capacity = host_cache_capacity  # in bytes (simulated)
        self.gpu_current_size = 0
        self.host_current_size = 0

        self.metadata: Dict[str, CacheEntryMetadata] = {}
        # LRU for GPU cache: stores kv_node_ids, ordered by last access
        self.gpu_lru = collections.OrderedDict() 
        
        self.lock = threading.Lock()
        logger.info(f"KV Cache Manager initialized with GPU capacity: {gpu_cache_capacity} bytes, Host capacity: {host_cache_capacity} bytes")

    def _get_tensor_size_in_bytes(self, tensor: torch.Tensor) -> int:
        return tensor.element_size() * tensor.nelement()

    def _free_gpu_memory_ref(self, kv_node_id: str):
        with self.lock:
            if kv_node_id in self.metadata and self.metadata[kv_node_id].gpu_ref is not None:
                entry = self.metadata[kv_node_id]
                logger.debug(f"Freeing GPU memory for node {kv_node_id} (size: {entry.size})")
                # In a real scenario, we'd explicitly free torch.Tensor memory if managed directly
                # For simulation, setting to None and adjusting size is sufficient.
                entry.gpu_ref = None
                self.gpu_current_size -= entry.size
                if kv_node_id in self.gpu_lru: 
                    self.gpu_lru.pop(kv_node_id)
                torch.cuda.empty_cache()

    def _free_host_memory_ref(self, kv_node_id: str):
        with self.lock:
            if kv_node_id in self.metadata and self.metadata[kv_node_id].host_ref is not None:
                entry = self.metadata[kv_node_id]
                logger.debug(f"Freeing Host memory for node {kv_node_id} (size: {entry.size})")
                entry.host_ref = None
                self.host_current_size -= entry.size

    def _evict_lru_from_gpu_to_host_if_needed(self):
        with self.lock:
            while self.gpu_current_size > self.gpu_cache_capacity and self.gpu_lru:
                lru_kv_node_id, _ = self.gpu_lru.popitem(last=False)  # Get and remove LRU item
                entry = self.metadata.get(lru_kv_node_id)

                if not entry or entry.gpu_ref is None: # Already evicted or doesn't exist
                    continue
                
                logger.info(f"Evicting LRU node {lru_kv_node_id} from GPU. Current GPU size: {self.gpu_current_size}/{self.gpu_cache_capacity}")

                if not entry.is_host_resident:
                    # First eviction: Copy to host memory
                    if self.host_current_size + entry.size > self.host_cache_capacity:
                        logger.error(f"Host cache full! Cannot evict {lru_kv_node_id} from GPU. Implementing host eviction policy might be needed.")
                        # In a real system, you'd implement a host eviction policy here.
                        # For this example, we'll just fail or not evict.
                        self.gpu_lru[lru_kv_node_id] = None # Put it back if we can't evict to host
                        return False # Indicate failure to evict to host
                    
                    logger.info(f"Copying KV tensors for {lru_kv_node_id} to host memory (first eviction).")
                    entry.host_ref = entry.gpu_ref.cpu()
                    entry.is_host_resident = True
                    self.host_current_size += entry.size
                else:
                    # Subsequent eviction: Host already has a copy, just free GPU memory
                    logger.info(f"Freeing GPU memory for {lru_kv_node_id}. Host already has a copy (swap-out-only-once).")
                
                self._free_gpu_memory_ref(lru_kv_node_id)
                logger.info(f"GPU cache size after eviction: {self.gpu_current_size}/{self.gpu_cache_capacity}")
            return True # Indicate success

    def add_or_update_kv_node(self, kv_node_id: str, kv_tensors: torch.Tensor) -> torch.Tensor:
        with self.lock:
            tensor_size = self._get_tensor_size_in_bytes(kv_tensors)
            if kv_node_id in self.metadata:
                entry = self.metadata[kv_node_id]
                logger.debug(f"Updating KV node {kv_node_id}. Old size: {entry.size}, New size: {tensor_size}")
                
                # If size changes, adjust current memory usage before updating
                if entry.gpu_ref is not None and entry.size != tensor_size:
                    self.gpu_current_size -= entry.size
                    self.gpu_current_size += tensor_size
                elif entry.gpu_ref is None and entry.host_ref is not None and entry.size != tensor_size:
                     # If only host resident, size change means we need to adjust host size (conceptual)
                     # For simplicity, we assume kv_tensors provided are always for GPU updates
                     pass 

                # Free old GPU ref if it exists
                if entry.gpu_ref is not None:
                    self._free_gpu_memory_ref(kv_node_id)

                entry.size = tensor_size # Update size
                entry.update_access_time()
            else:
                entry = CacheEntryMetadata(kv_node_id=kv_node_id, size=tensor_size)
                self.metadata[kv_node_id] = entry
                logger.info(f"Adding new KV node {kv_node_id} with size {tensor_size}.")

            # Ensure GPU capacity before adding new/updated tensor
            if self.gpu_current_size + tensor_size > self.gpu_cache_capacity:
                logger.warning(f"GPU cache full, attempting eviction for node {kv_node_id}.")
                if not self._evict_lru_from_gpu_to_host_if_needed():
                    logger.error(f"Failed to make space for {kv_node_id} in GPU. This node might not be cached in GPU.")
                    # If we can't make space, it means it can't be GPU resident right now.
                    # We could try to store it directly in host if not there, but for now
                    # we assume it must attempt to be GPU resident first if newly added/updated.
                    if not entry.is_host_resident:
                        if self.host_current_size + tensor_size > self.host_cache_capacity:
                            logger.critical(f"Cannot even fit new/updated node {kv_node_id} into host cache! Out of memory.")
                            raise MemoryError("System out of memory for KV cache.")
                        logger.warning(f"Adding new/updated node {kv_node_id} directly to host cache due to full GPU.")
                        entry.host_ref = kv_tensors.cpu()
                        entry.is_host_resident = True
                        self.host_current_size += tensor_size
                    return kv_tensors.cpu() # Return CPU tensor if it couldn't be GPU resident

            # Place/update in GPU memory
            entry.gpu_ref = kv_tensors.cuda()
            self.gpu_current_size += tensor_size
            self.gpu_lru[kv_node_id] = entry.last_accessed # Update LRU
            logger.debug(f"KV node {kv_node_id} is now GPU resident. Current GPU size: {self.gpu_current_size}/{self.gpu_cache_capacity}")
            return entry.gpu_ref

    def get_kv_node(self, kv_node_id: str) -> Optional[torch.Tensor]:
        with self.lock:
            entry = self.metadata.get(kv_node_id)
            if not entry:
                logger.debug(f"KV node {kv_node_id} not found in cache.")
                return None

            entry.update_access_time()
            self.gpu_lru.pop(kv_node_id, None) # Remove to re-insert at end (most recently used)
            self.gpu_lru[kv_node_id] = entry.last_accessed

            if entry.gpu_ref is not None:
                logger.debug(f"KV node {kv_node_id} found in GPU cache.")
                return entry.gpu_ref
            elif entry.is_host_resident and entry.host_ref is not None:
                logger.info(f"KV node {kv_node_id} found in Host cache. Promoting to GPU.")
                # Promote from host to GPU
                if self.gpu_current_size + entry.size > self.gpu_cache_capacity:
                    logger.warning(f"GPU cache full, attempting eviction for promotion of node {kv_node_id}.")
                    if not self._evict_lru_from_gpu_to_host_if_needed():
                        logger.error(f"Failed to make space for {kv_node_id} in GPU. Cannot promote.")
                        return entry.host_ref # Return host tensor if cannot promote
                
                entry.gpu_ref = entry.host_ref.cuda()
                self.gpu_current_size += entry.size
                logger.debug(f"KV node {kv_node_id} promoted to GPU. Current GPU size: {self.gpu_current_size}/{self.gpu_cache_capacity}")
                return entry.gpu_ref
            else:
                logger.debug(f"KV node {kv_node_id} found in metadata but not in GPU or Host memory refs.")
                return None

    def evict_from_system(self, kv_node_id: str):
        with self.lock:
            entry = self.metadata.pop(kv_node_id, None)
            if entry:
                self._free_gpu_memory_ref(kv_node_id)
                self._free_host_memory_ref(kv_node_id)
                logger.info(f"KV node {kv_node_id} completely evicted from system cache.")
            else:
                logger.debug(f"Attempted to evict non-existent KV node {kv_node_id}.")

    @property
    def gpu_utilization(self):
        with self.lock:
            return self.gpu_current_size / self.gpu_cache_capacity if self.gpu_cache_capacity > 0 else 0

    @property
    def host_utilization(self):
        with self.lock:
            return self.host_current_size / self.host_cache_capacity if self.host_cache_capacity > 0 else 0


# --- 3. LLM Inference Engine (Simplified) ---

class LLMInferenceEngine:
    def __init__(self, kv_cache_manager: HierarchicalKVManager, model_name: str = "dummy-llm"):
        self.kv_cache_manager = kv_cache_manager
        self.model_name = model_name
        logger.info(f"LLM Inference Engine initialized with model: {model_name}")

    def _generate_kv_tensors(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        # Simulate KV tensor generation by an LLM
        # In a real scenario, this would involve running a forward pass on the LLM
        # and extracting the KV cache. The size would depend on model hidden_size, num_heads, sequence_length.
        batch_size = 1
        seq_len = prompt_tokens.shape[0]
        hidden_size = 768 # Example hidden size
        num_layers = 12 # Example number of layers

        # Simulate KV tensors: [num_layers, batch_size, num_heads, seq_len, head_dim]
        # We'll simplify to a single tensor for a 'node'
        kv_tensor_shape = (num_layers, batch_size, 12, seq_len, hidden_size // 12) 
        
        # Use random data for simulation, ensure it's on CPU initially
        return torch.randn(kv_tensor_shape, dtype=torch.float16, device="cpu") 

    def process_query(self, conversation_id: str, prompt: str) -> str:
        logger.info(f"Processing query for conversation {conversation_id}: {prompt}")
        
        # Simulate tokenization
        prompt_tokens = torch.tensor([ord(c) for c in prompt], dtype=torch.long) 
        kv_node_id = f"conv_{conversation_id}_turn_{time.time()}" # Simple unique ID for this turn's KV cache

        # Try to get existing KV tensors for previous turns (simplified here for current turn)
        # In a real system, you'd iterate through relevant past turns or use a prefix-tree structure.
        cached_kv_tensors = self.kv_cache_manager.get_kv_node(conversation_id) # Using conversation_id as a cumulative node for simplicity
        if cached_kv_tensors is not None:
            logger.debug(f"Using cached KV tensors for conversation {conversation_id}.")
            # Simulate incorporating cached KV tensors
            pass

        # Generate new KV tensors for the current turn/full context
        new_kv_tensors = self._generate_kv_tensors(prompt_tokens)
        
        # Store/update the KV tensors in the hierarchical cache
        # For simplicity, we're treating the whole conversation's KV state as one 'node' that gets updated
        # A more granular approach would have a node per turn, or a complex prefix cache.
        gpu_resident_kv = self.kv_cache_manager.add_or_update_kv_node(conversation_id, new_kv_tensors)

        # Simulate LLM generation using the KV tensors
        response = f"LLM response to '{prompt}' for conversation {conversation_id}. KV cache status: GPU={self.kv_cache_manager.gpu_current_size}/{self.kv_cache_manager.gpu_cache_capacity} ({self.kv_cache_manager.gpu_utilization:.2%}), Host={self.kv_cache_manager.host_current_size}/{self.kv_cache_manager.host_cache_capacity} ({self.kv_cache_manager.host_utilization:.2%})."
        return response


# --- 4. FastAPI Backend (Simplified) ---

from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Customer Support LLM API")

# Global instance of KV Cache Manager and LLM Engine
kv_manager: Optional[HierarchicalKVManager] = None
llm_engine: Optional[LLMInferenceEngine] = None

class ChatRequest(BaseModel):
    conversation_id: str
    prompt: str

class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    cache_status: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    global kv_manager, llm_engine
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True, level="INFO") # Redirect loguru to print for simple execution
    logger.info("Starting up FastAPI application and initializing LLM components...")
    # Example capacities: 100MB GPU, 500MB Host
    kv_manager = HierarchicalKVManager(gpu_cache_capacity=100 * 1024 * 1024, host_cache_capacity=500 * 1024 * 1024)
    llm_engine = LLMInferenceEngine(kv_cache_manager=kv_manager)
    logger.info("LLM components initialized.")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if llm_engine is None or kv_manager is None:
        raise RuntimeError("LLM Engine or KV Manager not initialized.")

    response_text = llm_engine.process_query(request.conversation_id, request.prompt)
    
    cache_status = {
        "gpu_current_size": kv_manager.gpu_current_size,
        "gpu_capacity": kv_manager.gpu_cache_capacity,
        "gpu_utilization": f"{kv_manager.gpu_utilization:.2%}",
        "host_current_size": kv_manager.host_current_size,
        "host_capacity": kv_manager.host_cache_capacity,
        "host_utilization": f"{kv_manager.host_utilization:.2%}",
        "cached_nodes": len(kv_manager.metadata)
    }
    
    return ChatResponse(
        conversation_id=request.conversation_id,
        response=response_text,
        cache_status=cache_status
    )

# --- 5. Streamlit/Gradio UI (Conceptual client-side code) ---
# This part would typically be a separate file (e.g., app.py for Streamlit)
# but is included here conceptually to show interaction.

# import streamlit as st
# import requests

# FASTAPI_URL = "http://127.0.0.1:8000"

# st.title("Intelligent Customer Support Chatbot")

# if "conversation_id" not in st.session_state:
#     st.session_state.conversation_id = str(uuid.uuid4())
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("Type your message here..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             try:
#                 response = requests.post(
#                     f"{FASTAPI_URL}/chat",
#                     json={
#                         "conversation_id": st.session_state.conversation_id,
#                         "prompt": prompt
#                     }
#                 ).json()
#                 st.markdown(response["response"])
#                 st.write(f"Cache Status: {response["cache_status"]}")
#                 st.session_state.messages.append({"role": "assistant", "content": response["response"]})
#             except Exception as e:
#                 st.error(f"Error communicating with LLM: {e}")

# To run the FastAPI app:
# uvicorn customer_support_llm_cache:app --reload

# To run the Streamlit app (if in a separate file app.py):
# streamlit run app.py

