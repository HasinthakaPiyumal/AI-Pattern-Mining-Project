import torch
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer # Used for realistic token count, not full LLM inference
from loguru import logger

# --- 1. Pydantic Models for API Layer ---
class ChatRequest(BaseModel):
    customer_id: str
    message: str

class ChatResponse(BaseModel):
    customer_id: str
    response: str
    cache_status: str

# --- 2. Adaptive KV Cache Manager Components ---
@dataclass
class KVCacheEntry:
    key: Any
    kv_tensors: torch.Tensor = field(repr=False) # Hide large tensor from default repr
    location: str # "GPU" or "HOST"
    evicted_to_host_once: bool
    last_accessed: float
    size_bytes: int

class GPUMemoryPool:
    def __init__(self, max_size_bytes: int):
        self.max_size_bytes = max_size_bytes
        self._current_size_bytes = 0

    def allocate(self, size_bytes: int) -> bool:
        if self._current_size_bytes + size_bytes <= self.max_size_bytes:
            self._current_size_bytes += size_bytes
            logger.debug(f"GPU: Allocated {size_bytes} bytes. Current: {self._current_size_bytes}/{self.max_size_bytes}")
            return True
        logger.debug(f"GPU: Failed to allocate {size_bytes} bytes. Current: {self._current_size_bytes}/{self.max_size_bytes}")
        return False

    def free(self, size_bytes: int):
        self._current_size_bytes -= size_bytes
        if self._current_size_bytes < 0: self._current_size_bytes = 0 # Safety
        logger.debug(f"GPU: Freed {size_bytes} bytes. Current: {self._current_size_bytes}/{self.max_size_bytes}")

    def get_current_size(self) -> int:
        return self._current_size_bytes

class HostMemoryPool:
    def __init__(self):
        self._storage: Dict[Any, torch.Tensor] = {}

    def put(self, key: Any, kv_tensors: torch.Tensor):
        # Ensure it's on CPU before storing
        self._storage[key] = kv_tensors.cpu()
        logger.debug(f"Host: Stored KV for key {key}")

    def get(self, key: Any) -> torch.Tensor | None:
        return self._storage.get(key)

    def delete(self, key: Any):
        if key in self._storage:
            del self._storage[key]
            logger.debug(f"Host: Deleted KV for key {key}")

class AdaptiveKVCacheManager:
    def __init__(self, gpu_cache_size_mb: int):
        self.gpu_cache_size_bytes = gpu_cache_size_mb * 1024 * 1024
        self.gpu_memory_pool = GPUMemoryPool(self.gpu_cache_size_bytes)
        self.host_memory_pool = HostMemoryPool()
        self.gpu_cache_map: Dict[Any, KVCacheEntry] = {}
        self.host_cache_metadata: Dict[Any, KVCacheEntry] = {} # Stores metadata for host-resident entries

    def _get_tensor_size_bytes(self, tensors: torch.Tensor) -> int:
        # Simple approximation: num_elements * element_size_bytes
        return tensors.numel() * tensors.element_size()

    def get(self, key: Any) -> torch.Tensor | None:
        # 1. Check GPU cache
        if key in self.gpu_cache_map:
            entry = self.gpu_cache_map[key]
            entry.last_accessed = time.time()
            logger.info(f"Cache Hit (GPU) for key {key}")
            return entry.kv_tensors

        # 2. Check Host cache if not in GPU
        if key in self.host_cache_metadata:
            host_entry_meta = self.host_cache_metadata[key]
            kv_tensors_host = self.host_memory_pool.get(key)
            if kv_tensors_host is None: # Should not happen if metadata exists
                logger.error(f"Host metadata exists but tensor not found for key {key}")
                del self.host_cache_metadata[key]
                return None

            # Try to promote to GPU
            needed_size = self._get_tensor_size_bytes(kv_tensors_host)
            if self.gpu_memory_pool.allocate(needed_size):
                kv_tensors_gpu = kv_tensors_host.cuda()
                entry = KVCacheEntry(
                    key=key,
                    kv_tensors=kv_tensors_gpu,
                    location="GPU",
                    evicted_to_host_once=True, # It was on host, so it was evicted before
                    last_accessed=time.time(),
                    size_bytes=needed_size
                )
                self.gpu_cache_map[key] = entry
                logger.info(f"Cache Hit (Host -> GPU Promotion) for key {key}")
                return entry.kv_tensors
            else:
                # GPU full, cannot promote. Use directly from host (less efficient) or return None
                # For this implementation, we return None if it cannot be promoted to simulate memory pressure
                # In a real system, you might process directly from host or keep waiting for GPU space
                logger.warning(f"GPU full, cannot promote key {key} from host. Returning None.")
                return None

        logger.info(f"Cache Miss for key {key}")
        return None

    def put(self, key: Any, kv_tensors_cpu: torch.Tensor) -> torch.Tensor | None:
        # Ensure input tensors are on CPU first for consistent handling
        kv_tensors_cpu = kv_tensors_cpu.cpu()
        needed_size = self._get_tensor_size_bytes(kv_tensors_cpu)

        # Try to put directly into GPU
        if self.gpu_memory_pool.allocate(needed_size):
            kv_tensors_gpu = kv_tensors_cpu.cuda()
            entry = KVCacheEntry(
                key=key,
                kv_tensors=kv_tensors_gpu,
                location="GPU",
                evicted_to_host_once=False, # New in GPU, not yet evicted to host
                last_accessed=time.time(),
                size_bytes=needed_size
            )
            self.gpu_cache_map[key] = entry
            logger.info(f"KV data for key {key} placed in GPU.")
            return entry.kv_tensors
        else:
            # GPU full, need to evict or store directly to host
            logger.warning(f"GPU full. Attempting eviction for key {key}.")
            if not self._evict_lru_from_gpu(needed_size):
                logger.warning(f"Could not free enough GPU space for {key}. Storing only to Host.")
                # Store only to host
                self.host_memory_pool.put(key, kv_tensors_cpu)
                self.host_cache_metadata[key] = KVCacheEntry(
                    key=key,
                    kv_tensors=kv_tensors_cpu, # Metadata points to CPU tensor for size calculation if needed
                    location="HOST",
                    evicted_to_host_once=True, # It's now on host, so effectively evicted once
                    last_accessed=time.time(),
                    size_bytes=needed_size
                )
                return None # No GPU tensor returned
            else:
                # After eviction, retry placing in GPU
                return self.put(key, kv_tensors_cpu) # Recursive call, should succeed now or fail cleanly if still no space

    def _evict_lru_from_gpu(self, space_to_free: int) -> bool:
        if not self.gpu_cache_map:
            logger.warning("GPU cache is empty, cannot evict.")
            return False

        # Sort by last_accessed to find LRU
        sorted_keys = sorted(self.gpu_cache_map, key=lambda k: self.gpu_cache_map[k].last_accessed)
        
        freed_space = 0
        for key_to_evict in sorted_keys:
            entry = self.gpu_cache_map[key_to_evict]
            logger.info(f"Evicting LRU entry {key_to_evict} from GPU. (Size: {entry.size_bytes} bytes)")
            
            # Implement Swap-Out-Only-Once logic
            if not entry.evicted_to_host_once:
                # Copy to host memory pool for the first time
                self.host_memory_pool.put(key_to_evict, entry.kv_tensors.cpu())
                # Update metadata for this key in host_cache_metadata
                self.host_cache_metadata[key_to_evict] = KVCacheEntry(
                    key=key_to_evict,
                    kv_tensors=entry.kv_tensors.cpu(), # Store the CPU tensor in metadata
                    location="HOST",
                    evicted_to_host_once=True,
                    last_accessed=entry.last_accessed,
                    size_bytes=entry.size_bytes
                )
                logger.debug(f"Key {key_to_evict} copied to host memory for the first time.")
            else:
                logger.debug(f"Key {key_to_evict} already exists on host. Not copying again.")
            
            # Free GPU memory regardless
            self.gpu_memory_pool.free(entry.size_bytes)
            freed_space += entry.size_bytes
            del self.gpu_cache_map[key_to_evict]
            
            if freed_space >= space_to_free:
                logger.info(f"Evicted enough space ({freed_space} bytes) from GPU.")
                return True
        
        logger.warning(f"Failed to free enough space ({space_to_free} needed, {freed_space} freed) from GPU.")
        return False

    def clear_from_host(self, key: Any):
        if key in self.host_cache_metadata:
            self.host_memory_pool.delete(key)
            del self.host_cache_metadata[key]
            logger.info(f"Key {key} entirely cleared from host cache.")

# --- 3. LLM Inference Service (Simplified Simulation) ---
class LLMInferenceService:
    def __init__(self, model_name: str = "distilbert/distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"LLM Inference Service initialized on device: {self.device}")

    def generate_response_and_kv(self, conversation_history: List[str], kv_tensors_input: torch.Tensor | None) -> tuple[str, torch.Tensor]:
        # Simulate LLM processing
        full_context = " ".join(conversation_history)
        input_tokens = self.tokenizer.encode(full_context, return_tensors="pt").to(self.device)
        
        # Simulate generating a response
        dummy_response = f"Hello! I am an AI assistant. You asked about: {' '.join(conversation_history[-1].split()[:5])}...\nYour context size: {input_tokens.numel()} tokens."
        
        # Simulate generating new KV tensors (on GPU for this example)
        # Shape and size depend on model architecture and sequence length
        # For simplicity, let's make it proportional to input_tokens length
        kv_tensor_size = input_tokens.numel() * 1024 # Example: 1KB per token
        # Ensure a minimum size to make caching relevant
        kv_tensor_size = max(kv_tensor_size, 1024 * 10) # Minimum 10KB
        
        # Create a dummy tensor on the device
        # A common KV cache shape is (batch_size, num_heads, sequence_length, head_dim)
        # We'll just use a flat tensor for size calculation simplicity
        dummy_kv_tensors = torch.rand(kv_tensor_size // 4, dtype=torch.float32).to(self.device) # float32 = 4 bytes
        
        logger.info(f"LLM generated dummy KV tensors of size: {kv_tensor_size} bytes")
        return dummy_response, dummy_kv_tensors

# --- 4. Conversation Context and Session Management ---
customer_sessions: Dict[str, Dict[str, Any]] = {}

# --- 5. FastAPI Application ---
app = FastAPI(
    title="Intelligent Customer Support LLM with Adaptive KV Cache",
    description="Customer support LLM using a 'Swap-Out-Only-Once' KV cache strategy."
)

# Global instances
GPU_CACHE_SIZE_MB = 100 # Example: 100 MB GPU cache
kv_cache_manager = AdaptiveKVCacheManager(gpu_cache_size_mb=GPU_CACHE_SIZE_MB)
llm_service = LLMInferenceService()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    customer_id = request.customer_id
    user_message = request.message
    
    # Initialize session if new
    if customer_id not in customer_sessions:
        customer_sessions[customer_id] = {
            "conversation_history": [],
            "current_kv_cache_key": customer_id, # Using customer_id as key for simplicity
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    session = customer_sessions[customer_id]
    session["conversation_history"].append(f"User: {user_message}")
    current_conversation = session["conversation_history"]
    kv_key = session["current_kv_cache_key"]
    
    cache_status_message = ""
    
    # Try to retrieve KV cache
    kv_tensors_from_cache = kv_cache_manager.get(kv_key)
    if kv_tensors_from_cache is not None:
        cache_status_message = "Cache Hit (KV re-used)"
        session["cache_hits"] += 1
    else:
        cache_status_message = "Cache Miss (New KV generated)"
        session["cache_misses"] += 1
        logger.info(f"New session or KV not found for {kv_key}. Starting fresh.")
        # If not found, LLM will generate new KV from scratch
        kv_tensors_from_cache = None 

    # Simulate LLM inference
    llm_response, new_kv_tensors_gpu = llm_service.generate_response_and_kv(
        conversation_history=current_conversation,
        kv_tensors_input=kv_tensors_from_cache
    )
    session["conversation_history"].append(f"AI: {llm_response}")
    
    # Store the new KV tensors generated by the LLM
    # The put method handles GPU/Host placement based on the strategy
    kv_cache_manager.put(kv_key, new_kv_tensors_gpu.cpu()) # Pass CPU tensor to put

    logger.info(f"Customer {customer_id} - Cache Hits: {session['cache_hits']}, Misses: {session['cache_misses']}")
    logger.info(f"GPU Cache Usage: {kv_cache_manager.gpu_memory_pool.get_current_size() / (1024*1024):.2f}/{GPU_CACHE_SIZE_MB} MB")
    
    return ChatResponse(
        customer_id=customer_id,
        response=llm_response,
        cache_status=cache_status_message
    )

@app.post("/clear_customer_cache/{customer_id}")
async def clear_customer_cache_endpoint(customer_id: str):
    if customer_id in customer_sessions:
        kv_key = customer_sessions[customer_id]["current_kv_cache_key"]
        kv_cache_manager.clear_from_host(kv_key)
        # Also remove from GPU if it happens to be there (though get/put handles this implicitly)
        if kv_key in kv_cache_manager.gpu_cache_map:
            entry = kv_cache_manager.gpu_cache_map[kv_key]
            kv_cache_manager.gpu_memory_pool.free(entry.size_bytes)
            del kv_cache_manager.gpu_cache_map[kv_key]
            logger.info(f"Key {kv_key} also cleared from GPU.")
        del customer_sessions[customer_id]
        logger.info(f"Customer {customer_id} session and cache cleared.")
        return {"message": f"Cache for customer {customer_id} cleared."}
    return {"message": f"Customer {customer_id} not found or no cache to clear.", "status": 404}


# To run this application:
# 1. Save the code as `customer_support_llm_cache.py`
# 2. Install dependencies: `pip install fastapi uvicorn torch transformers loguru pydantic`
# 3. Run: `uvicorn customer_support_llm_cache:app --reload`
# 4. Access the API at http://127.0.0.1:8000/docs