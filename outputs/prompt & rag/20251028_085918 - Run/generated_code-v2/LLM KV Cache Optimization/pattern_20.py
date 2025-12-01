import torch
import collections
import logging
from typing import Optional, Dict, Set
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SwapOutOnlyOnceCache:
    def __init__(self, gpu_capacity: int, host_capacity: int):
        self.gpu_capacity = gpu_capacity
        self.host_capacity = host_capacity
        self.gpu_cache: collections.OrderedDict[str, torch.Tensor] = collections.OrderedDict()
        self.host_cache: Dict[str, torch.Tensor] = {}
        self.evicted_to_host_once: Set[str] = set()
        logging.info(f"Cache initialized: GPU capacity={gpu_capacity}, Host capacity={host_capacity}")

    def _evict_from_gpu_to_host(self):
        if not self.gpu_cache:
            return

        conversation_id, kv_tensors = self.gpu_cache.popitem(last=False)
        
        if conversation_id not in self.evicted_to_host_once:
            if len(self.host_cache) >= self.host_capacity:
                # Simple host eviction - in a real scenario, this would also be an LRU or similar
                # For this pattern, host retains until full cache eviction. If host is full, we'd need a host eviction strategy.
                # For now, let's assume host capacity is large enough or we evict a random item if needed.
                logging.warning(f"Host cache full. Evicting {next(iter(self.host_cache))} from host cache.")
                del self.host_cache[next(iter(self.host_cache))]
                
            self.host_cache[conversation_id] = kv_tensors
            self.evicted_to_host_once.add(conversation_id)
            logging.info(f"GPU evicted '{conversation_id}' to host for the FIRST time. Data copied.")
        else:
            logging.info(f"GPU evicted '{conversation_id}'. Data already on host, GPU memory freed.")

    def get_kv_cache(self, conversation_id: str) -> Optional[torch.Tensor]:
        if conversation_id in self.gpu_cache:
            kv_tensors = self.gpu_cache.pop(conversation_id)
            self.gpu_cache[conversation_id] = kv_tensors  # Move to end (most recently used)
            logging.info(f"'{conversation_id}' KV cache found in GPU cache.")
            return kv_tensors

        if conversation_id in self.host_cache:
            if len(self.gpu_cache) >= self.gpu_capacity:
                self._evict_from_gpu_to_host()

            kv_tensors = self.host_cache[conversation_id]
            # Data is moved from host to GPU. Host retains a copy based on the pattern description.
            self.gpu_cache[conversation_id] = kv_tensors
            logging.info(f"'{conversation_id}' KV cache found in Host cache, moved to GPU. Host retains copy.")
            return kv_tensors
        
        logging.info(f"'{conversation_id}' KV cache not found in any cache.")
        return None

    def update_kv_cache(self, conversation_id: str, new_kv_tensors: torch.Tensor):
        if conversation_id in self.gpu_cache:
            self.gpu_cache.pop(conversation_id)
        elif conversation_id in self.host_cache:
            # If it was in host, and now updated, it implies it's being used in GPU
            # So, ensure GPU has space and move it there.
            if len(self.gpu_cache) >= self.gpu_capacity:
                self._evict_from_gpu_to_host()
            # Do not remove from host_cache here, host retains copy based on the pattern description

        elif len(self.gpu_cache) >= self.gpu_capacity:
            self._evict_from_gpu_to_host()

        self.gpu_cache[conversation_id] = new_kv_tensors
        logging.info(f"'{conversation_id}' KV cache updated and placed/moved to GPU cache.")

    def evict_from_full_cache(self, conversation_id: str):
        if conversation_id in self.gpu_cache:
            self.gpu_cache.pop(conversation_id)
            logging.info(f"'{conversation_id}' KV cache evicted from GPU cache.")
        if conversation_id in self.host_cache:
            del self.host_cache[conversation_id]
            logging.info(f"'{conversation_id}' KV cache evicted from Host cache.")
        if conversation_id in self.evicted_to_host_once:
            self.evicted_to_host_once.remove(conversation_id)
            logging.info(f"'{conversation_id}' removed from 'evicted_to_host_once' tracking.")

class LLMService:
    def generate_response(self, conversation_id: str, prompt: str, kv_cache_manager: SwapOutOnlyOnceCache) -> str:
        # Simulate fetching or initializing KV cache
        kv_tensors = kv_cache_manager.get_kv_cache(conversation_id)

        if kv_tensors is None:
            # Simulate initial KV cache creation for a new conversation
            kv_tensors = torch.rand(1, 10, 512)  # Dummy KV tensors
            logging.info(f"Simulating new KV cache creation for '{conversation_id}'.")
        else:
            logging.info(f"Using existing KV cache for '{conversation_id}'.")

        # Simulate LLM processing
        response_text = f"Agent co-pilot response for '{prompt}' in conversation '{conversation_id}'."
        
        # Simulate updating KV cache after LLM processing
        new_kv_tensors = kv_tensors + torch.rand(1, 10, 512) * 0.1 # Simulate some update
        kv_cache_manager.update_kv_cache(conversation_id, new_kv_tensors)

        return response_text

class ChatRequest(BaseModel):
    conversation_id: str
    message: str

class ChatResponse(BaseModel):
    conversation_id: str
    response: str

app = FastAPI()

# Initialize cache manager and LLM service
kv_cache_manager = SwapOutOnlyOnceCache(gpu_capacity=3, host_capacity=10)
llm_service = LLMService()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    response = llm_service.generate_response(request.conversation_id, request.message, kv_cache_manager)
    return ChatResponse(conversation_id=request.conversation_id, response=response)

@app.post("/end_conversation")
async def end_conversation_endpoint(request: ChatRequest):
    kv_cache_manager.evict_from_full_cache(request.conversation_id)
    return {"message": f"Conversation {request.conversation_id} ended and cache evicted."}