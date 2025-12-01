import os
import json
import uuid
import time
import torch
from collections import deque
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Constants ---
HOST_KV_CACHE_DIR = "host_kv_cache"
GPU_CACHE_CAPACITY = 2  # Simulate GPU can hold KV cache for 2 conversations

# --- Setup Host KV Cache Directory ---
os.makedirs(HOST_KV_CACHE_DIR, exist_ok=True)

# --- Data Models ---
class ConversationContext(BaseModel):
    conversation_id: str
    history: List[Dict[str, str]] = []
    kv_cache_path: Optional[str] = None
    is_kv_cache_in_host_memory: bool = False

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    response_text: str
    conversation_id: str
    kv_cache_status: str

# --- Mock LLM Inference Service ---
class MockVLLM:
    def __init__(self, gpu_cache_capacity: int):
        self.gpu_cache_capacity = gpu_cache_capacity
        self.gpu_kv_cache: Dict[str, torch.Tensor] = {}
        self.lru_queue: deque = deque()  # Stores conversation_ids in LRU order (most recently used at front)
        self.mock_kv_tensor_size = (4, 1024) # Simulate some tensor size

    def _evict_lru(self):
        if not self.lru_queue:
            return None, None
        evicted_conversation_id = self.lru_queue.pop()
        evicted_tensors = self.gpu_kv_cache.pop(evicted_conversation_id)
        print(f"[MockVLLM] Evicted conversation {evicted_conversation_id} from GPU cache (LRU).")
        return evicted_conversation_id, evicted_tensors

    def generate(self, conversation_id: str, prompt: str, kv_tensors_from_host: Optional[torch.Tensor] = None):
        evicted_kv_cache_info = None

        # Simulate loading KV cache if provided from host
        if kv_tensors_from_host is not None:
            print(f"[MockVLLM] Loading KV cache for {conversation_id} from host to GPU.")
            # If it was already in GPU, update its LRU position
            if conversation_id in self.gpu_kv_cache:
                self.lru_queue.remove(conversation_id)
                self.lru_queue.appendleft(conversation_id)
                self.gpu_kv_cache[conversation_id] = kv_tensors_from_host # Simulate updating it
            else:
                # If not in GPU, add it, potentially evicting others
                if len(self.gpu_kv_cache) >= self.gpu_cache_capacity:
                    evicted_id, evicted_tensors = self._evict_lru()
                    if evicted_id:
                        evicted_kv_cache_info = {
                            "conversation_id": evicted_id,
                            "tensors": evicted_tensors
                        }
                self.gpu_kv_cache[conversation_id] = kv_tensors_from_host
                self.lru_queue.appendleft(conversation_id)

        else: # No KV cache from host, simulate fresh generation or reuse existing GPU cache
            if conversation_id in self.gpu_kv_cache:
                # Already in GPU, just update LRU position
                self.lru_queue.remove(conversation_id)
                self.lru_queue.appendleft(conversation_id)
                print(f"[MockVLLM] Reusing KV cache for {conversation_id} from GPU.")
            else:
                # Not in GPU and not in host (or first time), generate new KV and potentially evict
                if len(self.gpu_kv_cache) >= self.gpu_cache_capacity:
                    evicted_id, evicted_tensors = self._evict_lru()
                    if evicted_id:
                        evicted_kv_cache_info = {
                            "conversation_id": evicted_id,
                            "tensors": evicted_tensors
                        }
                # Simulate new KV cache generation for the current conversation
                self.gpu_kv_cache[conversation_id] = torch.rand(self.mock_kv_tensor_size)
                self.lru_queue.appendleft(conversation_id)
                print(f"[MockVLLM] Generated new KV cache for {conversation_id} on GPU.")

        # Simulate LLM response
        response_text = f"Mock LLM response to: '{prompt}' for conversation {conversation_id}."
        time.sleep(0.1) # Simulate some processing time

        return response_text, evicted_kv_cache_info

# --- Conversation Manager ---
class ConversationManager:
    def __init__(self, host_kv_cache_dir: str):
        self.host_kv_cache_dir = host_kv_cache_dir
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self._load_all_contexts()

    def _get_context_file_path(self, conversation_id: str) -> str:
        return os.path.join(self.host_kv_cache_dir, f"{conversation_id}_context.json")

    def _get_kv_tensor_file_path(self, conversation_id: str) -> str:
        return os.path.join(self.host_kv_cache_dir, f"{conversation_id}_kv_cache.pt")

    def _load_all_contexts(self):
        for filename in os.listdir(self.host_kv_cache_dir):
            if filename.endswith("_context.json"):
                filepath = os.path.join(self.host_kv_cache_dir, filename)
                with open(filepath, "r") as f:
                    context_data = json.load(f)
                    self.conversation_contexts[context_data["conversation_id"]] = ConversationContext(**context_data)
        print(f"[Manager] Loaded {len(self.conversation_contexts)} existing conversation contexts.")

    def save_context(self, context: ConversationContext):
        filepath = self._get_context_file_path(context.conversation_id)
        with open(filepath, "w") as f:
            f.write(context.json(indent=2))
        self.conversation_contexts[context.conversation_id] = context
        print(f"[Manager] Saved context for conversation {context.conversation_id}.")

    def get_or_create_context(self, conversation_id: Optional[str] = None) -> ConversationContext:
        if conversation_id and conversation_id in self.conversation_contexts:
            return self.conversation_contexts[conversation_id]
        else:
            new_id = str(uuid.uuid4())
            new_context = ConversationContext(conversation_id=new_id)
            self.save_context(new_context)
            print(f"[Manager] Created new conversation context {new_id}.")
            return new_context

    def delete_conversation(self, conversation_id: str):
        if conversation_id in self.conversation_contexts:
            context = self.conversation_contexts.pop(conversation_id)
            context_file = self._get_context_file_path(conversation_id)
            if os.path.exists(context_file):
                os.remove(context_file)
            if context.kv_cache_path and os.path.exists(context.kv_cache_path):
                os.remove(context.kv_cache_path)
            print(f"[Manager] Deleted conversation {conversation_id} and its associated files.")
        else:
            raise HTTPException(status_code=404, detail="Conversation not found.")

# --- FastAPI App Initialization ---
app = FastAPI()
mock_vllm = MockVLLM(gpu_cache_capacity=GPU_CACHE_CAPACITY)
conversation_manager = ConversationManager(host_kv_cache_dir=HOST_KV_CACHE_DIR)

# --- FastAPI Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    conversation_context = conversation_manager.get_or_create_context(request.conversation_id)
    request.conversation_id = conversation_context.conversation_id # Ensure response has ID

    kv_tensors_to_gpu = None
    kv_cache_status_message = ""

    # Check if KV cache is in host memory and load it if needed
    if conversation_context.is_kv_cache_in_host_memory and conversation_context.kv_cache_path:
        try:
            kv_tensors_to_gpu = torch.load(conversation_context.kv_cache_path)
            kv_cache_status_message = "Loaded KV cache from host memory."
            print(f"[API] Attempting to load KV cache for {conversation_context.conversation_id} from host.")
        except FileNotFoundError:
            # This case should ideally not happen if context is consistent, but handle it.
            conversation_context.is_kv_cache_in_host_memory = False
            conversation_context.kv_cache_path = None
            kv_cache_status_message = "KV cache path in context but file not found. Resetting."
            print(f"[API] Warning: KV cache file not found for {conversation_context.conversation_id}. Resetting context.")

    # Call mock LLM service
    response_text, evicted_kv_cache_info = mock_vllm.generate(
        conversation_id=conversation_context.conversation_id,
        prompt=request.message,
        kv_tensors_from_host=kv_tensors_to_gpu
    )

    # --- Implement Swap-Out-Only-Once Cache Strategy Logic ---
    if evicted_kv_cache_info:
        evicted_id = evicted_kv_cache_info["conversation_id"]
        evicted_tensors = evicted_kv_cache_info["tensors"]
        evicted_context = conversation_manager.conversation_contexts.get(evicted_id) # Get the full context

        if evicted_context and not evicted_context.is_kv_cache_in_host_memory:
            # FIRST EVICTION: Copy to host memory
            kv_file_path = conversation_manager._get_kv_tensor_file_path(evicted_id)
            torch.save(evicted_tensors, kv_file_path)
            evicted_context.kv_cache_path = kv_file_path
            evicted_context.is_kv_cache_in_host_memory = True
            conversation_manager.save_context(evicted_context)
            kv_cache_status_message += f" Evicted KV cache for {evicted_id} copied to host (first time)."
            print(f"[API] {evicted_id}: First eviction. Copied KV to host.")
        elif evicted_context and evicted_context.is_kv_cache_in_host_memory:
            # SUBSEQUENT EVICTION: Already in host, just free GPU (which MockVLLM already did)
            kv_cache_status_message += f" Evicted KV cache for {evicted_id} already on host. GPU memory freed."
            print(f"[API] {evicted_id}: Subsequent eviction. KV already on host. GPU freed.")
        else:
            kv_cache_status_message += f" Evicted KV cache for unknown/deleted conversation {evicted_id}."

    # Update current conversation's history
    conversation_context.history.append({"role": "user", "content": request.message})
    conversation_context.history.append({"role": "assistant", "content": response_text})
    conversation_manager.save_context(conversation_context)

    return ChatResponse(
        response_text=response_text,
        conversation_id=conversation_context.conversation_id,
        kv_cache_status=kv_cache_status_message.strip()
    )

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    try:
        conversation_manager.delete_conversation(conversation_id)
        return {"message": f"Conversation {conversation_id} and its cache deleted successfully."}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# To run this application:
# 1. Save the code as `main.py`.
# 2. Install dependencies: `pip install fastapi uvicorn pydantic torch`
# 3. Run from your terminal: `uvicorn main:app --reload`
# 4. Access the API at http://127.0.0.1:8000/docs for interactive documentation.