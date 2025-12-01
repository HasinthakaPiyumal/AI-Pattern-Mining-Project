import streamlit as st
from fastapi import FastAPI
import uvicorn
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import requests
import uuid # For more robust session IDs

# --- 1. KV Cache Management Layer (Core of the Pattern) ---

class GPUKVCacheManager:
    """Manages KV tensors in 'GPU' memory."""
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.cache = {}  # {node_id: kv_tensors_on_gpu}
        print(f"GPUKVCacheManager initialized on device: {self.device}")

    def _simulate_gpu_transfer(self, kv_tensors):
        """Simulates fast transfer to/from GPU."""
        # For demonstration, ensure tensors are on the correct device
        if self.device == "cuda":
            # Simulate actual GPU transfer if CUDA is available
            return tuple( (k.to(self.device), v.to(self.device)) for k, v in kv_tensors)
        return kv_tensors # No actual transfer if on CPU

    def allocate(self, node_id, kv_tensors):
        """Allocates (or copies) KV tensors to GPU memory."""
        if node_id not in self.cache:
            print(f"GPU: Allocating/Copying node {node_id} to GPU.")
            self.cache[node_id] = self._simulate_gpu_transfer(kv_tensors)
            return True
        return False # Already on GPU

    def retrieve(self, node_id):
        """Retrieves KV tensors from GPU memory."""
        if node_id in self.cache:
            return self.cache[node_id]
        return None

    def free(self, node_id):
        """Frees KV tensors from GPU memory."""
        if node_id in self.cache:
            print(f"GPU: Freeing node {node_id} from GPU.")
            del self.cache[node_id]
            if self.device == "cuda":
                torch.cuda.empty_cache() # Clear CUDA memory
            return True
        return False

    def contains(self, node_id):
        return node_id in self.cache

class HostKVCacheManager:
    """Manages KV tensors in 'Host' (CPU) memory."""
    def __init__(self):
        self.cache = {}  # {node_id: kv_tensors_on_cpu}
        self.device = "cpu"
        print("HostKVCacheManager initialized on CPU.")

    def _simulate_host_transfer(self, kv_tensors):
        """Simulates slower transfer to/from Host."""
        # Ensure tensors are on CPU when stored in host cache
        return tuple( (k.to(self.device), v.to(self.device)) for k, v in kv_tensors)

    def store(self, node_id, kv_tensors):
        """Stores KV tensors to host memory (swap-out-only-once)."""
        if node_id not in self.cache:
            print(f"Host: Storing node {node_id} to Host (first time).")
            self.cache[node_id] = self._simulate_host_transfer(kv_tensors)
            return True
        print(f"Host: Node {node_id} already exists on Host. No copy needed.")
        return False # Already exists, no need to copy again

    def retrieve(self, node_id):
        """Retrieves KV tensors from host memory."""
        if node_id in self.cache:
            return self.cache[node_id]
        return None

    def free(self, node_id):
        """Frees KV tensors from host memory."""
        if node_id in self.cache:
            print(f"Host: Freeing node {node_id} from Host.")
            del self.cache[node_id]
            return True
        return False

    def contains(self, node_id):
        return node_id in self.cache

class KVCache:
    """Implements the Swap-Out-Only-Once Cache Strategy."""
    def __init__(self, gpu_capacity=3, host_capacity=5):
        self.gpu_manager = GPUKVCacheManager()
        self.host_manager = HostKVCacheManager()
        self.gpu_capacity = gpu_capacity # Number of nodes (sessions)
        self.host_capacity = host_capacity # Number of nodes (sessions)
        self.gpu_lru = [] # Simple LRU for GPU eviction: list of node_ids
        self.lock = threading.Lock() # For thread-safe operations

    def _evict_gpu_lru_if_needed(self):
        """Evicts the least recently used item from GPU if capacity is exceeded."""
        while len(self.gpu_lru) >= self.gpu_capacity:
            node_to_evict = self.gpu_lru.pop(0) # Get LRU
            print(f"Cache: GPU capacity full. Evicting LRU node {node_to_evict} from GPU.")
            self._evict_from_gpu(node_to_evict)
        return None

    def _evict_from_gpu(self, node_id):
        """Evicts a specific node from GPU memory, applying the swap-out-only-once logic."""
        if self.gpu_manager.contains(node_id):
            kv_tensors = self.gpu_manager.retrieve(node_id) # Get for potential host copy (still on GPU)
            if not self.host_manager.contains(node_id):
                # First eviction from GPU: copy to host
                print(f"Cache: Node {node_id} evicted from GPU for the FIRST time. Copying to Host.")
                self.host_manager.store(node_id, kv_tensors) # Store CPU version of tensors
            else:
                # Subsequent eviction: host already has a copy, no need to copy again
                print(f"Cache: Node {node_id} evicted from GPU. Host already has a copy. Skipping copy to Host.")
            self.gpu_manager.free(node_id);
            # Remove from LRU list if it was there (should be before actual eviction)
            if node_id in self.gpu_lru:
                self.gpu_lru.remove(node_id)

    def get_or_load(self, node_id, kv_tensors_generator_func=None):
        """
        Attempts to get KV tensors for a node.
        If not on GPU, checks host. If on host, moves to GPU.
        If not anywhere, generates (if generator provided) and places on GPU.
        """
        with self.lock:
            # 1. Try to retrieve from GPU
            kv_tensors = self.gpu_manager.retrieve(node_id)
            if kv_tensors is not None:
                # Update LRU: move to end (most recently used)
                if node_id in self.gpu_lru:
                    self.gpu_lru.remove(node_id)
                self.gpu_lru.append(node_id)
                print(f"Cache: Node {node_id} found on GPU.")
                return kv_tensors

            # 2. Not on GPU, try to retrieve from Host
            kv_tensors_on_host = self.host_manager.retrieve(node_id)
            if kv_tensors_on_host is not None:
                print(f"Cache: Node {node_id} found on Host. Moving to GPU.")
                # Make space on GPU if needed
                self._evict_gpu_lru_if_needed()
                # Move from host (CPU) to GPU
                self.gpu_manager.allocate(node_id, kv_tensors_on_host)
                self.gpu_lru.append(node_id)
                return self.gpu_manager.retrieve(node_id) # Return the GPU version

            # 3. Not on GPU or Host, generate new (if function provided) and put on GPU
            if kv_tensors_generator_func:
                print(f"Cache: Node {node_id} not found anywhere. Generating new KV tensors and placing on GPU.")
                new_kv_tensors = kv_tensors_generator_func()
                # Make space on GPU if needed
                self._evict_gpu_lru_if_needed()
                self.gpu_manager.allocate(node_id, new_kv_tensors)
                self.gpu_lru.append(node_id)
                return self.gpu_manager.retrieve(node_id)

            print(f"Cache: Node {node_id} not found anywhere and no generator provided.")
            return None

    def update(self, node_id, new_kv_tensors):
        """Updates KV tensors for a node, always putting it on GPU."""
        with self.lock:
            print(f"Cache: Updating node {node_id} on GPU.")
            # If exists on GPU, free old one
            if self.gpu_manager.contains(node_id):
                self.gpu_manager.free(node_id)
            # Remove from LRU list if it was there (it's being updated, so effectively 'new')
            if node_id in self.gpu_lru:
                self.gpu_lru.remove(node_id)

            # Make space on GPU if needed before allocating new
            self._evict_gpu_lru_if_needed()

            # Allocate new on GPU
            self.gpu_manager.allocate(node_id, new_kv_tensors)
            self.gpu_lru.append(node_id) # Add to LRU as most recently used

            # Note: According to the pattern, host copy is *not* updated here.
            # It retains its "only once" stored state until the node is fully evicted.

    def evict_from_cache(self, node_id):
        """Evicts a node completely from both GPU and Host cache."""
        with self.lock:
            print(f"Cache: Evicting node {node_id} entirely from GPU and Host.")
            self.gpu_manager.free(node_id)
            self.host_manager.free(node_id)
            if node_id in self.gpu_lru:
                self.gpu_lru.remove(node_id)

    def get_cache_status(self):
        with self.lock:
            gpu_keys = list(self.gpu_manager.cache.keys())
            host_keys = list(self.host_manager.cache.keys())
            return {
                "gpu_cache_count": len(gpu_keys),
                "host_cache_count": len(host_keys),
                "gpu_keys": gpu_keys,
                "host_keys": host_keys,
                "gpu_lru_order": list(self.gpu_lru)
            }

# --- 2. LLM Chatbot Service ---

class LLMChatbot:
    def __init__(self, cache: KVCache):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cache = cache
        self.conversation_history = {} # {session_id: list_of_input_ids}
        self.kv_cache_nodes = {} # {session_id: current_kv_node_id} - one node per session for this demo
        print(f"LLMChatbot initialized on device: {self.device}")

        # Set pad_token_id for generation, if not already set (common for DialoGPT)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def _generate_dummy_kv_tensors(self, sequence_length):
        """
        Simulates generating KV tensors with a realistic shape for DialoGPT-small.
        Returns a tuple of tuples: ((key_layer_0, value_layer_0), ..., (key_layer_N, value_layer_N))
        Each key/value tensor: (batch_size, num_heads, sequence_length, head_dim)
        """
        dummy_kv = []
        batch_size = 1 # Assuming single user session
        num_heads = self.model.config.n_head
        hidden_size = self.model.config.hidden_size
        head_dim = hidden_size // num_heads

        for _ in range(self.model.config.n_layer):
            key_tensor = torch.randn(batch_size, num_heads, sequence_length, head_dim)
            value_tensor = torch.randn(batch_size, num_heads, sequence_length, head_dim)
            dummy_kv.append((key_tensor, value_tensor))
        return tuple(dummy_kv)

    def chat(self, session_id: str, user_message: str):
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
            self.kv_cache_nodes[session_id] = f"session_{session_id}" # Each session gets one KV cache node

        # Tokenize new input
        new_input_ids = self.tokenizer.encode(user_message + self.tokenizer.eos_token, return_tensors='pt').to(self.device)
        self.conversation_history[session_id].append(new_input_ids)

        # Concatenate history for the current turn (full prompt)
        full_input_ids = torch.cat(self.conversation_history[session_id], dim=-1)

        current_node_id = self.kv_cache_nodes[session_id]

        # In a real streaming LLM, `past_key_values` would be passed directly to the model's forward pass.
        # For this demo, we'll simulate getting/updating the entire KV state for a node.
        # `get_or_load` will handle loading from GPU/Host or generating new KVs.
        past_kv_tensors = self.cache.get_or_load(current_node_id,
                                                kv_tensors_generator_func=lambda: self._generate_dummy_kv_tensors(full_input_ids.shape[-1]))

        # Perform LLM inference
        # For DialoGPT's `generate`, it re-processes the entire `input_ids`.
        # The KV cache optimization is more relevant for iterative decoding where
        # `past_key_values` are explicitly passed and returned.
        # Here, we simulate the *management* of these `past_key_values` using our cache.
        with torch.no_grad():
            outputs = self.model.generate(
                full_input_ids,
                max_new_tokens=50,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
                # In a real optimized LLM setup, you'd pass past_key_values
                # and receive new_past_key_values.
                # E.g., `outputs, new_past_key_values = self.model(..., past_key_values=past_kv_tensors)`
            )

        # Extract generated response
        response_ids = outputs[:, full_input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # Simulate the LLM producing new KV tensors after generation.
        # This is where a real LLM would return `new_past_key_values`.
        # We update our cache with these "new" KVs for the current session's node.
        new_kv_tensors_from_llm = self._generate_dummy_kv_tensors(outputs.shape[-1])
        self.cache.update(current_node_id, new_kv_tensors_from_llm)

        # Add the bot's response to the conversation history for the next turn
        self.conversation_history[session_id].append(response_ids)

        return response_text

    def end_session(self, session_id: str):
        if session_id in self.kv_cache_nodes:
            node_id = self.kv_cache_nodes[session_id]
            self.cache.evict_from_cache(node_id) # Evict completely
            del self.kv_cache_nodes[session_id]
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
        print(f"Session {session_id} ended and its KV cache node evicted.")

# --- 3. FastAPI Backend ---

app = FastAPI()
# Initialize the KV cache and chatbot service globally
kv_cache = KVCache(gpu_capacity=3, host_capacity=5) # Smaller capacity for demo
chatbot_service = LLMChatbot(cache=kv_cache)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(session_id: str, message: str):
    response = chatbot_service.chat(session_id, message)
    cache_status = kv_cache.get_cache_status()
    return {"response": response, "cache_status": cache_status}

@app.post("/end_session")
async def end_session_endpoint(session_id: str):
    chatbot_service.end_session(session_id)
    cache_status = kv_cache.get_cache_status()
    return {"message": f"Session {session_id} ended.", "cache_status": cache_status}

# --- 4. Streamlit Frontend ---

def run_streamlit():
    st.title("Intelligent Customer Support Chatbot")
    st.write("Leveraging Swap-Out-Only-Once KV Cache Strategy")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4()) # More robust unique ID
        st.session_state.messages = []
        st.session_state.initial_cache_status = kv_cache.get_cache_status() # Get initial status

    st.sidebar.header("Cache Status")
    # Display the current cache status dynamically
    if "current_cache_status" not in st.session_state:
        st.session_state.current_cache_status = kv_cache.get_cache_status()
    st.sidebar.json(st.session_state.current_cache_status)


    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    fastapi_url = "http://localhost:8000/chat"
                    params = {"session_id": st.session_state.session_id, "message": prompt}
                    response = requests.post(fastapi_url, params=params).json()
                    bot_response = response["response"]
                    st.markdown(bot_response)
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    st.session_state.current_cache_status = response["cache_status"] # Update sidebar cache status
                    st.experimental_rerun() # Rerun to update sidebar immediately
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Make sure it's running.")
                except Exception as e:
                    st.error(f"Error communicating with backend: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {e}"})

    if st.sidebar.button("End Session and Clear Cache"):
        try:
            fastapi_url = "http://localhost:8000/end_session"
            params = {"session_id": st.session_state.session_id}
            response = requests.post(fastapi_url, params=params).json()
            st.sidebar.success(response["message"])
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4()) # Generate new session ID
            st.session_state.current_cache_status = response["cache_status"] # Update sidebar cache status
            st.experimental_rerun() # Rerun to clear chat and update sidebar
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Make sure it's running.")
        except Exception as e:
            st.error(f"Error ending session: {e}")

# --- Main entry point to run both FastAPI and Streamlit ---

def run_fastapi():
    # Use a custom server for FastAPI if running in a thread to avoid Streamlit's server conflicts
    # `log_level="warning"` to reduce console noise
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

if __name__ == "__main__":
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    # Give FastAPI a moment to start up
    time.sleep(3) # Increased sleep time for FastAPI to fully initialize

    # Run Streamlit (this will block)
    run_streamlit()