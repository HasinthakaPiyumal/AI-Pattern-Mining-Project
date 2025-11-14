
import collections
import time
import random

class KVCacheManager:
    """
    Conceptual KV Cache Manager simulating PagedAttention and Swap-Out-Only-Once.
    This is a simplified representation; real implementations are highly optimized
    C++/CUDA code within LLM serving frameworks like vLLM.
    """
    def __init__(self, gpu_cache_size_pages=1024, page_size=4096):
        self.gpu_cache_size_pages = gpu_cache_size_pages
        self.page_size = page_size
        self.gpu_cache = collections.OrderedDict() # {page_id: KV_tensor_data} - simulates GPU memory
        self.host_cache = {}                       # {page_id: KV_tensor_data} - simulates Host memory (Swap-Out-Only-Once)
        self.critical_nodes_replica = {}           # {node_id: KV_tensor_data} - simulates persistent host memory for critical nodes
        self.next_page_id = 0

    def _allocate_page(self, kv_data):
        page_id = self.next_page_id
        self.next_page_id += 1
        return page_id, kv_data

    def _evict_page(self):
        # Simulate LRU eviction from GPU cache
        if not self.gpu_cache:
            return None
        page_id_to_evict, kv_data_to_evict = self.gpu_cache.popitem(last=False)
        print(f"Evicting page {page_id_to_evict} from GPU cache.")

        # Swap-Out-Only-Once strategy: move to host cache if not already there
        if page_id_to_evict not in self.host_cache:
            self.host_cache[page_id_to_evict] = kv_data_to_evict
            print(f"Page {page_id_to_evict} moved to host cache (Swap-Out-Only-Once).")
        else:
            print(f"Page {page_id_to_evict} already in host cache, not re-copying.")
        return page_id_to_evict

    def get_or_create_kv_page(self, prefix_hash, kv_data):
        # Simulate PagedAttention: manage KV tensors at page granularity
        # For simplicity, `prefix_hash` will act as a page identifier here.
        # In a real scenario, this would involve complex page table management.
        if prefix_hash in self.gpu_cache:
            # Move to end to simulate LRU
            kv_data = self.gpu_cache.pop(prefix_hash)
            self.gpu_cache[prefix_hash] = kv_data
            print(f"KV data for prefix '{prefix_hash}' found in GPU cache.")
            return prefix_hash
        elif prefix_hash in self.host_cache:
            print(f"KV data for prefix '{prefix_hash}' found in Host cache. Swapping back to GPU.")
            # Simulate swapping back to GPU
            if len(self.gpu_cache) >= self.gpu_cache_size_pages:
                self._evict_page()
            kv_data = self.host_cache.pop(prefix_hash) # Remove from host cache when bringing to GPU
            self.gpu_cache[prefix_hash] = kv_data
            return prefix_hash
        else:
            print(f"Creating new KV data page for prefix '{prefix_hash}'.")
            if len(self.gpu_cache) >= self.gpu_cache_size_pages:
                self._evict_page() # Make space
            self.gpu_cache[prefix_hash] = kv_data # Use prefix_hash as a conceptual page_id for new data
            return prefix_hash

    def replicate_critical_node(self, node_id, kv_data):
        """Replicates a 'critical' KV node from GPU to persistent host memory."""
        self.critical_nodes_replica[node_id] = kv_data
        print(f"Critical KV node '{node_id}' replicated to persistent host memory.")

    def recover_critical_node(self, node_id):
        """Recovers a 'critical' KV node from persistent host memory."""
        if node_id in self.critical_nodes_replica:
            print(f"Recovering critical KV node '{node_id}' from persistent host memory.")
            return self.critical_nodes_replica[node_id]
        return None

class LLMInferenceEngine:
    """
    Simulated LLM Inference Engine incorporating KV Cache optimizations.
    """
    def __init__(self, model_name="dummy_llm", kv_cache_manager=None):
        self.model_name = model_name
        self.kv_cache_manager = kv_cache_manager if kv_cache_manager else KVCacheManager()
        self.prefix_kv_cache = {} # {prefix_string: prefix_hash} for KV Cache Reuse

    def _generate_kv_data(self, token_ids):
        # Simulate generating KV data for a given sequence of tokens
        # In a real LLM, this would be the output of attention layers.
        return f"KV_data_for_{ '_ '.join(map(str, token_ids))}_{random.randint(0, 1000)}"

    def _simulate_inference_step(self, prompt_tokens, use_kv_cache=False, cached_prefix_hash=None):
        # Simulate a single inference step, potentially using cached KV data
        # Returns a dummy response and potentially new KV data (or reference to cached)
        if use_kv_cache and cached_prefix_hash:
            print(f"Inference using cached KV data for prefix hash: {cached_prefix_hash}")
            # Simulate retrieving and using cached KV data
            kv_data = self.kv_cache_manager.gpu_cache.get(cached_prefix_hash) # direct access for simulation
            if not kv_data:
                print(f"Error: Cached KV data for {cached_prefix_hash} not found in GPU cache during inference.")
                # This would trigger a swap-in in a real system or regeneration.
                kv_data = self._generate_kv_data(prompt_tokens) # Fallback for simulation
        else:
            print(f"Running full inference for: {prompt_tokens}")
            kv_data = self._generate_kv_data(prompt_tokens)

        # Simulate some processing time
        time.sleep(0.01 + len(prompt_tokens) * 0.005)
        return f"Response for '{ ' '.join(prompt_tokens)}'", kv_data

    def infer(self, prompt):
        tokenized_prompt = prompt.lower().split() # Simple tokenization
        
        # KV Cache Reuse Logic
        longest_prefix_match = ""
        cached_prefix_hash = None
        for i in range(len(tokenized_prompt), 0, -1):
            prefix = " ".join(tokenized_prompt[:i])
            if prefix in self.prefix_kv_cache:
                longest_prefix_match = prefix
                cached_prefix_hash = self.prefix_kv_cache[prefix]
                print(f"Found KV cache reuse opportunity for prefix: '{longest_prefix_match}'")
                break

        if cached_prefix_hash:
            # Ensure the cached prefix is in the KV cache manager (PagedAttention will handle placement)
            # For simulation, we'll just check if it's "available" and assume PagedAttention deals with pages.
            print(f"Requesting KV cache manager to ensure prefix '{longest_prefix_match}' is in GPU for reuse.")
            # In a real system, the inference engine would get a page table reference.
            # Here, we pass the hash to ensure the manager handles it.
            self.kv_cache_manager.get_or_create_kv_page(cached_prefix_hash, "dummy_cached_kv_data") # dummy data
            
            # Simulate generating the remaining part of the prompt
            remaining_tokens = tokenized_prompt[len(longest_prefix_match.split()):]
            response, new_kv_data = self._simulate_inference_step(remaining_tokens, use_kv_cache=True, cached_prefix_hash=cached_prefix_hash)
            
            # Update cache for the full prompt, if it's a new entry
            full_prompt_key = " ".join(tokenized_prompt)
            if full_prompt_key not in self.prefix_kv_cache:
                self.prefix_kv_cache[full_prompt_key] = full_prompt_key # Use full prompt as hash for new data
                self.kv_cache_manager.get_or_create_kv_page(full_prompt_key, new_kv_data)
        else:
            print(f"No KV cache reuse for prompt: '{prompt}'")
            response, kv_data = self._simulate_inference_step(tokenized_prompt)
            
            # Store KV data for potential future reuse (conceptual page allocation)
            full_prompt_key = " ".join(tokenized_prompt)
            self.prefix_kv_cache[full_prompt_key] = full_prompt_key # Use full prompt as hash
            self.kv_cache_manager.get_or_create_kv_page(full_prompt_key, kv_data)

        return response

class Chatbot:
    """
    Intelligent Customer Support Chatbot Platform.
    """
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine
        print("Chatbot initialized with optimized LLM engine.")

    def process_query(self, user_query):
        print(f"\n--- Processing user query: '{user_query}' ---")
        start_time = time.time()
        response = self.llm_engine.infer(user_query)
        end_time = time.time()
        print(f"Chatbot response: {response}")
        print(f"Query processed in {end_time - start_time:.4f} seconds.")
        return response

# Main execution simulation
if __name__ == "__main__":
    kv_manager = KVCacheManager(gpu_cache_size_pages=3) # Small cache for demonstration
    llm_engine = LLMInferenceEngine(kv_cache_manager=kv_manager)
    chatbot = Chatbot(llm_engine)

    # Simulate critical node replication for fault tolerance
    chatbot.llm_engine.kv_cache_manager.replicate_critical_node("system_prompt_node", "pre-loaded system instructions KV data")

    queries = [
        "hello, how can I help you?",
        "hello, what is my order status?",
        "what is my order status for item X?",
        "how can I help you with account issues?",
        "what is the refund policy?",
        "can you help me with account issues related to billing?",
        "hello, can you help me with a new question?"
    ]

    for query in queries:
        chatbot.process_query(query)
        print("-" * 50)

    # Simulate recovery of a critical node after a hypothetical crash
    print("\n--- Simulating recovery of a critical node ---")
    recovered_data = chatbot.llm_engine.kv_cache_manager.recover_critical_node("system_prompt_node")
    if recovered_data:
        print(f"Successfully recovered: {recovered_data}")
    else:
        print("Failed to recover critical node.")

    print("\n--- End of Chatbot Simulation ---")
