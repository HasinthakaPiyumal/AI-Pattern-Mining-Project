class KVTensor:
    """
    A simplified representation of a Key-Value (KV) tensor for an LLM.
    In a real scenario, this would be a torch.Tensor or similar.
    """
    def __init__(self, node_id, data):
        self.node_id = node_id
        self.data = data # Simplified representation of tensor data
        self.size = 1 # Simplified size for capacity tracking (e.g., in number of nodes)

    def __repr__(self):
        return f"KVTensor(id={self.node_id})"

class KVCacheSimulator:
    """
    Simulates a hierarchical KV cache with a GPU (fast) and Host (slow) memory,
    implementing the "Swap-Out-Only-Once Cache Strategy".
    """
    def __init__(self, gpu_capacity_units, host_capacity_units):
        self.gpu_cache = {} # key: node_id, value: KVTensor. Represents GPU memory.
        self.host_cache = {} # key: node_id, value: KVTensor. Represents Host memory.
        self.evicted_to_host_tracker = set() # Stores node_ids already copied to host.

        self.gpu_capacity = gpu_capacity_units
        self.host_capacity = host_capacity_units

        self.current_gpu_usage = 0

    def _evict_from_gpu(self):
        """
        Evicts an item from GPU cache using a simplified FIFO-like policy
        and applies the "Swap-Out-Only-Once" strategy.
        """
        if not self.gpu_cache:
            return None # Nothing to evict

        # For simulation, evict the oldest item (first item inserted).
        # A real system would use a more sophisticated policy (e.g., LRU, LFU).
        key_to_evict = next(iter(self.gpu_cache))
        tensor_to_evict = self.gpu_cache.pop(key_to_evict)
        self.current_gpu_usage -= tensor_to_evict.size

        print(f"\n--- Evicting '{key_to_evict}' from GPU ---")

        if key_to_evict not in self.evicted_to_host_tracker:
            # This is the FIRST time this node's KV tensor is evicted from GPU.
            if self.current_host_usage() + tensor_to_evict.size <= self.host_capacity:
                self.host_cache[key_to_evict] = tensor_to_evict
                self.evicted_to_host_tracker.add(key_to_evict)
                print(f"ACTION: Copied '{key_to_evict}' to host memory (first eviction).")
            else:
                print(f"WARNING: Host memory full. '{key_to_evict}' cannot be copied. Lost from cache hierarchy.")
        else:
            # This node's KV tensor was ALREADY copied to host memory in a previous eviction.
            # No need to copy again; just free the GPU space.
            print(f"ACTION: '{key_to_evict}' already in host. GPU memory freed (no copy needed).")
        
        return key_to_evict

    def current_host_usage(self):
        """
        Calculates the current host memory usage.
        """
        return sum(t.size for t in self.host_cache.values())

    def get_kv_tensor(self, node_id):
        """
        Retrieves a KV tensor. Prioritizes GPU. If in Host, attempts to promote to GPU.
        Handles GPU full scenarios by evicting based on the strategy.
        """
        # 1. Check GPU cache (fast access)
        if node_id in self.gpu_cache:
            print(f"Access: '{node_id}' found in GPU cache.")
            return self.gpu_cache[node_id]

        # 2. Check Host cache (slower access)
        if node_id in self.host_cache:
            print(f"Access: '{node_id}' found in host cache. Attempting to promote to GPU.")
            tensor = self.host_cache.pop(node_id) # Temporarily remove from host for potential promotion

            # Try to promote to GPU
            if self.current_gpu_usage + tensor.size <= self.gpu_capacity:
                self.gpu_cache[node_id] = tensor
                self.current_gpu_usage += tensor.size
                print(f"PROMOTION: '{node_id}' successfully promoted to GPU from host.")
                return tensor
            else:
                # GPU is full, evict an item to make space for promotion
                print(f"PROMOTION FAILED: GPU full. Evicting an item from GPU to make space for '{node_id}'.")
                self._evict_from_gpu() # This applies the swap-out-only-once logic
                
                # After eviction, try to add to GPU again
                if self.current_gpu_usage + tensor.size <= self.gpu_capacity:
                    self.gpu_cache[node_id] = tensor
                    self.current_gpu_usage += tensor.size
                    print(f"PROMOTION: '{node_id}' successfully promoted to GPU after eviction.")
                    return tensor
                else:
                    # Still no space after eviction, put it back in host (or drop, depending on policy)
                    self.host_cache[node_id] = tensor 
                    print(f"PROMOTION FAILED: No GPU space even after eviction. '{node_id}' remains in host.")
                    return tensor # Return from host, as it couldn't be promoted to GPU
        
        print(f"Access: '{node_id}' not found in any cache.")
        return None

    def add_kv_tensor(self, node_id, data):
        """
        Adds a new KV tensor or updates an existing one, prioritizing GPU memory.
        Handles GPU full scenarios by evicting based on the strategy.
        """
        new_tensor = KVTensor(node_id, data)

        # 1. If already in GPU, update it
        if node_id in self.gpu_cache:
            old_tensor = self.gpu_cache[node_id]
            self.current_gpu_usage -= old_tensor.size
            self.gpu_cache[node_id] = new_tensor
            self.current_gpu_usage += new_tensor.size
            print(f"Update: '{node_id}' updated in GPU cache.")
            return

        # 2. If in Host (but not GPU), remove from host as it's moving to GPU
        if node_id in self.host_cache:
            self.host_cache.pop(node_id)
            print(f"Removal: '{node_id}' removed from host cache for re-addition/promotion to GPU.")

        # 3. Try to add to GPU
        if self.current_gpu_usage + new_tensor.size <= self.gpu_capacity:
            self.gpu_cache[node_id] = new_tensor
            self.current_gpu_usage += new_tensor.size
            print(f"ADD: '{node_id}' added to GPU cache.")
        else:
            # GPU is full, evict to make space
            print(f"ADD FAILED: GPU full. Evicting an item from GPU to make space for '{node_id}'.")
            self._evict_from_gpu() # This applies swap-out-only-once
            
            # After eviction, try to add to GPU again
            if self.current_gpu_usage + new_tensor.size <= self.gpu_capacity:
                self.gpu_cache[node_id] = new_tensor
                self.current_gpu_usage += new_tensor.size
                print(f"ADD: '{node_id}' added to GPU cache after eviction.")
            else:
                # Still no space after eviction, add to host if possible, or drop
                print(f"WARNING: Still no GPU space after eviction for '{node_id}'. Attempting to add to host.")
                if self.current_host_usage() + new_tensor.size <= self.host_capacity:
                    self.host_cache[node_id] = new_tensor
                    # Mark as if it was evicted to host, as it's now resident there
                    self.evicted_to_host_tracker.add(node_id) 
                    print(f"ADD: '{node_id}' added directly to host cache.")
                else:
                    print(f"WARNING: Host memory also full. '{node_id}' dropped from cache hierarchy.")

    def display_cache_status(self):
        print("\n--- Cache Status ---")
        print(f"GPU Cache (Used: {self.current_gpu_usage}/{self.gpu_capacity} units): {list(self.gpu_cache.keys())}")
        print(f"Host Cache (Used: {self.current_host_usage()}/{self.host_capacity} units): {list(self.host_cache.keys())}")
        print(f"Evicted to Host Tracker: {self.evicted_to_host_tracker}")
        print("--------------------\n")

# --- LLM Chatbot Simulation using the Cache --- 
class LLMChatbot:
    """
    Simulates an LLM chatbot that utilizes the KVCacheSimulator for managing
    its Key-Value (KV) states across conversation turns.
    """
    def __init__(self, cache_simulator):
        self.cache = cache_simulator
        self.conversation_history_kv = [] # Tracks the node_ids for the current conversation

    def simulate_chat_turn(self, user_input, turn_id):
        print(f"\n===== Chat Turn {turn_id}: User says '{user_input}' ====")
        # Simulate generating a new KV state for this turn (e.g., representing attention keys/values)
        new_kv_node_id = f"KV_Turn_{turn_id}"
        new_kv_data = f"KV_Data_for_Turn_{turn_id}_Input: {user_input}"

        print(f"LLM processing... generating KV for '{new_kv_node_id}'.")

        # Add the new KV state to the cache
        self.cache.add_kv_tensor(new_kv_node_id, new_kv_data)
        self.conversation_history_kv.append(new_kv_node_id)
        self.cache.display_cache_status()

        # Simulate accessing previous KV states for context during response generation
        # The LLM might need KV states from earlier turns to maintain coherence.
        print("LLM accessing previous KV states for context:")
        # Access the last few turns, simulating looking back in conversation history
        for i in range(max(0, len(self.conversation_history_kv) - 3), len(self.conversation_history_kv)):
            node_id_to_access = self.conversation_history_kv[i]
            print(f"  Attempting to retrieve '{node_id_to_access}'...")
            retrieved_tensor = self.cache.get_kv_tensor(node_id_to_access)
            if retrieved_tensor:
                print(f"  Successfully retrieved {retrieved_tensor.node_id}.")
            else:
                print(f"  Could not retrieve {node_id_to_access}.")
            self.cache.display_cache_status()
        
        print(f"Chatbot response: (Simulated response based on {len(self.conversation_history_kv)} KV states in history)." + 
              f" Current GPU usage: {self.cache.current_gpu_usage}/{self.cache.gpu_capacity}, " + 
              f"Host usage: {self.cache.current_host_usage()}/{self.cache.host_capacity}.")

# --- Demonstration --- 
if __name__ == "__main__":
    print("Initializing KV Cache Simulator with Swap-Out-Only-Once Strategy")
    # Smaller GPU capacity to quickly demonstrate evictions
    gpu_cap = 2 
    host_cap = 4 # Larger host capacity
    cache_sim = KVCacheSimulator(gpu_cap, host_cap)
    chatbot = LLMChatbot(cache_sim)

    # Simulate a conversation
    chatbot.simulate_chat_turn("Hello, I have a question about my order.", 1)
    chatbot.simulate_chat_turn("My order number is #12345.", 2)
    chatbot.simulate_chat_turn("It was placed last week.", 3)
    chatbot.simulate_chat_turn("Can you tell me its status?", 4)
    chatbot.simulate_chat_turn("Also, what is your return policy?", 5)
    chatbot.simulate_chat_turn("I might need to return item X.", 6)

    # Further demonstrate a re-access of an older, already evicted item
    print("\n--- Demonstrating repeated access of an older KV state ---")
    print("Attempting to access KV_Turn_1 (should be in host and promoted to GPU if space)")
    cache_sim.get_kv_tensor("KV_Turn_1")
    cache_sim.display_cache_status()

    print("Attempting to access KV_Turn_2 (should be in host and promoted to GPU if space)")
    cache_sim.get_kv_tensor("KV_Turn_2")
    cache_sim.display_cache_status()

    print("End of simulation.")

