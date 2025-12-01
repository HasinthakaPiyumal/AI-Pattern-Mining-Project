import time
import collections

class KVCacheManager:
    def __init__(self, gpu_capacity_bytes, host_capacity_bytes):
        self.gpu_capacity = gpu_capacity_bytes
        self.host_capacity = host_capacity_bytes

        self.gpu_cache = collections.OrderedDict()  # Stores node_id -> KV_tensors, ordered by access
        self.host_cache = {}  # Stores node_id -> KV_tensors
        self.metadata = {}

        self.current_gpu_usage = 0
        self.current_host_usage = 0

    def _evict_lru_from_gpu(self):
        if not self.gpu_cache:
            return None
        
        lru_node_id, _ = self.gpu_cache.popitem(last=False) # Get and remove LRU
        node_size = self.metadata[lru_node_id]['size']
        
        # Apply Swap-Out-Only-Once strategy
        if not self.metadata[lru_node_id]['has_host_copy']:
            # First eviction from GPU, copy to host
            if self.current_host_usage + node_size > self.host_capacity:
                # Host full, cannot evict from GPU. This is a critical scenario.
                # For simplicity, we'll just re-add to GPU cache (or raise an error)
                # A real system would have more sophisticated host eviction.
                print(f"CRITICAL: Host memory full, cannot evict {lru_node_id} from GPU.")
                self.gpu_cache[lru_node_id] = "dummy_kv_tensors_for_" + lru_node_id # Re-add
                self.gpu_cache.move_to_end(lru_node_id, last=False) # Make it LRU again
                return None
            
            self.host_cache[lru_node_id] = "dummy_kv_tensors_for_" + lru_node_id # Simulate copy
            self.current_host_usage += node_size
            self.metadata[lru_node_id]['has_host_copy'] = True
            # print(f"Evicted {lru_node_id} from GPU (first time), copied to host. GPU usage: {self.current_gpu_usage}, Host usage: {self.current_host_usage}")
        else:
            # Subsequent eviction from GPU, host already has a copy
            pass # Do nothing, host copy remains
            # print(f"Evicted {lru_node_id} from GPU (subsequent). GPU usage: {self.current_gpu_usage}, Host usage: {self.current_host_usage}")
            
        self.current_gpu_usage -= node_size
        return lru_node_id

    def load_kv_node(self, node_id, size):
        self.metadata.setdefault(node_id, {'size': size, 'has_host_copy': False, 'last_access_time': 0})
        self.metadata[node_id]['last_access_time'] = time.time()

        if node_id in self.gpu_cache:
            self.gpu_cache.move_to_end(node_id) # Mark as recently used
            return f"Node {node_id} already in GPU, refreshed access."

        # Ensure GPU has capacity
        while self.current_gpu_usage + size > self.gpu_capacity:
            evicted_node = self._evict_lru_from_gpu()
            if evicted_node is None: # Cannot make space
                return f"ERROR: Cannot load {node_id}. GPU and Host memory full or host cannot accept evictions."
            
        # Load into GPU
        if node_id in self.host_cache:
            # Node exists in host, move to GPU
            kv_tensors = self.host_cache.pop(node_id)
            self.current_host_usage -= size
            self.gpu_cache[node_id] = kv_tensors
            self.current_gpu_usage += size
            self.gpu_cache.move_to_end(node_id) # Mark as recently used
            # print(f"Loaded {node_id} from Host to GPU. GPU usage: {self.current_gpu_usage}, Host usage: {self.current_host_usage}")
            return f"Node {node_id} moved from Host to GPU."
        else:
            # New node, add directly to GPU
            self.gpu_cache[node_id] = "dummy_kv_tensors_for_" + node_id
            self.current_gpu_usage += size
            self.gpu_cache.move_to_end(node_id) # Mark as recently used
            self.metadata[node_id]['has_host_copy'] = False # New node has no host copy yet
            # print(f"Loaded new {node_id} to GPU. GPU usage: {self.current_gpu_usage}, Host usage: {self.current_host_usage}")
            return f"New node {node_id} loaded to GPU."

    def evict_from_gpu(self, node_id):
        if node_id not in self.gpu_cache:
            return f"Node {node_id} not in GPU cache."

        node_size = self.metadata[node_id]['size']

        # Remove from GPU cache
        self.gpu_cache.pop(node_id)
        self.current_gpu_usage -= node_size

        # Apply Swap-Out-Only-Once strategy
        if not self.metadata[node_id]['has_host_copy']:
            # First eviction from GPU, copy to host
            if self.current_host_usage + node_size > self.host_capacity:
                print(f"WARNING: Host memory full, cannot copy {node_id} from GPU to Host.")
                # In a real system, you might discard it or have a global eviction policy
                # For this simulation, we'll just not copy to host, losing the data if not present elsewhere
                return f"Node {node_id} evicted from GPU, but host is full. Data lost."
                
            self.host_cache[node_id] = "dummy_kv_tensors_for_" + node_id # Simulate copy
            self.current_host_usage += node_size
            self.metadata[node_id]['has_host_copy'] = True
            return f"Node {node_id} evicted from GPU (first time), copied to host."
        else:
            # Subsequent eviction from GPU, host already has a copy
            # Simply free GPU memory
            return f"Node {node_id} evicted from GPU (subsequent), host copy retained."

    def evict_from_host(self, node_id):
        if node_id not in self.host_cache:
            return f"Node {node_id} not in host cache."

        node_size = self.metadata[node_id]['size']
        self.host_cache.pop(node_id)
        self.current_host_usage -= node_size
        
        # Remove metadata entirely as it's gone from both caches
        if node_id in self.metadata:
            del self.metadata[node_id]

        return f"Node {node_id} evicted from host and metadata cleared."

    def get_cache_status(self):
        return {
            "gpu_usage": self.current_gpu_usage,
            "gpu_capacity": self.gpu_capacity,
            "host_usage": self.current_host_usage,
            "host_capacity": self.host_capacity,
            "gpu_items": list(self.gpu_cache.keys()),
            "host_items": list(self.host_cache.keys()),
            "metadata_keys": list(self.metadata.keys())
        }


class ChatbotSimulator:
    def __init__(self, kv_cache_manager):
        self.kv_cache_manager = kv_cache_manager
        self.conversations = {}
        self.next_node_id = 0

    def _generate_kv_tensors(self, conversation_id, query, response):
        # Simulate KV tensor generation and size estimation
        # In a real LLM, this would be the actual output of the forward pass for KV cache
        size = len(query) + len(response) + 10 # Arbitrary size based on text length
        node_id = f"conv_{conversation_id}_node_{self.next_node_id}"
        self.next_node_id += 1
        return node_id, max(size, 20) # Ensure a minimum size

    def process_query(self, user_id, conversation_id, query):
        print(f"\n--- User {user_id}, Conv {conversation_id}: {query} ---")

        # Simulate LLM processing and KV cache interaction
        # For simplicity, let's assume each query potentially generates a new KV node
        # or interacts with an existing one if the conversation_id is repeated.
        
        # If this is a new conversation or a switch, we might simulate loading previous context
        # For this simulation, we'll simplify and say each query *attempts* to load its conversation's KV node.

        response = f"Bot response to '{query}' in conversation {conversation_id}."
        node_id, kv_size = self._generate_kv_tensors(conversation_id, query, response)
        
        # Try to load/access the KV node for this conversation context
        load_status = self.kv_cache_manager.load_kv_node(node_id, kv_size)
        print(f"KV Cache Load Status for {node_id}: {load_status}")
        
        print("Current Cache Status:", self.kv_cache_manager.get_cache_status())
        return response

    def end_conversation(self, conversation_id):
        print(f"\n--- Ending Conversation {conversation_id} ---")
        # In a real scenario, we'd identify all KV nodes belonging to this conversation_id
        # and evict them completely. For simplicity, we'll just demonstrate evicting a specific node.
        
        # This part of the simulation is tricky without a clear mapping of conv_id to specific kv_node_ids
        # Let's assume we evict the *last used* node for a conversation when it ends.
        # A more robust system would track all active KV nodes per conversation.
        
        nodes_to_evict = [nid for nid in self.kv_cache_manager.metadata if nid.startswith(f"conv_{conversation_id}")]
        
        for node_id in nodes_to_evict:
            if node_id in self.kv_cache_manager.gpu_cache:
                evict_gpu_status = self.kv_cache_manager.evict_from_gpu(node_id)
                print(f"Evicting {node_id} from GPU: {evict_gpu_status}")
            if node_id in self.kv_cache_manager.host_cache:
                evict_host_status = self.kv_cache_manager.evict_from_host(node_id)
                print(f"Evicting {node_id} from Host: {evict_host_status}")
        
        print("Current Cache Status after conversation end:", self.kv_cache_manager.get_cache_status())


# --- Example Usage --- 
if __name__ == "__main__":
    # Simulate a GPU with 200 units capacity and Host with 500 units capacity
    kv_manager = KVCacheManager(gpu_capacity_bytes=200, host_capacity_bytes=500)
    chatbot_sim = ChatbotSimulator(kv_manager)

    print("Initial Cache Status:", kv_manager.get_cache_status())

    # Scenario 1: New conversations, GPU fills up, first evictions to host
    chatbot_sim.process_query(user_id=1, conversation_id='A', query="What are your return policies?")
    chatbot_sim.process_query(user_id=2, conversation_id='B', query="How do I reset my password?")
    chatbot_sim.process_query(user_id=1, conversation_id='A', query="And what about exchanges?") # Re-access 'A'
    chatbot_sim.process_query(user_id=3, conversation_id='C', query="Tell me about your new product features.")
    chatbot_sim.process_query(user_id=4, conversation_id='D', query="I have a billing inquiry.")
    chatbot_sim.process_query(user_id=5, conversation_id='E', query="What's the status of my order?") # This should trigger GPU eviction

    # Scenario 2: Re-accessing an already evicted node from GPU (but still in host)
    print("\n--- Simulating re-access of an evicted node (A) ---")
    chatbot_sim.process_query(user_id=1, conversation_id='A', query="Can I get a refund?") # 'A' should be in host, re-promoted to GPU

    # Scenario 3: Another eviction of node 'A' from GPU (should not copy to host again)
    print("\n--- Forcing another eviction to see swap-out-only-once ---")
    chatbot_sim.process_query(user_id=6, conversation_id='F', query="Technical support question.")
    chatbot_sim.process_query(user_id=7, conversation_id='G', query="Product availability.")
    chatbot_sim.process_query(user_id=8, conversation_id='H', query="Compatibility with device.")
    chatbot_sim.process_query(user_id=9, conversation_id='I', query="Subscription details.")

    # Scenario 4: Evicting a conversation completely
    chatbot_sim.end_conversation('B')
    chatbot_sim.end_conversation('A')

    # Check final status
    print("\nFinal Cache Status:", kv_manager.get_cache_status())

    # Attempt to load a node for a conversation that ended and was fully evicted
    print("\n--- Attempting to load a fully evicted node (A) ---")
    chatbot_sim.process_query(user_id=1, conversation_id='A', query="Trying to bring back conversation A.")
    print("Final Cache Status after attempted reload:", kv_manager.get_cache_status())
