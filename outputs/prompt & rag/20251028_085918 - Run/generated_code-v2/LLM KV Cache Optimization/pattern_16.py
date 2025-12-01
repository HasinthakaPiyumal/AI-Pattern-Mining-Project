import collections
import random

class KVCacheNode:
    def __init__(self, node_id, kv_tensors_data, size):
        self.node_id = node_id
        self.kv_tensors_data = kv_tensors_data  # Simulated KV tensors
        self.size = size  # Simulated size of KV tensors in memory
        self.last_accessed = 0  # For potential LRU-like eviction

    def __repr__(self):
        return f"KVCacheNode(id={self.node_id}, size={self.size}, accessed={self.last_accessed})"

class KVCacheManager:
    def __init__(self, gpu_capacity_mb, host_capacity_mb):
        self.gpu_capacity = gpu_capacity_mb
        self.host_capacity = host_capacity_mb

        self.gpu_cache = collections.OrderedDict() # Simulates GPU memory, ordered for LRU-like eviction
        self.host_cache = {}
        self.host_residents = set() # Node IDs that have a copy in host memory

        self.current_gpu_usage = 0
        self.current_host_usage = 0

        self.clock = 0 # Simple clock for last_accessed

    def _get_eviction_candidate_gpu(self):
        # For simplicity, implementing a basic FIFO-like eviction for GPU
        # In a real system, this would be LRU, LFU, or a more advanced policy
        if not self.gpu_cache:
            return None
        # Get the least recently used (first inserted) item
        return next(iter(self.gpu_cache))

    def _evict_from_gpu(self, node_id_to_evict):
        if node_id_to_evict not in self.gpu_cache:
            return False

        node = self.gpu_cache.pop(node_id_to_evict)
        self.current_gpu_usage -= node.size

        if node.node_id not in self.host_residents:
            # First time eviction, copy to host
            if self.current_host_usage + node.size > self.host_capacity:
                print(f"[CACHE] Host cache full, cannot evict {node.node_id} from GPU. This is a critical state.")
                # In a real scenario, this would involve evicting from host as well
                return False # Indicate failure to evict to host
            self.host_cache[node.node_id] = node
            self.host_residents.add(node.node_id)
            self.current_host_usage += node.size
            print(f"[CACHE] Evicted node {node.node_id} from GPU to Host (first time). GPU usage: {self.current_gpu_usage}MB, Host usage: {self.current_host_usage}MB")
        else:
            # Subsequent eviction, host already has a copy, just free GPU memory
            print(f"[CACHE] Evicted node {node.node_id} from GPU (host already has copy). GPU usage: {self.current_gpu_usage}MB")
        return True

    def add_node(self, node_id, kv_tensors_data, size):
        self.clock += 1
        if node_id in self.gpu_cache:
            # Node already in GPU, update access time and move to end (LRU)
            node = self.gpu_cache.pop(node_id)
            node.last_accessed = self.clock
            self.gpu_cache[node_id] = node
            print(f"[CACHE] Node {node_id} already in GPU, updated access.")
            return True

        # If node is in host, retrieve it
        if node_id in self.host_cache:
            node = self.host_cache.pop(node_id)
            self.current_host_usage -= node.size
            print(f"[CACHE] Retrieved node {node_id} from Host.")
        else:
            # New node entirely
            node = KVCacheNode(node_id, kv_tensors_data, size)

        # Try to add to GPU cache
        while self.current_gpu_usage + node.size > self.gpu_capacity:
            eviction_candidate_id = self._get_eviction_candidate_gpu()
            if eviction_candidate_id is None:
                print(f"[CACHE] GPU cache full, and no candidate for eviction. Cannot add node {node_id}.")
                # If we can't evict from GPU, and the new node couldn't be placed
                # we might need to place it directly in host or fail
                if node_id not in self.host_residents:
                    if self.current_host_usage + node.size > self.host_capacity:
                        print(f"[CACHE] Host cache also full, dropping node {node_id}.")
                        return False
                    else:
                        self.host_cache[node.node_id] = node
                        self.host_residents.add(node.node_id)
                        self.current_host_usage += node.size
                        print(f"[CACHE] Added node {node.node_id} directly to Host as GPU is full.")
                return False
            if not self._evict_from_gpu(eviction_candidate_id):
                print(f"[CACHE] Failed to evict {eviction_candidate_id} from GPU. Cannot add node {node_id}.")
                return False

        node.last_accessed = self.clock
        self.gpu_cache[node_id] = node
        self.current_gpu_usage += node.size
        print(f"[CACHE] Added node {node_id} to GPU. GPU usage: {self.current_gpu_usage}MB, Host usage: {self.current_host_usage}MB")
        return True

    def access_node(self, node_id):
        self.clock += 1
        if node_id in self.gpu_cache:
            node = self.gpu_cache.pop(node_id)
            node.last_accessed = self.clock
            self.gpu_cache[node_id] = node # Move to end (most recently used)
            print(f"[CACHE] Accessed node {node_id} in GPU.")
            return node.kv_tensors_data
        elif node_id in self.host_cache:
            print(f"[CACHE] Node {node_id} found in Host, moving to GPU.")
            node = self.host_cache.pop(node_id)
            self.current_host_usage -= node.size

            # Attempt to move from host to GPU
            while self.current_gpu_usage + node.size > self.gpu_capacity:
                eviction_candidate_id = self._get_eviction_candidate_gpu()
                if eviction_candidate_id is None:
                    print(f"[CACHE] GPU cache full, no candidate for eviction to bring {node_id} from host. Placing {node_id} back in host.")
                    self.host_cache[node.node_id] = node
                    self.current_host_usage += node.size
                    return None # Could not move to GPU
                if not self._evict_from_gpu(eviction_candidate_id):
                    print(f"[CACHE] Failed to evict {eviction_candidate_id} from GPU to bring {node_id} from host. Placing {node_id} back in host.")
                    self.host_cache[node.node_id] = node
                    self.current_host_usage += node.size
                    return None
            
            node.last_accessed = self.clock
            self.gpu_cache[node_id] = node
            self.current_gpu_usage += node.size
            print(f"[CACHE] Moved node {node_id} from Host to GPU. GPU usage: {self.current_gpu_usage}MB, Host usage: {self.current_host_usage}MB")
            return node.kv_tensors_data
        else:
            print(f"[CACHE] Node {node_id} not found in any cache.")
            return None

    def remove_node_entirely(self, node_id):
        removed = False
        if node_id in self.gpu_cache:
            node = self.gpu_cache.pop(node_id)
            self.current_gpu_usage -= node.size
            print(f"[CACHE] Removed node {node_id} from GPU entirely.")
            removed = True
        if node_id in self.host_cache:
            node = self.host_cache.pop(node_id)
            self.current_host_usage -= node.size
            print(f"[CACHE] Removed node {node_id} from Host entirely.")
            removed = True
        if node_id in self.host_residents:
            self.host_residents.remove(node_id)
            print(f"[CACHE] Removed node {node_id} from host_residents.")
            removed = True
        
        if not removed:
            print(f"[CACHE] Node {node_id} not found in any cache to remove.")

class LLMInferenceService:
    def __init__(self, kv_cache_manager):
        self.kv_cache_manager = kv_cache_manager
        self.next_node_id = 1

    def generate_response(self, conversation_id, query_text):
        # Simulate LLM processing and generating new KV tensors
        node_id = f"conv-{conversation_id}-node-{self.next_node_id}"
        self.next_node_id += 1
        # Simulate KV tensors data and their size
        kv_data = {f"key_{i}": f"val_{i}" for i in range(random.randint(1, 3))}
        size = random.randint(10, 50) # Simulate 10-50MB for KV tensors
        
        print(f"\n[LLM_SERVICE] Generating KV tensors for {node_id} (size: {size}MB) for conversation {conversation_id}.")
        success = self.kv_cache_manager.add_node(node_id, kv_data, size)
        if success:
            print(f"[LLM_SERVICE] KV tensors for {node_id} added to cache.")
        else:
            print(f"[LLM_SERVICE] Failed to add KV tensors for {node_id} to cache.")
        return node_id if success else None
    
    def retrieve_context(self, node_id):
        print(f"\n[LLM_SERVICE] Attempting to retrieve context for node {node_id}.")
        return self.kv_cache_manager.access_node(node_id)

class ConversationContextManager:
    def __init__(self, kv_cache_manager):
        self.kv_cache_manager = kv_cache_manager
        self.active_conversations = collections.defaultdict(list) # conversation_id -> list of node_ids
        self.next_conversation_id = 1

    def start_conversation(self):
        conv_id = f"conv-{self.next_conversation_id}"
        self.next_conversation_id += 1
        print(f"\n[CONVERSATION_MANAGER] Started new conversation: {conv_id}")
        return conv_id

    def add_node_to_conversation(self, conversation_id, node_id):
        if node_id and node_id not in self.active_conversations[conversation_id]:
            self.active_conversations[conversation_id].append(node_id)
            print(f"[CONVERSATION_MANAGER] Added node {node_id} to conversation {conversation_id}.")

    def end_conversation(self, conversation_id):
        print(f"\n[CONVERSATION_MANAGER] Ending conversation: {conversation_id}")
        for node_id in self.active_conversations.pop(conversation_id, []):
            self.kv_cache_manager.remove_node_entirely(node_id)
        print(f"[CONVERSATION_MANAGER] All nodes for conversation {conversation_id} removed from cache.")

# --- Simulation/Example Usage ---
if __name__ == "__main__":
    # Initialize the cache manager with capacities
    gpu_cap = 100 # MB
    host_cap = 200 # MB
    cache_manager = KVCacheManager(gpu_cap, host_cap)

    llm_service = LLMInferenceService(cache_manager)
    conv_manager = ConversationContextManager(cache_manager)

    print("--- Scenario 1: Basic Node Addition and Eviction (First Time) ---")
    conv1_id = conv_manager.start_conversation()
    nodes_conv1 = []
    for i in range(3):
        node_id = llm_service.generate_response(conv1_id, f"query {i}")
        conv_manager.add_node_to_conversation(conv1_id, node_id)
        nodes_conv1.append(node_id)

    print("\n--- Filling GPU cache and forcing first eviction to Host ---")
    # Generate more nodes to force eviction from GPU to Host
    conv2_id = conv_manager.start_conversation()
    nodes_conv2 = []
    for i in range(4):
        node_id = llm_service.generate_response(conv2_id, f"query {i}")
        conv_manager.add_node_to_conversation(conv2_id, node_id)
        nodes_conv2.append(node_id)

    print("\n--- Scenario 2: Accessing a Node from Host (moves back to GPU) ---")
    # Node 0 from conv1 should be in host now. Access it.
    accessed_data = llm_service.retrieve_context(nodes_conv1[0])
    if accessed_data:
        print(f"Successfully retrieved data for {nodes_conv1[0]} from cache.")

    print("\n--- Scenario 3: Subsequent Eviction (Swap-Out-Only-Once in action) ---")
    # Generate more nodes to force nodes_conv1[0] (which is now in GPU) to be evicted again.
    conv3_id = conv_manager.start_conversation()
    for i in range(5):
        node_id = llm_service.generate_response(conv3_id, f"query {i}")
        conv_manager.add_node_to_conversation(conv3_id, node_id)
    
    print("\n--- Current Cache State ---")
    print(f"GPU Cache: {[n for n in cache_manager.gpu_cache.keys()]}, Usage: {cache_manager.current_gpu_usage}MB")
    print(f"Host Cache: {[n for n in cache_manager.host_cache.keys()]}, Usage: {cache_manager.current_host_usage}MB")
    print(f"Host Residents: {cache_manager.host_residents}")

    print("\n--- Ending Conversations (full eviction) ---")
    conv_manager.end_conversation(conv1_id)
    conv_manager.end_conversation(conv2_id)
    conv_manager.end_conversation(conv3_id)

    print("\n--- Final Cache State ---")
    print(f"GPU Cache: {[n for n in cache_manager.gpu_cache.keys()]}, Usage: {cache_manager.current_gpu_usage}MB")
    print(f"Host Cache: {[n for n in cache_manager.host_cache.keys()]}, Usage: {cache_manager.current_host_usage}MB")
    print(f"Host Residents: {cache_manager.host_residents}")
