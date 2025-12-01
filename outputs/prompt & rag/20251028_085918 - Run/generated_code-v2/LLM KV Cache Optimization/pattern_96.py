import collections
import uuid

class KVCacheNode:
    def __init__(self, node_id, kv_tensors):
        self.node_id = node_id
        self.kv_tensors = kv_tensors
        self.host_copy_exists = False

class GPUMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def add_or_access(self, node):
        if node.node_id in self.cache:
            self.cache.move_to_end(node.node_id)
            return None
        else:
            if len(self.cache) >= self.capacity:
                # Evict LRU node
                lru_node_id, lru_node = self.cache.popitem(last=False)
                self.cache[node.node_id] = node
                return lru_node
            else:
                self.cache[node.node_id] = node
                return None

    def remove(self, node_id):
        if node_id in self.cache:
            del self.cache[node_id]

    def get(self, node_id):
        if node_id in self.cache:
            self.cache.move_to_end(node_id)
            return self.cache[node_id]
        return None

class HostMemory:
    def __init__(self):
        self.cache = {}

    def add(self, node):
        self.cache[node.node_id] = node

    def remove(self, node_id):
        if node_id in self.cache:
            del self.cache[node_id]

    def get(self, node_id):
        return self.cache.get(node_id)

class KVCacheManager:
    def __init__(self, gpu_capacity):
        self.gpu_cache = GPUMemory(gpu_capacity)
        self.host_cache = HostMemory()
        self.metrics = {
            "gpu_to_host_copies": 0,
            "gpu_frees_without_copy": 0
        }

    def access_conversation_node(self, node_id, initial_kv_data=None):
        # 1. Check GPU cache
        node = self.gpu_cache.get(node_id)
        if node:
            return node

        # 2. Check Host cache
        node = self.host_cache.get(node_id)
        if node:
            # Move from Host to GPU
            self.host_cache.remove(node_id)
            evicted_gpu_node = self.gpu_cache.add_or_access(node)
            if evicted_gpu_node:
                self._handle_gpu_eviction(evicted_gpu_node)
            return node

        # 3. New node
        if initial_kv_data is None:
            raise ValueError("initial_kv_data must be provided for a new node.")
        new_node = KVCacheNode(node_id, initial_kv_data)
        evicted_gpu_node = self.gpu_cache.add_or_access(new_node)
        if evicted_gpu_node:
            self._handle_gpu_eviction(evicted_gpu_node)
        return new_node

    def _handle_gpu_eviction(self, evicted_node):
        if not evicted_node.host_copy_exists:
            # First time eviction from GPU, copy to host
            self.host_cache.add(evicted_node)
            evicted_node.host_copy_exists = True
            self.metrics["gpu_to_host_copies"] += 1
        else:
            # Subsequent eviction, host already has a copy, just free GPU memory
            self.metrics["gpu_frees_without_copy"] += 1

    def evict_node_from_system(self, node_id):
        self.gpu_cache.remove(node_id)
        self.host_cache.remove(node_id)

    def get_metrics(self):
        return self.metrics

if __name__ == "__main__":
    print("Simulating Intelligent Customer Support Chatbot with Adaptive KV Cache Management\n")

    # Initialize Cache Manager with GPU capacity of 3
    cache_manager = KVCacheManager(gpu_capacity=3)

    # --- Scenario 1: New Conversations Filling GPU --- 
    print("--- Scenario 1: New Conversations Filling GPU ---")
    conv1_id = str(uuid.uuid4())[:8]
    conv2_id = str(uuid.uuid4())[:8]
    conv3_id = str(uuid.uuid4())[:8]
    conv4_id = str(uuid.uuid4())[:8]

    print(f"Accessing new conversation {conv1_id}")
    cache_manager.access_conversation_node(conv1_id, {"q": "data1", "k": "data1"})
    print(f"GPU Cache: {list(cache_manager.gpu_cache.cache.keys())}")
    print(f"Host Cache: {list(cache_manager.host_cache.cache.keys())}\n")

    print(f"Accessing new conversation {conv2_id}")
    cache_manager.access_conversation_node(conv2_id, {"q": "data2", "k": "data2"})
    print(f"GPU Cache: {list(cache_manager.gpu_cache.cache.keys())}")
    print(f"Host Cache: {list(cache_manager.host_cache.cache.keys())}\n")

    print(f"Accessing new conversation {conv3_id}")
    cache_manager.access_conversation_node(conv3_id, {"q": "data3", "k": "data3"})
    print(f"GPU Cache: {list(cache_manager.gpu_cache.cache.keys())}")
    print(f"Host Cache: {list(cache_manager.host_cache.cache.keys())}\n")

    print(f"GPU capacity reached. Accessing new conversation {conv4_id}. {conv1_id} (LRU) should be evicted from GPU to Host.")
    cache_manager.access_conversation_node(conv4_id, {"q": "data4", "k": "data4"})
    print(f"GPU Cache: {list(cache_manager.gpu_cache.cache.keys())}")
    print(f"Host Cache: {list(cache_manager.host_cache.cache.keys())}")
    print(f"Metrics: {cache_manager.get_metrics()}\n")

    # --- Scenario 2: Re-accessing a node from Host (promoted to GPU) --- 
    print("--- Scenario 2: Re-accessing a node from Host (promoted to GPU) ---")
    print(f"Re-accessing conversation {conv1_id} (currently in Host memory). This should move it back to GPU, evicting {conv2_id} from GPU.")
    cache_manager.access_conversation_node(conv1_id)
    print(f"GPU Cache: {list(cache_manager.gpu_cache.cache.keys())}")
    print(f"Host Cache: {list(cache_manager.host_cache.cache.keys())}")
    print(f"Metrics: {cache_manager.get_metrics()}\n") # conv2_id should be freed from GPU, not copied again

    # --- Scenario 3: Evicting a node already in Host from GPU (no copy) --- 
    print("--- Scenario 3: Evicting a node already in Host from GPU (no copy) ---")
    print(f"Accessing new conversation {str(uuid.uuid4())[:8]} to force eviction of {conv3_id} from GPU (it's already in host now). No new copy to host should occur.")
    cache_manager.access_conversation_node(str(uuid.uuid4())[:8], {"q": "data5", "k": "data5"})
    print(f"GPU Cache: {list(cache_manager.gpu_cache.cache.keys())}")
    print(f"Host Cache: {list(cache_manager.host_cache.cache.keys())}")
    print(f"Metrics: {cache_manager.get_metrics()}\n")

    # --- Scenario 4: Full System Eviction --- 
    print("--- Scenario 4: Full System Eviction ---")
    print(f"Evicting conversation {conv1_id} entirely from the system.")
    cache_manager.evict_node_from_system(conv1_id)
    print(f"GPU Cache: {list(cache_manager.gpu_cache.cache.keys())}")
    print(f"Host Cache: {list(cache_manager.host_cache.cache.keys())}")
    print(f"Metrics: {cache_manager.get_metrics()}\n")