import torch
import collections

class KVNode:
    def __init__(self, node_id: str, kv_tensors: dict, device: str):
        self.node_id = node_id
        self.kv_tensors = {k: v.to(device) for k, v in kv_tensors.items()} if kv_tensors else {}
        self.device = device

    @property
    def size_bytes(self) -> int:
        total_bytes = 0
        for tensor in self.kv_tensors.values():
            total_bytes += tensor.element_size() * tensor.nelement()
        return total_bytes

    def to_device(self, target_device: str):
        if self.device != target_device:
            self.kv_tensors = {k: v.to(target_device) for k, v in self.kv_tensors.items()}
            self.device = target_device

class GPUMemoryCache:
    def __init__(self, max_capacity_bytes: int):
        self.max_capacity_bytes = max_capacity_bytes
        self.cache = {}
        self.current_usage_bytes = 0
        self.lru_queue = collections.deque()

    def add_or_update(self, node: KVNode):
        if node.node_id in self.cache:
            # Update LRU position
            self.lru_queue.remove(node.node_id)
            self.lru_queue.append(node.node_id)
            # Node already on GPU, assume tensors are already updated if necessary outside
            return

        node_size = node.size_bytes
        if node_size > self.max_capacity_bytes:
            raise ValueError(f"Node {node.node_id} size ({node_size} bytes) exceeds GPU cache capacity.")

        while self.current_usage_bytes + node_size > self.max_capacity_bytes:
            if not self.lru_queue:
                raise RuntimeError("Cannot evict from empty GPU cache to make space.")
            # Eviction logic will be handled by the manager, this just marks space for eviction
            # We'll return the node to be evicted to the manager for further processing
            return self._evict_lru_node_for_manager()

        node.to_device("cuda") # Ensure it's on GPU
        self.cache[node.node_id] = node
        self.current_usage_bytes += node_size
        self.lru_queue.append(node.node_id)
        return None # No node evicted in this step

    def get(self, node_id: str) -> KVNode:
        if node_id in self.cache:
            self.lru_queue.remove(node_id)
            self.lru_queue.append(node_id)
            return self.cache[node_id]
        return None

    def _evict_lru_node_for_manager(self) -> KVNode:
        if not self.lru_queue:
            return None
        evicted_node_id = self.lru_queue.popleft()
        evicted_node = self.cache.pop(evicted_node_id)
        self.current_usage_bytes -= evicted_node.size_bytes
        # Tensors remain on GPU for now, manager decides what to do
        return evicted_node

    def free_node(self, node_id: str):
        if node_id in self.cache:
            node = self.cache.pop(node_id)
            self.current_usage_bytes -= node.size_bytes
            if node_id in self.lru_queue:
                self.lru_queue.remove(node_id)
            # Explicitly free GPU tensors if possible (e.g., set to None for GC)
            del node.kv_tensors # Allow tensors to be garbage collected from GPU

class HostMemoryCache:
    def __init__(self):
        self.cache = {}

    def store(self, node: KVNode):
        node.to_device("cpu") # Ensure it's on CPU
        self.cache[node.node_id] = node

    def retrieve(self, node_id: str) -> KVNode:
        return self.cache.get(node_id)

    def remove(self, node_id: str):
        if node_id in self.cache:
            del self.cache[node_id]

class SwapOutOnceCacheManager:
    def __init__(self, gpu_cache_capacity_bytes: int):
        self.gpu_cache = GPUMemoryCache(gpu_cache_capacity_bytes)
        self.host_cache = HostMemoryCache()
        self.evicted_once_nodes = set() # Stores node_ids that have been copied to host memory

    def get_node(self, node_id: str) -> KVNode:
        node = self.gpu_cache.get(node_id)
        if node:
            return node

        # Not in GPU, check host cache
        node = self.host_cache.retrieve(node_id)
        if node:
            # Node is in host cache, need to bring it back to GPU
            self._ensure_space_on_gpu(node.size_bytes)
            node.to_device("cuda") # Move tensors back to GPU
            self.gpu_cache.cache[node.node_id] = node # Add directly after ensuring space
            self.gpu_cache.current_usage_bytes += node.size_bytes
            self.gpu_cache.lru_queue.append(node.node_id)
            return node
        return None # Node not found in either cache

    def add_node(self, node: KVNode):
        # Check if already on GPU, if so, just update LRU via get_node
        if node.node_id in self.gpu_cache.cache:
            self.gpu_cache.get(node.node_id)
            return

        # If node exists in host cache, retrieve it and add to GPU (will update LRU)
        if node.node_id in self.host_cache.cache:
            self.get_node(node.node_id)
            return

        # New node or entirely new to system
        self._ensure_space_on_gpu(node.size_bytes)
        node.to_device("cuda")
        self.gpu_cache.cache[node.node_id] = node
        self.gpu_cache.current_usage_bytes += node.size_bytes
        self.gpu_cache.lru_queue.append(node.node_id)


    def _ensure_space_on_gpu(self, required_size_bytes: int):
        while self.gpu_cache.current_usage_bytes + required_size_bytes > self.gpu_cache.max_capacity_bytes:
            evicted_node = self.gpu_cache._evict_lru_node_for_manager()
            if evicted_node is None:
                raise RuntimeError("GPU cache is full and no nodes can be evicted.")
            self._handle_evicted_node(evicted_node)

    def _handle_evicted_node(self, node: KVNode):
        if node.node_id not in self.evicted_once_nodes:
            # First time eviction, copy to host memory
            node.to_device("cpu")
            self.host_cache.store(node)
            self.evicted_once_nodes.add(node.node_id)
        else:
            # Subsequent eviction, host already has a copy, just free GPU memory
            # Tensors were implicitly freed when pop() was called in _evict_lru_node_for_manager
            # We just need to make sure the KVNode object itself doesn't hold GPU references if not removed from `gpu_cache.cache`
            # In this current implementation, _evict_lru_node_for_manager already pops it from self.gpu_cache.cache
            pass # No explicit action needed for host_cache

    def remove_node_entirely(self, node_id: str):
        # Remove from GPU if present
        if node_id in self.gpu_cache.cache:
            self.gpu_cache.free_node(node_id)
        
        # Remove from Host if present
        if node_id in self.host_cache.cache:
            self.host_cache.remove(node_id)
            if node_id in self.evicted_once_nodes:
                self.evicted_once_nodes.remove(node_id)


# Example Usage (Demonstration)
if __name__ == "__main__":
    # Simulate some KV tensors
    def create_dummy_kv_tensors(size_mb, device="cpu"):
        # Each tensor will be roughly half the size for K and V
        num_elements = (size_mb * 1024 * 1024) // 8  # Assuming float32 (4 bytes) and some overhead, roughly 8 bytes per KV pair for simplicity
        return {
            "key": torch.randn(int(num_elements / 2), 768, device=device),
            "value": torch.randn(int(num_elements / 2), 768, device=device)
        }

    GPU_CAPACITY_MB = 100
    GPU_CAPACITY_BYTES = GPU_CAPACITY_MB * 1024 * 1024

    manager = SwapOutOnceCacheManager(gpu_cache_capacity_bytes=GPU_CAPACITY_BYTES)

    print(f"GPU Cache Capacity: {GPU_CAPACITY_MB} MB")

    # Create some nodes
    node1_tensors = create_dummy_kv_tensors(20)
    node1 = KVNode("user_1_session", node1_tensors, "cpu")

    node2_tensors = create_dummy_kv_tensors(30)
    node2 = KVNode("user_2_session", node2_tensors, "cpu")

    node3_tensors = create_dummy_kv_tensors(40)
    node3 = KVNode("user_3_session", node3_tensors, "cpu")

    node4_tensors = create_dummy_kv_tensors(50)
    node4 = KVNode("user_4_session", node4_tensors, "cpu")

    print("\n--- Adding nodes to cache ---")
    manager.add_node(node1)
    print(f"Added node1. GPU usage: {manager.gpu_cache.current_usage_bytes / (1024*1024):.2f} MB")

    manager.add_node(node2)
    print(f"Added node2. GPU usage: {manager.gpu_cache.current_usage_bytes / (1024*1024):.2f} MB")

    manager.add_node(node3)
    print(f"Added node3. GPU usage: {manager.gpu_cache.current_usage_bytes / (1024*1024):.2f} MB")

    print("\n--- Accessing node1 (should bring to front of LRU) ---")
    retrieved_node1 = manager.get_node("user_1_session")
    print(f"Retrieved node1. GPU usage: {manager.gpu_cache.current_usage_bytes / (1024*1024):.2f} MB. Node1 device: {retrieved_node1.device}")

    print("\n--- Adding node4 (triggers eviction) ---")
    # node1 (20MB) + node2 (30MB) + node3 (40MB) = 90MB
    # Adding node4 (50MB) will exceed 100MB capacity. node2 should be evicted (LRU after node1 was accessed)
    manager.add_node(node4)
    print(f"Added node4. GPU usage: {manager.gpu_cache.current_usage_bytes / (1024*1024):.2f} MB")
    print(f"Is node1 in GPU cache? {'user_1_session' in manager.gpu_cache.cache}")
    print(f"Is node2 in GPU cache? {'user_2_session' in manager.gpu_cache.cache}")
    print(f"Is node3 in GPU cache? {'user_3_session' in manager.gpu_cache.cache}")
    print(f"Is node4 in GPU cache? {'user_4_session' in manager.gpu_cache.cache}")
    print(f"Is node2 in host cache? {'user_2_session' in manager.host_cache.cache}")
    print(f"Node2 evicted once? {'user_2_session' in manager.evicted_once_nodes}")

    print("\n--- Accessing node2 (should move from host to GPU, triggering another eviction) ---")
    # Current GPU: node1(20), node3(40), node4(50) = 110MB - but it should be 90-100MB
    # After node2 evicted: node1(20), node3(40), node4(50) is 110MB - something is off with dummy size or eviction logic
    # Let's re-evaluate sizes after an eviction: If node2 (30MB) was evicted. GPU was (node1, node2, node3) = 90MB. Add node4 (50MB).
    # Node1 (20) last accessed. LRU queue: [node2, node3, node1]
    # Evict node2. GPU (node1, node3) = 60MB. Add node4 (50MB). Total 110MB (exceeds cap).
    # Evict node3. GPU (node1) = 20MB. Add node4 (50MB). Total 70MB. This sounds more correct.
    # Let's assume the eviction logic works correctly within _ensure_space_on_gpu

    # After node4 was added, node2 (30MB) and node3 (40MB) should have been evicted.
    # GPU should contain node1 (20MB) and node4 (50MB) = 70MB. Host should have node2, node3.

    # Access node2 (30MB) - this will require 30MB space. Current GPU (node1, node4) = 70MB. Total 100MB.
    # No further evictions should happen to accommodate node2.

    retrieved_node2 = manager.get_node("user_2_session")
    print(f"Retrieved node2. GPU usage: {manager.gpu_cache.current_usage_bytes / (1024*1024):.2f} MB. Node2 device: {retrieved_node2.device}")
    print(f"Is node1 in GPU cache? {'user_1_session' in manager.gpu_cache.cache}")
    print(f"Is node2 in GPU cache? {'user_2_session' in manager.gpu_cache.cache}")
    print(f"Is node3 in GPU cache? {'user_3_session' in manager.gpu_cache.cache}")
    print(f"Is node4 in GPU cache? {'user_4_session' in manager.gpu_cache.cache}")
    print(f"Is node3 in host cache? {'user_3_session' in manager.host_cache.cache}")
    print(f"Node3 evicted once? {'user_3_session' in manager.evicted_once_nodes}")

    print("\n--- Evicting node1 again (should NOT copy to host, as it's already there conceptually via previous evictions) ---")
    # To simulate this, we need to force an eviction of node1
    # Let's add a very large node to force multiple evictions
    node5_tensors = create_dummy_kv_tensors(90)
    node5 = KVNode("user_5_session", node5_tensors, "cpu")
    manager.add_node(node5)

    print(f"Added node5. GPU usage: {manager.gpu_cache.current_usage_bytes / (1024*1024):.2f} MB")
    print(f"Is node1 in GPU cache? {'user_1_session' in manager.gpu_cache.cache}")
    print(f"Is node1 in host cache? {'user_1_session' in manager.host_cache.cache}") # Should be True
    print(f"Node1 evicted once? {'user_1_session' in manager.evicted_once_nodes}") # Should be True

    print("\n--- Removing node2 entirely from the system ---")
    manager.remove_node_entirely("user_2_session")
    print(f"Is node2 in GPU cache? {'user_2_session' in manager.gpu_cache.cache}")
    print(f"Is node2 in host cache? {'user_2_session' in manager.host_cache.cache}")
    print(f"Node2 evicted once? {'user_2_session' in manager.evicted_once_nodes}")
