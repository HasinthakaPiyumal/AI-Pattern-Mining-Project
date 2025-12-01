import collections

class KVCacheNode:
    def __init__(self, node_id, key_tensors, value_tensors):
        self.node_id = node_id
        self.key_tensors = key_tensors
        self.value_tensors = value_tensors

class GPUManager:
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.gpu_memory = {}

    def allocate(self, node_id, key_tensors, value_tensors):
        if len(self.gpu_memory) < self.max_capacity:
            self.gpu_memory[node_id] = KVCacheNode(node_id, key_tensors, value_tensors)
            return True
        return False

    def free(self, node_id):
        if node_id in self.gpu_memory:
            del self.gpu_memory[node_id]
            return True
        return False

    def retrieve(self, node_id):
        return self.gpu_memory.get(node_id)

    def has(self, node_id):
        return node_id in self.gpu_memory

    def is_full(self):
        return len(self.gpu_memory) >= self.max_capacity

class HostManager:
    def __init__(self):
        self.host_memory = {}

    def store(self, node_id, key_tensors, value_tensors):
        self.host_memory[node_id] = KVCacheNode(node_id, key_tensors, value_tensors)

    def free(self, node_id):
        if node_id in self.host_memory:
            del self.host_memory[node_id]
            return True
        return False

    def retrieve(self, node_id):
        return self.host_memory.get(node_id)

    def has(self, node_id):
        return node_id in self.host_memory

class SwapOutOnlyOnceKVCache:
    def __init__(self, gpu_capacity):
        self.gpu_manager = GPUManager(gpu_capacity)
        self.host_manager = HostManager()
        self.copied_to_host_once = set()
        self.gpu_eviction_queue = collections.deque()

    def put(self, node_id, key_tensors, value_tensors):
        if self.gpu_manager.has(node_id):
            self.gpu_manager.gpu_memory[node_id] = KVCacheNode(node_id, key_tensors, value_tensors)
            # Move to front of queue to mark as most recently used
            if node_id in self.gpu_eviction_queue:
                self.gpu_eviction_queue.remove(node_id)
            self.gpu_eviction_queue.appendleft(node_id)
            return

        while self.gpu_manager.is_full():
            self._evict_from_gpu()

        self.gpu_manager.allocate(node_id, key_tensors, value_tensors)
        self.gpu_eviction_queue.appendleft(node_id)

    def get(self, node_id):
        if self.gpu_manager.has(node_id):
            # Move to front of queue to mark as most recently used
            if node_id in self.gpu_eviction_queue:
                self.gpu_eviction_queue.remove(node_id)
            self.gpu_eviction_queue.appendleft(node_id)
            return self.gpu_manager.retrieve(node_id)

        if self.host_manager.has(node_id):
            while self.gpu_manager.is_full():
                self._evict_from_gpu()

            node = self.host_manager.retrieve(node_id)
            self.gpu_manager.allocate(node_id, node.key_tensors, node.value_tensors)
            self.gpu_eviction_queue.appendleft(node_id)
            return self.gpu_manager.retrieve(node_id)

        return None

    def _evict_from_gpu(self):
        if not self.gpu_eviction_queue:
            return

        node_id_to_evict = self.gpu_eviction_queue.pop()

        if node_id_to_evict in self.copied_to_host_once:
            # Data is already in host, just free from GPU
            self.gpu_manager.free(node_id_to_evict)
        else:
            # First eviction from GPU, copy to host
            node = self.gpu_manager.retrieve(node_id_to_evict)
            if node:
                self.host_manager.store(node.node_id, node.key_tensors, node.value_tensors)
                self.copied_to_host_once.add(node.node_id)
            self.gpu_manager.free(node_id_to_evict)

    def remove_from_cache(self, node_id):
        if self.gpu_manager.has(node_id):
            self.gpu_manager.free(node_id)
            if node_id in self.gpu_eviction_queue:
                self.gpu_eviction_queue.remove(node_id)

        if self.host_manager.has(node_id):
            self.host_manager.free(node_id)
            if node_id in self.copied_to_host_once:
                self.copied_to_host_once.remove(node_id)

