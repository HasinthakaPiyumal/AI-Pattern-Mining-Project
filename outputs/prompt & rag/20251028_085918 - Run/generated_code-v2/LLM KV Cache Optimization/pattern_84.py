import collections
from datetime import datetime

class KVCacheEntry:
    def __init__(self, node_id, kv_tensors):
        self.node_id = node_id
        self.kv_tensors = kv_tensors
        self.in_gpu_memory = False
        self.in_host_memory = False
        self.last_accessed = datetime.now()

    def update_access_time(self):
        self.last_accessed = datetime.now()

class MemoryManager:
    def __init__(self, capacity):
        self.capacity = capacity
        self.store = collections.OrderedDict()

    def current_size(self):
        return len(self.store)

    def is_full(self):
        return self.current_size() >= self.capacity

    def get_entry(self, node_id):
        if node_id in self.store:
            entry = self.store.pop(node_id)
            entry.update_access_time()
            self.store[node_id] = entry
            return entry
        return None

    def evict_lru_entry(self):
        if not self.store:
            return None
        lru_node_id, lru_entry = self.store.popitem(last=False)
        return lru_entry

class GPUMemoryManager(MemoryManager):
    def __init__(self, capacity):
        super().__init__(capacity)

    def add_entry(self, entry):
        self.store[entry.node_id] = entry
        entry.in_gpu_memory = True
        entry.update_access_time()

    def remove_entry(self, node_id):
        if node_id in self.store:
            entry = self.store.pop(node_id)
            entry.in_gpu_memory = False
            return entry
        return None

class HostMemoryManager(MemoryManager):
    def __init__(self, capacity):
        super().__init__(capacity)

    def add_entry(self, entry):
        self.store[entry.node_id] = entry
        entry.in_host_memory = True
        entry.update_access_time()

    def remove_entry(self, node_id):
        if node_id in self.store:
            entry = self.store.pop(node_id)
            entry.in_host_memory = False
            return entry
        return None

class AdaptiveKVCache:
    def __init__(self, gpu_capacity, host_capacity):
        self.gpu_manager = GPUMemoryManager(gpu_capacity)
        self.host_manager = HostMemoryManager(host_capacity)

    def get(self, node_id):
        entry = self.gpu_manager.get_entry(node_id)
        if entry:
            return entry.kv_tensors

        entry = self.host_manager.get_entry(node_id)
        if entry:
            if self.gpu_manager.is_full():
                self._evict_from_gpu()
            
            self.host_manager.remove_entry(node_id)
            self.gpu_manager.add_entry(entry)
            return entry.kv_tensors
        
        return None

    def put(self, node_id, kv_tensors):
        if node_id in self.gpu_manager.store:
            entry = self.gpu_manager.get_entry(node_id)
            entry.kv_tensors = kv_tensors
            return

        if node_id in self.host_manager.store:
            entry = self.host_manager.get_entry(node_id)
            if self.gpu_manager.is_full():
                self._evict_from_gpu()
            self.host_manager.remove_entry(node_id)
            entry.kv_tensors = kv_tensors
            self.gpu_manager.add_entry(entry)
            return

        if self.gpu_manager.is_full():
            self._evict_from_gpu()

        new_entry = KVCacheEntry(node_id, kv_tensors)
        self.gpu_manager.add_entry(new_entry)

    def _evict_from_gpu(self):
        entry_to_evict = self.gpu_manager.evict_lru_entry()
        if entry_to_evict:
            self._handle_eviction(entry_to_evict)

    def _handle_eviction(self, entry_to_evict):
        self.gpu_manager.remove_entry(entry_to_evict.node_id)

        if not entry_to_evict.in_host_memory:
            if self.host_manager.is_full():
                host_lru_entry = self.host_manager.evict_lru_entry()
                if host_lru_entry:
                    self.host_manager.remove_entry(host_lru_entry.node_id)
            
            self.host_manager.add_entry(entry_to_evict)
