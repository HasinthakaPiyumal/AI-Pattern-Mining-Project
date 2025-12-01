import collections
import numpy as np

class KVCacheNode:
    def __init__(self, node_id, kv_data):
        self.node_id = node_id
        self.kv_data = kv_data

class HierarchicalKVCacheManager:
    def __init__(self, gpu_capacity, host_capacity):
        self.gpu_capacity = gpu_capacity
        self.host_capacity = host_capacity
        self.gpu_cache = collections.OrderedDict()
        self.host_cache = {}
        self._nodes_ever_on_host = set()
        self.transfer_gpu_to_host_count = 0
        self.transfer_host_to_gpu_count = 0

    def _evict_from_gpu(self):
        if not self.gpu_cache:
            return
        lru_node_id, lru_node = self.gpu_cache.popitem(last=False)

        if lru_node_id not in self._nodes_ever_on_host:
            if len(self.host_cache) < self.host_capacity:
                self.host_cache[lru_node_id] = lru_node.kv_data  # Store KV data in host
                self._nodes_ever_on_host.add(lru_node_id)
                self.transfer_gpu_to_host_count += 1
            else:
                # In a real scenario, an eviction policy for host cache would be needed
                # For this simulation, we'll just not copy if host is full and it's new
                pass
        # Else, node_id is already in _nodes_ever_on_host, so do not copy, just free GPU memory

    def add_or_access_kv_pair(self, node_id, new_kv_data=None):
        if node_id in self.gpu_cache:
            # Node is in GPU, move to end (most recently used)
            node = self.gpu_cache.pop(node_id)
            self.gpu_cache[node_id] = node
            return f