import torch
import collections
from loguru import logger

class KVCacheNode:
    def __init__(self, node_id, kv_tensors=None, location="gpu", swapped_out_once=False):
        self.node_id = node_id
        self.kv_tensors = kv_tensors  # This would be a tuple of (key_tensor, value_tensor)
        self.location = location  # "gpu" or "host"
        self.swapped_out_once = swapped_out_once
        self.last_accessed = 0 # For LRU simulation

class KVCacheManager:
    def __init__(self, gpu_capacity_mb, host_capacity_mb):
        self.gpu_capacity_bytes = gpu_capacity_mb * 1024 * 1024
        self.host_capacity_bytes = host_capacity_mb * 1024 * 1024

        self.gpu_memory_usage_bytes = 0
        self.host_memory_usage_bytes = 0

        # Simulating memory with dictionaries mapping node_id to KVCacheNode
        self.gpu_cache = collections.OrderedDict() # For LRU
        self.host_cache = {} # No specific order needed for host

        self.node_metadata = {} # Stores KVCacheNode objects (main source of truth)
        self.access_counter = 0

        logger.info(f"KVCacheManager initialized with GPU capacity: {gpu_capacity_mb}MB, Host capacity: {host_capacity_mb}MB")

    def _get_tensor_size_bytes(self, kv_tensors):
        if kv_tensors is None:
            return 0
        total_size = 0
        for tensor in kv_tensors:
            if tensor is not None:
                total_size += tensor.numel() * tensor.element_size()
        return total_size

    def _evict_from_gpu_lru(self, required_space_bytes):
        """Evicts the least recently used node(s) from GPU to make space."""
        evicted_nodes_info = []
        while self.gpu_memory_usage_bytes + required_space_bytes > self.gpu_capacity_bytes and self.gpu_cache:
            lru_node_id, _ = self.gpu_cache.popitem(last=False) # Get LRU
            node = self.node_metadata[lru_node_id]
            node_size = self._get_tensor_size_bytes(node.kv_tensors)

            logger.warning(f"GPU full, evicting LRU node {lru_node_id}. Swapped out once: {node.swapped_out_once}")

            if not node.swapped_out_once:
                # First time eviction: copy to host
                if self.host_memory_usage_bytes + node_size > self.host_capacity_bytes:
                    logger.error(f"Host memory full, cannot evict node {lru_node_id} from GPU. This is a critical state.")
                    # In a real system, you might evict from host too, or throw an error
                    raise MemoryError("Host memory capacity exceeded during GPU eviction.")
                
                # Copy data to host (simulated by updating node.location)
                self.host_cache[lru_node_id] = node.kv_tensors
                self.host_memory_usage_bytes += node_size
                node.location = "host"
                node.swapped_out_once = True
                logger.info(f"Node {lru_node_id} (first time) moved from GPU to Host. Host usage: {self.host_memory_usage_bytes / (1024*1024):.2f}MB")
            else:
                # Subsequent eviction: data is already on host, just free GPU memory
                node.location = "host"
                logger.info(f"Node {lru_node_id} (subsequent) freed from GPU. Data remains on Host.")
            
            self.gpu_memory_usage_bytes -= node_size
            node.kv_tensors = None # GPU data is gone
            evicted_nodes_info.append(lru_node_id)
        
        return evicted_nodes_info

    def _update_access_time(self, node_id):
        self.access_counter += 1
        if node_id in self.node_metadata:
            self.node_metadata[node_id].last_accessed = self.access_counter

    def get_kv_tensors(self, node_id, current_kv_tensors=None):
        """
        Retrieves KV tensors for a given node_id.
        If not on GPU, tries to move from host to GPU.
        If not in cache, optionally allows adding new tensors.
        """
        self._update_access_time(node_id)
        
        if node_id in self.gpu_cache:
            # Already on GPU, just update LRU
            self.gpu_cache.move_to_end(node_id)
            logger.debug(f"Node {node_id} found on GPU.")
            return self.node_metadata[node_id].kv_tensors
        
        # Not on GPU, check host
        if node_id in self.host_cache:
            node = self.node_metadata[node_id]
            node_size = self._get_tensor_size_bytes(self.host_cache[node_id])

            # Try to move from host to GPU
            if self.gpu_memory_usage_bytes + node_size > self.gpu_capacity_bytes:
                logger.warning(f"GPU full, need to evict to bring {node_id} from host.")
                self._evict_from_gpu_lru(node_size) # Make space
            
            if self.gpu_memory_usage_bytes + node_size <= self.gpu_capacity_bytes:
                # Move to GPU
                node.kv_tensors = self.host_cache.pop(node_id) # Remove from host
                self.host_memory_usage_bytes -= node_size
                self.gpu_cache[node_id] = node.kv_tensors # Add to GPU
                self.gpu_cache.move_to_end(node_id)
                self.gpu_memory_usage_bytes += node_size
                node.location = "gpu"
                logger.info(f"Node {node_id} moved from Host to GPU. GPU usage: {self.gpu_memory_usage_bytes / (1024*1024):.2f}MB, Host usage: {self.host_memory_usage_bytes / (1024*1024):.2f}MB")
                return node.kv_tensors
            else:
                logger.error(f"Could not make space on GPU for node {node_id} from host.")
                return None # Failed to retrieve to GPU
        
        # Not in cache at all, potentially a new node
        if current_kv_tensors is not None:
            return self.add_or_update_kv_tensors(node_id, current_kv_tensors)
        
        logger.warning(f"Node {node_id} not found in cache and no new tensors provided.")
        return None

    def add_or_update_kv_tensors(self, node_id, kv_tensors):
        """Adds new KV tensors for a node or updates existing ones on GPU."""
        node_size = self._get_tensor_size_bytes(kv_tensors)

        if node_id in self.node_metadata:
            # Node exists, it might be on host or already on GPU
            node = self.node_metadata[node_id]
            if node.location == "host":
                # If on host, moving to GPU, similar to retrieve_to_gpu
                if self.host_cache.get(node_id) is not None:
                    # Remove old data from host if it was there
                    old_host_data_size = self._get_tensor_size_bytes(self.host_cache.pop(node_id))
                    self.host_memory_usage_bytes -= old_host_data_size

            # Evict from GPU if necessary before placing new/updated data
            if self.gpu_memory_usage_bytes + node_size > self.gpu_capacity_bytes:
                self._evict_from_gpu_lru(node_size)

            if self.gpu_memory_usage_bytes + node_size <= self.gpu_capacity_bytes:
                # Place new tensors on GPU
                # Ensure tensors are on CUDA if GPU is available
                new_kv_tensors_on_gpu = tuple(t.to("cuda") for t in kv_tensors) if torch.cuda.is_available() else kv_tensors
                node.kv_tensors = new_kv_tensors_on_gpu
                self.gpu_cache[node_id] = node.kv_tensors
                self.gpu_cache.move_to_end(node_id)
                self.gpu_memory_usage_bytes += node_size
                node.location = "gpu"
                node.swapped_out_once = False # If new data, reset this flag as it's a 