import collections
import time
import random

# --- Configuration --- #
GPU_MEMORY_SIZE = 1024  # Simulated GPU memory in arbitrary units (e.g., MB or pages)
HOST_MEMORY_SIZE = 4096 # Simulated Host memory in arbitrary units
KV_PAGE_SIZE = 64       # Size of a KV page/block
CRITICAL_NODE_REPLICATION_INTERVAL = 5 # Replicate every N updates

class KVCacheManager:
    """
    Manages the KV cache, simulating GPU and host memory, PagedAttention,
    KV Cache Reuse, Critical KV Cache Node Replication, and Swap-Out-Only-Once strategy.
    """
    def __init__(self):
        self.gpu_cache = collections.OrderedDict()  # Stores KV tensors on 'GPU' (key: prefix_hash, value: list of kv_pages)
        self.host_cache = collections.OrderedDict() # Stores KV tensors on 'Host' (key: prefix_hash, value: list of kv_pages)
        self.gpu_memory_usage = 0
        self.host_memory_usage = 0
        self.page_counter = 0 # For unique page IDs
        self.replication_counter = 0
        print("KVCacheManager initialized.")

    def _allocate_memory_page(self, location='gpu'):
        """
        Simulates allocating a memory page. In a real system, this would involve actual memory allocation.
        Returns a unique page ID and updates memory usage.
        """
        self.page_counter += 1
        page_id = f"page_{self.page_counter}"
        if location == 'gpu':
            if self.gpu_memory_usage + KV_PAGE_SIZE > GPU_MEMORY_SIZE:
                # Simulate eviction to host if GPU is full (simplified LRU-like)
                self._evict_to_host()
            self.gpu_memory_usage += KV_PAGE_SIZE
            return page_id, 'gpu'
        elif location == 'host':
            if self.host_memory_usage + KV_PAGE_SIZE > HOST_MEMORY_SIZE:
                raise MemoryError("Host memory full!") # Simplified error for full host memory
            self.host_memory_usage += KV_PAGE_SIZE
            return page_id, 'host'
        return None, None

    def _free_memory_page(self, page_id, location):
        """
        Simulates freeing a memory page.
        """
        if location == 'gpu':
            self.gpu_memory_usage -= KV_PAGE_SIZE
            # Remove the page from its corresponding entry in gpu_cache (simplified)
            for prefix_hash, pages in self.gpu_cache.items():
                self.gpu_cache[prefix_hash] = [p for p in pages if p['id'] != page_id]
                if not self.gpu_cache[prefix_hash]: # Remove if no pages left
                    del self.gpu_cache[prefix_hash]
            print(f"Freed GPU page {page_id}. GPU usage: {self.gpu_memory_usage}/{GPU_MEMORY_SIZE}")
        elif location == 'host':
            self.host_memory_usage -= KV_PAGE_SIZE
            for prefix_hash, pages in self.host_cache.items():
                self.host_cache[prefix_hash] = [p for p in pages if p['id'] != page_id]
                if not self.host_cache[prefix_hash]:
                    del self.host_cache[prefix_hash]
            print(f"Freed Host page {page_id}. Host usage: {self.host_memory_usage}/{HOST_MEMORY_SIZE}")

    def _evict_to_host(self):
        """
        Implements Swap-Out-Only-Once Cache Strategy conceptually.
        Evicts the least recently used KV pages from GPU to host memory.
        Only copies to host memory *once* upon initial eviction.
        """
        if not self.gpu_cache:
            return

        # Find the LRU prefix to evict (simplified: just take the first entry in OrderedDict)
        lru_prefix_hash, lru_pages = next(iter(self.gpu_cache.items()))
        
        print(f"Evicting prefix '{lru_prefix_hash}' from GPU to Host. Pages: {[p['id'] for p in lru_pages]}")

        # Move pages from GPU to host
        for page_data in lru_pages:
            page_id = page_data['id']
            # If this prefix's pages are already in host_cache, we just update their status/location
            # Otherwise, it's the 'Swap-Out-Only-Once' - copy it now.
            
            # Simulate the 