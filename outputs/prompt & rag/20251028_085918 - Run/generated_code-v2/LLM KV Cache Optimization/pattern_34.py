
from gpu_cache import GPUCache
from host_cache import HostCache

class FaultTolerantKVStore:
    """Manages a multi-level KV cache with replication for fault tolerance.

    This store uses a fast GPU cache and a slower, more persistent host cache.
    Critical KV nodes are replicated from GPU to host memory to enable faster
    recovery from GPU failures.
    """

    def __init__(self, gpu_cache: GPUCache, host_cache: HostCache):
        self.gpu_cache = gpu_cache
        self.host_cache = host_cache
        self.critical_keys = set()  # Stores keys that should be replicated to host
        print("FaultTolerantKVStore initialized.")

    def store(self, key: str, value: any, is_critical: bool = False):
        """Stores a key-value pair in the GPU cache and optionally replicates to host.

        Args:
            key: The key to store.
            value: The value associated with the key.
            is_critical: If True, the key-value pair is also replicated to the host cache.
        """
        self.gpu_cache.put(key, value)
        # print(f"KVStore: Stored \'{key}\' in GPU cache.")
        if is_critical:
            self.host_cache.put(key, value)
            self.critical_keys.add(key)
            # print(f"KVStore: Replicated \'{key}\' to host cache (critical).")

    def get(self, key: str):
        """Retrieves a value, prioritizing the GPU cache, then falling back to host.

        Args:
            key: The key to retrieve.

        Returns:
            The value associated with the key, or None if not found in either cache.
        """
        value = self.gpu_cache.get(key)
        if value is not None:
            # print(f"KVStore: Retrieved \'{key}\' from GPU cache.")
            return value
        
        value = self.host_cache.get(key)
        if value is not None:
            # print(f"KVStore: Retrieved \'{key}\' from HOST cache (fallback).")
            return value

        # print(f"KVStore: \'{key}\' not found in either cache.")
        return None

    def mark_critical(self, key: str, value: any = None):
        """Marks a key as critical and replicates its current value (if exists) to host.

        If the value is not provided, it attempts to get it from the GPU cache.
        """
        if key not in self.critical_keys:
            self.critical_keys.add(key)
            if value is None:
                value = self.gpu_cache.get(key)
            if value is not None:
                self.host_cache.put(key, value)
                print(f"KVStore: Marked \'{key}\' as critical and replicated to host.")
            else:
                print(f"KVStore: Marked \'{key}\' as critical, but no value found to replicate.")

    def unmark_critical(self, key: str):
        """Removes a key from the critical set and optionally deletes it from host cache."""
        if key in self.critical_keys:
            self.critical_keys.remove(key)
            self.host_cache.delete(key)
            print(f"KVStore: Unmarked \'{key}\' as critical and deleted from host cache.")

    def handle_gpu_failure(self):
        """Simulates a GPU failure and initiates recovery using the host cache."""
        print("\nKVStore: Initiating GPU failure handling...")
        self.gpu_cache.clear()  # Simulate GPU memory loss

        print("KVStore: Recovering critical nodes from host cache to GPU cache...")
        recovered_count = 0
        for key in self.critical_keys:
            value = self.host_cache.get(key)
            if value is not None:
                self.gpu_cache.put(key, value)
                recovered_count += 1
        print(f"KVStore: Recovered {recovered_count} critical nodes to GPU cache.")

    def get_all_gpu_keys(self):
        return self.gpu_cache.get_all_keys()

    def get_all_host_keys(self):
        return self.host_cache.get_all_keys()

    def __repr__(self):
        return f"FaultTolerantKVStore(GPU keys={len(self.gpu_cache)}, Host keys={len(self.host_cache)}, Critical keys={len(self.critical_keys)})"
