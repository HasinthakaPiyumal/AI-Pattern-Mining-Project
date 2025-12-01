import collections

class GPUKVCache:
    """Simulates a fast, volatile GPU memory KV cache."""
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache = {}
        self.lru = collections.OrderedDict() # For simulating eviction if capacity is reached

    def set(self, key: str, value: any):
        if key in self.cache:
            self.lru.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Evict the least recently used item
                lru_key = next(iter(self.lru))
                del self.cache[lru_key]
                del self.lru[lru_key]
            self.lru[key] = None
        self.cache[key] = value
        # print(f"[GPU Cache] Set: {key}")

    def get(self, key: str) -> any:
        if key in self.cache:
            self.lru.move_to_end(key)
            # print(f"[GPU Cache] Get: {key}")
            return self.cache[key]
        # print(f"[GPU Cache] Miss: {key}")
        return None

    def remove(self, key: str):
        if key in self.cache:
            del self.cache[key]
            del self.lru[key]
            # print(f"[GPU Cache] Removed: {key}")

    def clear(self):
        self.cache.clear()
        self.lru.clear()
        # print("[GPU Cache] Cleared (simulating failure)")

    def items(self):
        return self.cache.items()

    def __len__(self):
        return len(self.cache)


class HostKVCache:
    """Simulates a slower, more persistent host memory KV cache."""
    def __init__(self):
        self.cache = {}

    def set(self, key: str, value: any):
        self.cache[key] = value
        # print(f"[Host Cache] Set: {key}")

    def get(self, key: str) -> any:
        # print(f"[Host Cache] Get: {key}")
        return self.cache.get(key)

    def remove(self, key: str):
        if key in self.cache:
            del self.cache[key]
            # print(f"[Host Cache] Removed: {key}")

    def clear(self):
        self.cache.clear()
        # print("[Host Cache] Cleared")

    def items(self):
        return self.cache.items()

    def __len__(self):
        return len(self.cache)


class KVCacheReplicator:
    """Manages the replication of critical KV cache nodes from GPU to Host memory.
    Criticality is determined by explicit marking or a simulated access frequency threshold.
    """
    def __init__(self,
                 gpu_cache: GPUKVCache,
                 host_cache: HostKVCache,
                 critical_access_threshold: int = 5):
        self.gpu_cache = gpu_cache
        self.host_cache = host_cache
        self.critical_keys = set() # Keys explicitly marked as critical
        self.access_frequency = collections.defaultdict(int) # Tracks access for dynamic criticality
        self.critical_access_threshold = critical_access_threshold

    def _is_critical(self, key: str) -> bool:
        """Determines if a key is critical based on explicit marking or access frequency."""
        return key in self.critical_keys or self.access_frequency[key] >= self.critical_access_threshold

    def mark_critical(self, key: str):
        """Explicitly marks a key as critical for replication."""
        self.critical_keys.add(key)
        # print(f"[Replicator] Marked '{key}' as critical.")
        # Immediately replicate if it's already in GPU cache
        if self.gpu_cache.get(key) is not None:
            self.replicate_key(key, self.gpu_cache.get(key))

    def replicate_key(self, key: str, value: any):
        """Replicates a single KV pair to the host cache."""
        self.host_cache.set(key, value)
        # print(f"[Replicator] Replicated '{key}' to host cache.")

    def add_to_cache(self, key: str, value: any, force_critical: bool = False):
        """Adds a KV pair to the GPU cache and replicates if critical or forced."""
        self.gpu_cache.set(key, value)
        if force_critical:
            self.mark_critical(key)
        if self._is_critical(key):
            self.replicate_key(key, value)
        # print(f"[Replicator] Added '{key}' to GPU cache, replicated if critical.")

    def get_from_cache(self, key: str) -> any:
        """Retrieves a KV pair, trying GPU first, then Host. Updates access frequency."""
        self.access_frequency[key] += 1 # Update frequency on access

        value = self.gpu_cache.get(key)
        if value is not None:
            # If it becomes critical due to access, replicate it
            if self._is_critical(key) and self.host_cache.get(key) is None:
                self.replicate_key(key, value)
            return value

        # GPU miss, try host cache
        value = self.host_cache.get(key)
        if value is not None:
            # If found in host, put back to GPU (simulating recovery/prefetch)
            self.gpu_cache.set(key, value)
            # print(f"[Replicator] Recovered '{key}' from host to GPU cache.")
            return value

        # print(f"[Replicator] Cache miss for '{key}' in both GPU and Host.")
        return None

    def replicate_all_critical_nodes(self):
        """Iterates through current GPU cache and replicates all identified critical nodes."""
        # print("[Replicator] Performing bulk replication of critical nodes...")
        for key, value in self.gpu_cache.items():
            if self._is_critical(key):
                self.replicate_key(key, value)
        # print("[Replicator] Bulk replication complete.")

    def simulate_gpu_failure(self):
        """Simulates a GPU failure by clearing the GPU cache."""
        # print("\n!!! SIMULATING GPU FAILURE !!!\n")
        self.gpu_cache.clear()

    def recover_from_failure(self):
        """Recovers critical nodes from host cache back to GPU cache."""
        # print("\n!!! INITIATING RECOVERY FROM GPU FAILURE !!!\n")
        recovered_count = 0
        for key in self.critical_keys:
            # Also check if keys that became critical via frequency are in host
            host_value = self.host_cache.get(key)
            if host_value is not None:
                self.gpu_cache.set(key, host_value)
                recovered_count += 1
        # Also recover based on frequency if they were replicated
        for key, value in self.host_cache.items():
            if key not in self.gpu_cache.cache and self._is_critical(key):
                 self.gpu_cache.set(key, value)
                 recovered_count += 1

        # print(f"[Replicator] Recovered {recovered_count} critical nodes from host to GPU cache.")
        # print("!!! RECOVERY COMPLETE !!!\n")

# Example Usage:
if __name__ == "__main__":
    # Initialize caches and replicator
    gpu_mem = GPUKVCache(capacity=5)
    host_mem = HostKVCache()
    replicator = KVCacheReplicator(gpu_mem, host_mem, critical_access_threshold=2)

    print("--- Initial State ---")
    replicator.add_to_cache("system_prompt_kv", "sys_prompt_data_1", force_critical=True)
    replicator.add_to_cache("doc_prefix_covid_kv", "covid_data_a")
    replicator.add_to_cache("doc_prefix_flu_kv", "flu_data_b")
    replicator.add_to_cache("patient_A_symptoms_kv", "fever,cough")
    replicator.add_to_cache("patient_B_history_kv", "hypertension")

    print(f"GPU Cache size: {len(gpu_mem)}")
    print(f"Host Cache size: {len(host_mem)}")
    print(f"Critical Keys: {replicator.critical_keys}")

    print("\n--- Simulating Usage and Dynamic Criticality ---")
    replicator.get_from_cache("doc_prefix_covid_kv") # Access 1
    replicator.get_from_cache("doc_prefix_flu_kv") # Access 1
    replicator.get_from_cache("doc_prefix_covid_kv") # Access 2 - now critical for replication

    # The 'system_prompt_kv' was force_critical and should be replicated.
    # 'doc_prefix_covid_kv' should now also be replicated because its access frequency hit the threshold.
    replicator.replicate_all_critical_nodes() # Ensures all critical nodes are on host

    print(f"GPU Cache after usage: {len(gpu_mem)}")
    print(f"Host Cache after usage and replication: {len(host_mem)}")
    print(f"Host Cache Content: {host_mem.cache.keys()}")

    print("\n--- Simulating GPU Failure and Recovery ---")
    replicator.simulate_gpu_failure()
    print(f"GPU Cache after failure: {len(gpu_mem)}")

    # Attempt to retrieve a critical item (should trigger recovery or be available from host)
    recovered_sys_prompt = replicator.get_from_cache("system_prompt_kv")
    recovered_covid_prefix = replicator.get_from_cache("doc_prefix_covid_kv")
    non_critical_item = replicator.get_from_cache("patient_A_symptoms_kv")

    print(f"System Prompt (recovered): {recovered_sys_prompt}")
    print(f"COVID Prefix (recovered): {recovered_covid_prefix}")
    print(f"Patient A Symptoms (NOT recovered): {non_critical_item}")

    replicator.recover_from_failure()
    print(f"GPU Cache after recovery: {len(gpu_mem)}")
    print(f"GPU Cache Content: {gpu_mem.cache.keys()}")

    assert recovered_sys_prompt == "sys_prompt_data_1"
    assert recovered_covid_prefix == "covid_data_a"
    assert non_critical_item is None # This should not be recovered as it wasn't critical/replicated
    assert "system_prompt_kv" in gpu_mem.cache
    assert "doc_prefix_covid_kv" in gpu_mem.cache
    assert "patient_A_symptoms_kv" not in gpu_mem.cache

    print("\n--- Demonstration Complete --- ")
