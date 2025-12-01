import collections

class LLMKVCache:
    def __init__(self, gpu_capacity: int):
        self.gpu_cache = collections.OrderedDict()
        self.host_cache = {}
        self.swapped_out_once_nodes = set()
        self.gpu_capacity = gpu_capacity

    def _evict_lru_from_gpu(self):
        if not self.gpu_cache:
            return

        node_id, kv_tensors = self.gpu_cache.popitem(last=False)
        if node_id not in self.swapped_out_once_nodes:
            self.host_cache[node_id] = kv_tensors
            self.swapped_out_once_nodes.add(node_id)
            print(f"[EVICTION] Node '{node_id}' evicted from GPU to Host (first time, copied).")
        else:
            print(f"[EVICTION] Node '{node_id}' evicted from GPU (already in Host, freed GPU memory).")

    def add_or_update_gpu_cache(self, node_id: str, kv_tensors: str):
        if node_id in self.gpu_cache:
            self.gpu_cache.move_to_end(node_id)
        else:
            if len(self.gpu_cache) >= self.gpu_capacity:
                self._evict_lru_from_gpu()
            self.gpu_cache[node_id] = kv_tensors
        print(f"[GPU ADD/UPDATE] Node '{node_id}' added/updated in GPU cache.")

    def retrieve_kv_tensors(self, node_id: str):
        if node_id in self.gpu_cache:
            self.gpu_cache.move_to_end(node_id)
            print(f"[RETRIEVE] Node '{node_id}' found in GPU cache.")
            return self.gpu_cache[node_id]

        if node_id in self.host_cache:
            print(f"[PROMOTION] Node '{node_id}' promoted from Host to GPU.")
            if len(self.gpu_cache) >= self.gpu_capacity:
                self._evict_lru_from_gpu()
            kv_tensors = self.host_cache[node_id]
            self.gpu_cache[node_id] = kv_tensors
            self.gpu_cache.move_to_end(node_id)
            return kv_tensors

        print(f"[FETCH NEW] Node '{node_id}' not found. Simulating fetch/creation.")
        new_kv_tensors = f"KV_Tensors_for_{node_id}"
        self.add_or_update_gpu_cache(node_id, new_kv_tensors)
        return new_kv_tensors

    def get_cache_status(self):
        print("\n--- Cache Status ---")
        print(f"GPU Cache ({len(self.gpu_cache)}/{self.gpu_capacity}): {list(self.gpu_cache.keys())}")
        print(f"Host Cache ({len(self.host_cache)}): {list(self.host_cache.keys())}")
        print(f"Swapped Out Once Nodes: {list(self.swapped_out_once_nodes)}")
        print("--------------------\n")


if __name__ == "__main__":
    gpu_capacity = 3
    cache = LLMKVCache(gpu_capacity)

    print("--- Simulation Start ---")

    # 1. Initial additions to the GPU cache
    print("\n--- Step 1: Initial additions ---")
    cache.retrieve_kv_tensors("conv_A")
    cache.retrieve_kv_tensors("conv_B")
    cache.retrieve_kv_tensors("conv_C")
    cache.get_cache_status()

    # 2. Eviction of LRU items from GPU to host (first time copy)
    print("\n--- Step 2: GPU full, conv_A evicted to Host (first time) ---")
    cache.retrieve_kv_tensors("conv_D") # conv_A should be evicted
    cache.get_cache_status()

    # 3. Re-accessing items in GPU to change LRU order
    print("\n--- Step 3: Re-access conv_B to make it MRU ---")
    cache.retrieve_kv_tensors("conv_B")
    cache.get_cache_status()

    # 4. Eviction of items from GPU that have already been copied to host (no new copy operation)
    print("\n--- Step 4: GPU full again, conv_C evicted (already in Host) ---")
    cache.retrieve_kv_tensors("conv_E") # conv_C should be evicted
    cache.get_cache_status()

    # 5. Promotion of items from host back to GPU
    print("\n--- Step 5: Promote conv_A from Host to GPU ---")
    cache.retrieve_kv_tensors("conv_A") # conv_D should be evicted to make space
    cache.get_cache_status()

    print("\n--- Step 6: Access conv_C (already swapped out once) ---")
    cache.retrieve_kv_tensors("conv_C") # conv_E should be evicted to make space
    cache.get_cache_status()

    print("\n--- Step 7: Another eviction, conv_B to host (already swapped out once) ---")
    cache.retrieve_kv_tensors("conv_F") # conv_B should be evicted
    cache.get_cache_status()

    print("--- Simulation End ---")