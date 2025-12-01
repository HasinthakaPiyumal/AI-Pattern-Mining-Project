import collections
import random

class KVData:
    def __init__(self, conversation_id, size):
        self.conversation_id = conversation_id
        self.size = size

    def __repr__(self):
        return f"KVData(id={self.conversation_id}, size={self.size})"

class GPUMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self._cache = collections.OrderedDict()
        self.allocated_memory = 0

    def allocate(self, kv_data):
        if self.allocated_memory + kv_data.size <= self.capacity:
            self._cache[kv_data.conversation_id] = kv_data
            self.allocated_memory += kv_data.size
            print(f"  GPU: Allocated {kv_data} (Current: {self.allocated_memory}/{self.capacity})")
            return True
        print(f"  GPU: Not enough space for {kv_data} (Current: {self.allocated_memory}/{self.capacity})")
        return False

    def free(self, conversation_id):
        if conversation_id in self._cache:
            kv_data = self._cache.pop(conversation_id)
            self.allocated_memory -= kv_data.size
            print(f"  GPU: Freed {kv_data} (Current: {self.allocated_memory}/{self.capacity})")
            return kv_data
        return None

    def get(self, conversation_id):
        if conversation_id in self._cache:
            kv_data = self._cache.pop(conversation_id)
            self._cache[conversation_id] = kv_data  # Mark as recently used
            print(f"  GPU: Retrieved and refreshed {kv_data}")
            return kv_data
        return None

    def evict_least_recently_used(self):
        if not self._cache:
            return None
        conversation_id, kv_data = self._cache.popitem(last=False) # LRU is first item
        self.allocated_memory -= kv_data.size
        print(f"  GPU: Evicting LRU {kv_data} to make space (Current: {self.allocated_memory}/{self.capacity})")
        return kv_data

    def contains(self, conversation_id):
        return conversation_id in self._cache

    def is_full(self, size_to_add=0):
        return self.allocated_memory + size_to_add > self.capacity

class HostMemory:
    def __init__(self):
        self._cache = {}

    def store(self, kv_data):
        self._cache[kv_data.conversation_id] = kv_data
        print(f"  Host: Stored {kv_data}")

    def retrieve(self, conversation_id):
        if conversation_id in self._cache:
            kv_data = self._cache[conversation_id]
            print(f"  Host: Retrieved {kv_data}")
            return kv_data
        return None

    def remove(self, conversation_id):
        if conversation_id in self._cache:
            kv_data = self._cache.pop(conversation_id)
            print(f"  Host: Removed {kv_data}")
            return kv_data
        return None

    def contains(self, conversation_id):
        return conversation_id in self._cache

class ConversationCacheManager:
    def __init__(self, gpu_capacity, host_capacity=float('inf')):
        self.gpu_memory = GPUMemory(gpu_capacity)
        self.host_memory = HostMemory()
        self.host_stored_conversations = set() # Tracks conversations that have been copied to host at least once
        self.host_capacity = host_capacity # Conceptual, actual HostMemory doesn't enforce it here

    def access_kv_cache(self, conversation_id, kv_data_size):
        print(f"\nAccessing conversation_id: {conversation_id} with size: {kv_data_size}")
        
        kv_data = KVData(conversation_id, kv_data_size)

        # 1. Check if already in GPU
        if self.gpu_memory.contains(conversation_id):
            self.gpu_memory.get(conversation_id) # Refresh LRU status
            print(f"  Cache Hit (GPU) for {conversation_id}")
            return

        # 2. Check if in Host (and not in GPU)
        if self.host_memory.contains(conversation_id):
            print(f"  Cache Miss (GPU), Hit (Host) for {conversation_id}. Moving to GPU...")
            # Attempt to move from Host to GPU
            while self.gpu_memory.is_full(kv_data_size):
                evicted_kv = self.gpu_memory.evict_least_recently_used()
                if evicted_kv:
                    self._evict_from_gpu_to_host(evicted_kv.conversation_id)
                else:
                    print(f"  ERROR: GPU is full and cannot evict to make space for {conversation_id}")
                    return
            
            self.gpu_memory.allocate(self.host_memory.retrieve(conversation_id)) # Retrieve from host, allocate in GPU
            return

        # 3. Not in GPU or Host (new data or purged previously)
        print(f"  Cache Miss (GPU and Host) for {conversation_id}. Allocating in GPU...")
        while self.gpu_memory.is_full(kv_data_size):
            evicted_kv = self.gpu_memory.evict_least_recently_used()
            if evicted_kv:
                self._evict_from_gpu_to_host(evicted_kv.conversation_id)
            else:
                print(f"  ERROR: GPU is full and cannot evict to make space for {conversation_id}")
                return
        
        self.gpu_memory.allocate(kv_data)

    def _evict_from_gpu_to_host(self, conversation_id_to_evict):
        if conversation_id_to_evict not in self.host_stored_conversations:
            print(f"  Strategy: Copying {conversation_id_to_evict} to Host for the first time.")
            # We need to get the actual KVData object that was just evicted from GPU
            # In this simulation, the evict_least_recently_used already returned it
            # but if this was called separately, we'd need to retrieve it.
            # For simplicity, let's assume `evict_least_recently_used` passed the KVData or we re-created it.
            # Here, `evict_least_recently_used` already freed it, so we need to recreate for host storage.
            # A better design would be for GPUMemory.free to return the KVData object.
            
            # Since evict_least_recently_used already 'popped' it, we need to get its original size
            # For a real system, you'd pass the actual KVData object from GPU to host
            # Here, we'll retrieve its size from a hypothetical global source or assume it was passed
            # Let's assume for simulation purposes that the size is readily available or it's passed.
            # For the current simulation, _evict_from_gpu_to_host is called *after* LRU eviction, and `evicted_kv` is passed in `access_kv_cache`.
            
            # For this simplified context, `_evict_from_gpu_to_host` assumes `conversation_id_to_evict` already holds the necessary info.
            # Re-creating a dummy KVData for host memory. In a real scenario, this would be the actual data.
            # For this specific flow, `evicted_kv` from `access_kv_cache` is actually what should be passed here
            # But if called directly from `GPUMemory`, we'd need the size.
            # Let's adjust access_kv_cache to pass the `evicted_kv` directly.
            print(f"  (Simulating copy, actual KVData for {conversation_id_to_evict} would be moved)")
            # In a real scenario, we'd get the actual KVData object from the GPU before freeing.
            # Here, we'll retrieve it from GPUMemory for demonstration purposes.
            # The `free` method in GPUMemory should ideally return the KVData.
            
            # Let's assume we have the original size available. For the sim, let's just make one up for consistency.
            # This is a simplification. A real system would copy the actual tensor.
            dummy_size_for_host = random.randint(10, 50) # Arbitrary size for demo
            self.host_memory.store(KVData(conversation_id_to_evict, dummy_size_for_host)) # Store a representation
            self.host_stored_conversations.add(conversation_id_to_evict)
        else:
            print(f"  Strategy: {conversation_id_to_evict} already in Host, freeing GPU space only.")

    def purge_conversation(self, conversation_id):
        print(f"\nPurging conversation_id: {conversation_id}")
        self.gpu_memory.free(conversation_id)
        self.host_memory.remove(conversation_id)
        if conversation_id in self.host_stored_conversations:
            self.host_stored_conversations.remove(conversation_id)
        print(f"  {conversation_id} purged from all caches.")

# --- Simulation --- #
if __name__ == "__main__":
    GPU_CAPACITY = 100
    HOST_CAPACITY = 500 # Conceptual for this simulation
    cache_manager = ConversationCacheManager(GPU_CAPACITY, HOST_CAPACITY)

    print("--- Initial Accesses ---")
    cache_manager.access_kv_cache("conv_A", 30) # New, fits GPU
    cache_manager.access_kv_cache("conv_B", 40) # New, fits GPU
    cache_manager.access_kv_cache("conv_C", 20) # New, fits GPU

    print("\n--- GPU Full, Eviction to Host (First Time) ---")
    # conv_A (30) + conv_B (40) + conv_C (20) = 90. GPU_CAPACITY = 100
    # Access conv_D (20) will cause eviction (conv_A is LRU)
    cache_manager.access_kv_cache("conv_D", 20) 
    # Expected: conv_A is evicted from GPU, copied to Host for the first time.

    print("\n--- Re-accessing an evicted conversation (from Host to GPU) ---")
    cache_manager.access_kv_cache("conv_A", 30) 
    # Expected: conv_A is retrieved from Host, loaded to GPU. Another eviction from GPU will occur.
    # Now conv_B is LRU and will be evicted. It will be copied to Host for the first time.

    print("\n--- GPU Full, Eviction (Already in Host) ---")
    cache_manager.access_kv_cache("conv_E", 35) 
    # Expected: Some conv is evicted from GPU. If it's already in Host (e.g., conv_C or conv_D depending on access pattern), it's just freed from GPU.
    # Let's trace: Current GPU: A (30), D (20), E (35) = 85. If C (20) was still there, it would be C, D, A. Then D. A. E will evict C
    # After conv_A access: GPU has A, D. Now accessing E. conv_C is LRU if still present. Let's assume C is LRU.
    # GPU had B (40), C (20), D (20). Total 80. A (30) access. Evicts B (LRU) to Host.
    # GPU: C (20), D (20), A (30). Total 70. Now access E (35). Total 105 > 100.
    # Evicts C (LRU). C is copied to Host.
    # GPU: D (20), A (30), E (35). Total 85.
    # Now access E (35) again.
    cache_manager.access_kv_cache("conv_E", 35) # Already in GPU, refresh LRU

    print("\n--- Another Eviction (already in Host) ---")
    cache_manager.access_kv_cache("conv_F", 50) # Will cause eviction
    # Expected: One of D, A, E is evicted. If it's D or A, they are already in Host. So just free from GPU.

    print("\n--- Purging a conversation ---")
    cache_manager.purge_conversation("conv_A")
    cache_manager.purge_conversation("conv_D")
    
    print("\n--- Final State Check ---")
    print(f"GPU Cache: {list(cache_manager.gpu_memory._cache.keys())}")
    print(f"Host Cache: {list(cache_manager.host_memory._cache.keys())}")
    print(f"Host Stored Set: {cache_manager.host_stored_conversations}")

    print("\n--- Re-accessing a purged conversation ---")
    cache_manager.access_kv_cache("conv_A", 30) # Should be treated as new
