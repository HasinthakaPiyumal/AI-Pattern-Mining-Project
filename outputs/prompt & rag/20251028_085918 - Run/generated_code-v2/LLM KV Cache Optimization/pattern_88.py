import collections
import time
import random

class KVCacheTensor:
    def __init__(self, id, conversation_id, size):
        self.id = id
        self.conversation_id = conversation_id
        self.size = size
        self.in_gpu = False
        self.in_host = False
        self.last_accessed = 0

class GPUMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.current_usage = 0
        self.cached_items = {}
        self.lru_queue = collections.OrderedDict()

    def _update_lru(self, kv_id):
        if kv_id in self.lru_queue:
            self.lru_queue.move_to_end(kv_id)

    def allocate(self, kv_tensor):
        if kv_tensor.id in self.cached_items:
            self._update_lru(kv_tensor.id)
            return True

        if kv_tensor.size > self.capacity:
            print(f"Error: KV tensor {kv_tensor.id} is larger than GPU capacity.")
            return False

        if self.current_usage + kv_tensor.size <= self.capacity:
            self.cached_items[kv_tensor.id] = kv_tensor
            self.current_usage += kv_tensor.size
            kv_tensor.in_gpu = True
            self.lru_queue[kv_tensor.id] = None
            self._update_lru(kv_tensor.id)
            return True
        else:
            return False

    def free(self, kv_id):
        if kv_id in self.cached_items:
            kv_tensor = self.cached_items.pop(kv_id)
            self.current_usage -= kv_tensor.size
            kv_tensor.in_gpu = False
            if kv_id in self.lru_queue:
                del self.lru_queue[kv_id]

    def access(self, kv_id):
        self._update_lru(kv_id)

    def evict_lru(self):
        if not self.lru_queue:
            return None
        kv_id = next(iter(self.lru_queue))
        return kv_id

class HostMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.current_usage = 0
        self.cached_items = {}

    def allocate(self, kv_tensor):
        if kv_tensor.id in self.cached_items:
            return

        if kv_tensor.size > self.capacity:
            print(f"Error: KV tensor {kv_tensor.id} is larger than Host capacity.")
            return

        if self.current_usage + kv_tensor.size <= self.capacity:
            self.cached_items[kv_tensor.id] = kv_tensor
            self.current_usage += kv_tensor.size
            kv_tensor.in_host = True
        else:
            print(f"Error: Host memory full, cannot allocate {kv_tensor.id}.")

    def free(self, kv_id):
        if kv_id in self.cached_items:
            kv_tensor = self.cached_items.pop(kv_id)
            self.current_usage -= kv_tensor.size
            kv_tensor.in_host = False

    def has_item(self, kv_id):
        return kv_id in self.cached_items

class AdaptiveKVCacheManager:
    def __init__(self, gpu_capacity, host_capacity):
        self.gpu_mem = GPUMemory(gpu_capacity)
        self.host_mem = HostMemory(host_capacity)
        self.all_kv_tensors = {}
        self.gpu_to_host_copies = 0
        self.gpu_frees_no_copy = 0

    def process_kv_request(self, kv_id, kv_size, conversation_id):
        print(f"\nProcessing request for KV_ID: {kv_id} (Conv: {conversation_id}, Size: {kv_size})")

        if kv_id not in self.all_kv_tensors:
            kv_tensor = KVCacheTensor(kv_id, conversation_id, kv_size)
            self.all_kv_tensors[kv_id] = kv_tensor
        else:
            kv_tensor = self.all_kv_tensors[kv_id]

        if kv_tensor.in_gpu:
            self.gpu_mem.access(kv_id)
            print(f"  KV tensor {kv_id} already in GPU, refreshing LRU.")
            return

        print(f"  KV tensor {kv_id} not in GPU, attempting to load...")

        # Try to allocate in GPU, if it fails, evict until space is made
        while not self.gpu_mem.allocate(kv_tensor):
            print(f"  GPU full (usage: {self.gpu_mem.current_usage}/{self.gpu_mem.capacity}), need to evict.")
            evicted_kv_id = self.gpu_mem.evict_lru()
            if evicted_kv_id is None:
                print("    GPU is full and no items to evict! Cannot fulfill request.")
                return

            evicted_kv_tensor = self.all_kv_tensors[evicted_kv_id]
            print(f"    Evicting LRU item: {evicted_kv_id} (Size: {evicted_kv_tensor.size}).")

            if not evicted_kv_tensor.in_host:
                self.host_mem.allocate(evicted_kv_tensor)
                self.gpu_to_host_copies += 1
                print(f"      -> Copying {evicted_kv_id} to host memory (first time).")
            else:
                self.gpu_frees_no_copy += 1
                print(f"      -> {evicted_kv_id} already in host, freeing GPU block only.")

            self.gpu_mem.free(evicted_kv_id)
            print(f"      -> Freed {evicted_kv_id} from GPU.")

        kv_tensor.last_accessed = time.time()
        print(f"  Successfully loaded {kv_id} into GPU.")

    def get_status(self):
        return {
            "gpu_usage": f"{self.gpu_mem.current_usage}/{self.gpu_mem.capacity}",
            "host_usage": f"{self.host_mem.current_usage}/{self.host_mem.capacity}",
            "gpu_items": len(self.gpu_mem.cached_items),
            "host_items": len(self.host_mem.cached_items),
            "total_kv_tensors": len(self.all_kv_tensors),
            "gpu_to_host_copies": self.gpu_to_host_copies,
            "gpu_frees_no_copy": self.gpu_frees_no_copy
        }

def simulate_interactions(manager, num_requests):
    conversation_ids = [f"conv_{i}" for i in range(5)]
    kv_id_counter = 0

    for i in range(num_requests):
        # Simulate accessing existing or new KV tensors
        if i < 10 or random.random() < 0.7: # Mostly access existing, some new
            if manager.all_kv_tensors and random.random() < 0.8:
                # Access an existing KV tensor
                kv_id = random.choice(list(manager.all_kv_tensors.keys()))
                kv_tensor = manager.all_kv_tensors[kv_id]
                kv_size = kv_tensor.size
                conversation_id = kv_tensor.conversation_id
            else:
                # Create a new KV tensor
                kv_id_counter += 1
                kv_id = f"kv_{kv_id_counter}"
                kv_size = random.randint(1, 10) * 10 # Simulate varying tensor sizes
                conversation_id = random.choice(conversation_ids)
        else:
            # Create a new KV tensor less frequently
            kv_id_counter += 1
            kv_id = f"kv_{kv_id_counter}"
            kv_size = random.randint(1, 10) * 10
            conversation_id = random.choice(conversation_ids)
        
        manager.process_kv_request(kv_id, kv_size, conversation_id)
        print(f"Current Status: {manager.get_status()}")

if __name__ == "__main__":
    GPU_CAPACITY = 200 # MB
    HOST_CAPACITY = 1000 # MB
    NUM_SIMULATED_REQUESTS = 30

    print("Starting KV Cache Simulation with Swap-Out-Only-Once Strategy")
    manager = AdaptiveKVCacheManager(GPU_CAPACITY, HOST_CAPACITY)

    simulate_interactions(manager, NUM_SIMULATED_REQUESTS)

    print("\n--- Final Cache Status ---")
    final_status = manager.get_status()
    for key, value in final_status.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print("\n--- Host Memory Details ---")
    for kv_id, kv_tensor in manager.host_mem.cached_items.items():
        print(f"  {kv_id} (Conv: {kv_tensor.conversation_id}, Size: {kv_tensor.size})")
