
class MockKVTensor:
    def __init__(self, kv_id, size_bytes=100):
        self.kv_id = kv_id
        self.size_bytes = size_bytes
        self.data = f"KV_DATA_{kv_id}"

    def get_size(self):
        return self.size_bytes

    def __repr__(self):
        return f"MockKVTensor(id={self.kv_id}, size={self.size_bytes}B)"


class SwapOutOnlyOnceKVManager:
    def __init__(self, gpu_capacity_bytes=500, host_capacity_bytes=1000):
        self.gpu_capacity_bytes = gpu_capacity_bytes
        self.host_capacity_bytes = host_capacity_bytes

        self.gpu_memory_pool = {}  # {kv_id: MockKVTensor}
        self.host_memory_store = {} # {kv_id: MockKVTensor}
        self.evicted_once_tracker = set() # {kv_id}
        self.lru_queue = [] # List for LRU: most recently used is at the end

        self._current_gpu_usage_bytes = 0
        self._current_host_usage_bytes = 0

    def _update_lru(self, kv_id):
        if kv_id in self.lru_queue:
            new_lru_queue = []
            for item in self.lru_queue:
                if item != kv_id:
                    new_lru_queue.append(item)
            self.lru_queue = new_lru_queue
        self.lru_queue.append(kv_id)

    def _evict_from_gpu(self, required_space_bytes):
        evicted_kvs = []
        while self._current_gpu_usage_bytes + required_space_bytes > self.gpu_capacity_bytes and self.lru_queue:
            kv_id_to_evict = self.lru_queue[0] # Least recently used
            self.lru_queue = self.lru_queue[1:] # Remove from front (simulates popleft)

            if kv_id_to_evict not in self.gpu_memory_pool:
                continue

            kv_tensor = self.gpu_memory_pool[kv_id_to_evict]
            evicted_kvs.append(kv_id_to_evict)
            kv_size = kv_tensor.get_size()

            if kv_id_to_evict not in self.evicted_once_tracker:
                # First time eviction: copy to host
                if self._current_host_usage_bytes + kv_size > self.host_capacity_bytes:
                    pass
                else:
                    self.host_memory_store[kv_id_to_evict] = kv_tensor
                    self._current_host_usage_bytes += kv_size
                    self.evicted_once_tracker.add(kv_id_to_evict)
            else:
                # Subsequent eviction: just free GPU, host already has a copy
                pass

            self._free_gpu_memory(kv_id_to_evict)
        return evicted_kvs

    def _free_gpu_memory(self, kv_id):
        if kv_id in self.gpu_memory_pool:
            kv_tensor = self.gpu_memory_pool.pop(kv_id)
            self._current_gpu_usage_bytes -= kv_tensor.get_size()

    def store_kv(self, kv_id, kv_tensor):
        kv_size = kv_tensor.get_size()

        if kv_id in self.gpu_memory_pool:
            self._update_lru(kv_id)
            return

        if kv_id in self.host_memory_store:
            self.retrieve_kv(kv_id)
            return

        if self._current_gpu_usage_bytes + kv_size > self.gpu_capacity_bytes:
            self._evict_from_gpu(kv_size)

        if self._current_gpu_usage_bytes + kv_size <= self.gpu_capacity_bytes:
            self.gpu_memory_pool[kv_id] = kv_tensor
            self._current_gpu_usage_bytes += kv_size
            self._update_lru(kv_id)
        else:
            pass

    def retrieve_kv(self, kv_id):
        if kv_id in self.gpu_memory_pool:
            self._update_lru(kv_id)
            return self.gpu_memory_pool[kv_id]
        elif kv_id in self.host_memory_store:
            kv_tensor = self.host_memory_store[kv_id]
            kv_size = kv_tensor.get_size()

            if self._current_gpu_usage_bytes + kv_size > self.gpu_capacity_bytes:
                self._evict_from_gpu(kv_size)

            if self._current_gpu_usage_bytes + kv_size <= self.gpu_capacity_bytes:
                self.gpu_memory_pool[kv_id] = kv_tensor
                self._current_gpu_usage_bytes += kv_size
                self._update_lru(kv_id)
                return kv_tensor
            else:
                pass
        return None

    def get_gpu_usage(self):
        return self._current_gpu_usage_bytes

    def get_host_usage(self):
        return self._current_host_usage_bytes

    def get_stats(self):
        return {
            "gpu_usage": self._current_gpu_usage_bytes,
            "gpu_capacity": self.gpu_capacity_bytes,
            "host_usage": self._current_host_usage_bytes,
            "host_capacity": self.host_capacity_bytes,
            "num_kvs_gpu": len(self.gpu_memory_pool),
            "num_kvs_host": len(self.host_memory_store),
            "num_evicted_once": len(self.evicted_once_tracker)
        }


class MockLLM:
    def __init__(self, kv_manager):
        self.kv_manager = kv_manager
        self.kv_counter = 0
        self.known_kv_ids = []
        self.access_index = 0

    def _generate_kv_tensors(self, num_tensors_hint):
        kvs = []
        for i in range(num_tensors_hint):
            self.kv_counter += 1
            kv_id = f"KV_{self.kv_counter}"
            size = ((self.kv_counter + i) % 3 + 1) * 70
            kv_tensor = MockKVTensor(kv_id, size)
            kvs.append(kv_tensor)
            self.known_kv_ids.append(kv_id)
        return kvs

    def infer(self, processed_input, patient_id):
        num_new_kvs = (len(processed_input) % 3) + 1
        new_kvs = self._generate_kv_tensors(num_new_kvs)
        for kv_tensor in new_kvs:
            self.kv_manager.store_kv(kv_tensor.kv_id, kv_tensor)

        access_limit = 2
        accessed_count = 0
        if self.known_kv_ids:
            for _ in range(access_limit):
                if accessed_count >= len(self.known_kv_ids):
                    break
                kv_id_to_access = self.known_kv_ids[self.access_index % len(self.known_kv_ids)]
                self.kv_manager.retrieve_kv(kv_id_to_access)
                self.access_index += 1
                accessed_count += 1
            
        output = f"Diagnostic suggestion for '{processed_input}': Potential condition X for patient {patient_id}. Consider test Y. (Simulated output)"
        return output


class PatientDataPreprocessor:
    def preprocess(self, patient_data):
        return f"Tokenized_Input: {patient_data['symptoms']} - {patient_data['history']}"


def main():
    kv_manager = SwapOutOnlyOnceKVManager(gpu_capacity_bytes=500, host_capacity_bytes=1000)
    llm_model = MockLLM(kv_manager)
    preprocessor = PatientDataPreprocessor()

    patient_requests = [
        {"patient_id": "P001", "data": {"symptoms": "Fever, cough", "history": "Flu shot last year"}},
        {"patient_id": "P002", "data": {"symptoms": "Headache, nausea", "history": "Migraines in past"}},
        {"patient_id": "P001", "data": {"symptoms": "Fatigue, mild fever", "history": "Follow-up for previous symptoms"}},
        {"patient_id": "P003", "data": {"symptoms": "Chest pain", "history": "Smoker, family heart history"}},
        {"patient_id": "P002", "data": {"symptoms": "Dizziness, blurred vision", "history": "Still experiencing headache"}},
        {"patient_id": "P001", "data": {"symptoms": "Rash, no fever", "history": "Allergies to certain meds"}},
        {"patient_id": "P004", "data": {"symptoms": "Joint pain", "history": "Recent travel"}},
        {"patient_id": "P001", "data": {"symptoms": "Stomach ache", "history": "Ate something suspicious"}},
    ]

    print("--- Starting AI-powered Medical Assistant Simulation ---")
    print(f"Initial GPU Capacity: {kv_manager.gpu_capacity_bytes}B, Host Capacity: {kv_manager.host_capacity_bytes}B")

    for i, request in enumerate(patient_requests):
        print(f"\n--- Processing Request {i+1} for Patient {request['patient_id']} ---")
        processed_input = preprocessor.preprocess(request["data"])
        llm_output = llm_model.infer(processed_input, request["patient_id"])
        print(f"  Result: {llm_output[:80]}...")
        stats = kv_manager.get_stats()
        print(f"  Cache Stats: GPU {stats['gpu_usage']}/{stats['gpu_capacity']}B | Host {stats['host_usage']}/{stats['host_capacity']}B | KVs ever evicted to Host: {stats['num_evicted_once']}")

    print("\n--- Simulation Complete ---")
    final_stats = kv_manager.get_stats()
    print(f"Final GPU Usage: {final_stats['gpu_usage']}B")
    print(f"Final Host Usage: {final_stats['host_usage']}B")
    print(f"Unique KV IDs ever copied to Host: {len(kv_manager.evicted_once_tracker)}")
    print(f"All KV IDs currently in Host store: {list(kv_manager.host_memory_store.keys())}")


if __name__ == "__main__":
    main()
