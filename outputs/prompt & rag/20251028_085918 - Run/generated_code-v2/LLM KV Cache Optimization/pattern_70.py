class GPUSimulatedKVStore:
    def __init__(self):
        self.cache = {}
        self.critical_keys = set()

    def set(self, key, value):
        self.cache[key] = value

    def get(self, key):
        return self.cache.get(key)

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
            if key in self.critical_keys:
                self.critical_keys.remove(key)

    def mark_critical(self, key):
        if key in self.cache:
            self.critical_keys.add(key)

    def get_critical_nodes(self):
        critical_nodes = {key: self.cache[key] for key in self.critical_keys if key in self.cache}
        return critical_nodes

    def clear(self):
        self.cache.clear()
        self.critical_keys.clear()


class HostKVStore:
    def __init__(self):
        self.cache = {}

    def set(self, key, value):
        self.cache[key] = value

    def get(self, key):
        return self.cache.get(key)

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        self.cache.clear()


class KVReplicationService:
    def __init__(self, gpu_cache: GPUSimulatedKVStore, host_cache: HostKVStore):
        self.gpu_cache = gpu_cache
        self.host_cache = host_cache

    def replicate_critical_nodes(self):
        critical_data = self.gpu_cache.get_critical_nodes()
        for key, value in critical_data.items():
            self.host_cache.set(key, value)
        

    def recover_from_gpu_failure(self):
        self.gpu_cache.clear() # Simulate GPU cache being lost
        for key, value in self.host_cache.cache.items():
            self.gpu_cache.set(key, value)
            self.gpu_cache.mark_critical(key) # Re-mark as critical after recovery


# Example Usage and Simulation
if __name__ == "__main__":
    gpu_kv_store = GPUSimulatedKVStore()
    host_kv_store = HostKVStore()
    replication_service = KVReplicationService(gpu_kv_store, host_kv_store)

    print("--- Initial State ---")
    print(f"GPU Cache: {gpu_kv_store.cache}")
    print(f"Host Cache: {host_kv_store.cache}")

    # Simulate adding data to GPU cache
    gpu_kv_store.set("system_prompt_kv", {"prompt": "You are a medical assistant."})
    gpu_kv_store.set("drug_interaction_cache_id_123", {"drug_a": "aspirin", "drug_b": "warfarin", "interaction": "high_risk"})
    gpu_kv_store.set("patient_context_session_abc", {"patient_id": "P101", "conditions": ["hypertension"]})
    gpu_kv_store.set("common_diagnosis_flow_kv", {"symptoms": "fever, cough", "diagnosis": "common cold"})

    # Mark critical nodes for replication
    gpu_kv_store.mark_critical("system_prompt_kv")
    gpu_kv_store.mark_critical("drug_interaction_cache_id_123")
    print("\n--- After Adding Data and Marking Critical Nodes ---")
    print(f"GPU Cache: {gpu_kv_store.cache}")
    print(f"GPU Critical Keys: {gpu_kv_store.critical_keys}")
    print(f"Host Cache: {host_kv_store.cache}") # Host cache should still be empty

    # Replicate critical nodes from GPU to Host
    replication_service.replicate_critical_nodes()
    print("\n--- After Replication ---")
    print(f"GPU Cache: {gpu_kv_store.cache}")
    print(f"Host Cache: {host_kv_store.cache}") # Host cache should now have critical data

    # Simulate a GPU failure (clear GPU cache)
    print("\n--- Simulating GPU Failure ---")
    gpu_kv_store.clear()
    print(f"GPU Cache (after failure): {gpu_kv_store.cache}")
    print(f"Host Cache (intact): {host_kv_store.cache}")

    # Recover from GPU failure
    replication_service.recover_from_gpu_failure()
    print("\n--- After Recovery from Host Cache ---")
    print(f"GPU Cache (after recovery): {gpu_kv_store.cache}")
    print(f"GPU Critical Keys (after recovery): {gpu_kv_store.critical_keys}")
    print(f"Host Cache: {host_kv_store.cache}")

    # Verify data integrity after recovery
    assert gpu_kv_store.get("system_prompt_kv") == {"prompt": "You are a medical assistant."}
    assert gpu_kv_store.get("drug_interaction_cache_id_123") == {"drug_a": "aspirin", "drug_b": "warfarin", "interaction": "high_risk"}
    assert gpu_kv_store.get("patient_context_session_abc") is None # Non-critical data was lost
    print("\nRecovery successful for critical nodes!")
