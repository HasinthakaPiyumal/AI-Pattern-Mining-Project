class KVCacheManager:
    def __init__(self):
        self.gpu_cache = {}
        self.host_cache = {}
        self.critical_keys = set()

    def set_kv(self, key, value, is_critical=False):
        self.gpu_cache[key] = value
        if is_critical:
            self.critical_keys.add(key)
            self._replicate_to_host(key, value)

    def get_kv(self, key):
        return self.gpu_cache.get(key)

    def _replicate_to_host(self, key, value):
        self.host_cache[key] = value

    def simulate_gpu_failure(self):
        print("Simulating GPU failure: Clearing GPU cache...")
        self.gpu_cache = {}

    def recover_from_failure(self):
        print("Recovering from failure: Repopulating critical nodes from host cache...")
        for key in self.critical_keys:
            if key in self.host_cache:
                self.gpu_cache[key] = self.host_cache[key]


if __name__ == "__main__":
    cache_manager = KVCacheManager()

    # Simulate LLM interaction: setting KV pairs
    cache_manager.set_kv("system_prompt_embedding", "embedding_data_123", is_critical=True)
    cache_manager.set_kv("patient_summary_kv", "patient_data_abc", is_critical=False)
    cache_manager.set_kv("medical_guideline_prefix", "guideline_part_xyz", is_critical=True)
    cache_manager.set_kv("current_session_token", "token_value_456")

    print("--- Initial State ---")
    print("GPU Cache:", cache_manager.gpu_cache)
    print("Host Cache:", cache_manager.host_cache)
    print("Critical Keys:", cache_manager.critical_keys)

    # Accessing data before failure
    print("\nAccessing 'system_prompt_embedding' from GPU cache:", cache_manager.get_kv("system_prompt_embedding"))

    # Simulate a GPU failure
    cache_manager.simulate_gpu_failure()
    print("GPU Cache after failure:", cache_manager.gpu_cache)
    print("Accessing 'system_prompt_embedding' after failure (should be None):")
    print(cache_manager.get_kv("system_prompt_embedding"))

    # Recover from the failure
    cache_manager.recover_from_failure()

    print("\n--- After Recovery ---")
    print("GPU Cache after recovery:", cache_manager.gpu_cache)
    print("Host Cache after recovery:", cache_manager.host_cache)
    print("Accessing 'system_prompt_embedding' after recovery (should be restored):")
    print(cache_manager.get_kv("system_prompt_embedding"))

    print("Accessing 'patient_summary_kv' after recovery (should still be None as it wasn't critical):")
    print(cache_manager.get_kv("patient_summary_kv"))