class KVCacheManager:
    def __init__(self):
        self.gpu_cache = {}
        self.host_cache = {}

    def add_to_gpu_cache(self, key, value):
        self.gpu_cache[key] = value

    def replicate_critical_to_host(self, critical_keys):
        for key in critical_keys:
            if key in self.gpu_cache:
                self.host_cache[key] = self.gpu_cache[key]

    def simulate_gpu_failure(self):
        self.gpu_cache = {}

    def recover_from_host(self, critical_keys):
        for key in critical_keys:
            if key in self.host_cache:
                self.gpu_cache[key] = self.host_cache[key]

    def get_from_gpu_cache(self, key):
        return self.gpu_cache.get(key)

    def print_cache_status(self):
        print("\n--- Cache Status ---")
        print(f"GPU Cache: {self.gpu_cache}")
        print(f"Host Cache: {self.host_cache}")
        print("--------------------")

# Workflow Simulation
kv_cache_manager = KVCacheManager()

critical_keys = ["system_prompt_kv", "faq_product_A_prefix_kv"]

# 1. Populate GPU Cache
kv_cache_manager.add_to_gpu_cache("system_prompt_kv", "system_context_embedding_001")
kv_cache_manager.add_to_gpu_cache("faq_product_A_prefix_kv", "faq_product_A_embedding_prefix_001")
kv_cache_manager.add_to_gpu_cache("non_critical_query_kv", "user_query_embedding_001")
kv_cache_manager.print_cache_status()

# 2. Replication
kv_cache_manager.replicate_critical_to_host(critical_keys)
kv_cache_manager.print_cache_status()

# 3. Simulate Operation (before failure)
print("\n--- Before GPU Failure ---")
print(f"Retrieving 'system_prompt_kv': {kv_cache_manager.get_from_gpu_cache('system_prompt_kv')}")
print(f"Retrieving 'non_critical_query_kv': {kv_cache_manager.get_from_gpu_cache('non_critical_query_kv')}")

# 4. Simulate GPU Failure
kv_cache_manager.simulate_gpu_failure()
print("\n!!! GPU FAILURE SIMULATED !!!")
kv_cache_manager.print_cache_status()

# 5. Attempt Retrieval (Post-Failure)
print("\n--- After GPU Failure ---")
print(f"Retrieving 'system_prompt_kv': {kv_cache_manager.get_from_gpu_cache('system_prompt_kv')}")
print(f"Retrieving 'non_critical_query_kv': {kv_cache_manager.get_from_gpu_cache('non_critical_query_kv')}")

# 6. Recovery
kv_cache_manager.recover_from_host(critical_keys)
print("\n--- Recovering from Host Memory ---")
kv_cache_manager.print_cache_status()

# 7. Verify Recovery
print("\n--- After Recovery ---")
print(f"Retrieving 'system_prompt_kv': {kv_cache_manager.get_from_gpu_cache('system_prompt_kv')}")
print(f"Retrieving 'faq_product_A_prefix_kv': {kv_cache_manager.get_from_gpu_cache('faq_product_A_prefix_kv')}")
print(f"Retrieving 'non_critical_query_kv': {kv_cache_manager.get_from_gpu_cache('non_critical_query_kv')}")