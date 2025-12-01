"""kv_cache_manager.py: Manages a multi-level KV cache with replication for fault tolerance."""

class KVCacheManager:
    def __init__(self):
        self.gpu_kv_cache = {}
        self.host_kv_cache = {}
        self.critical_nodes = set() # Stores keys of critical nodes

    def add_to_gpu_cache(self, key: str, value: str, is_critical: bool = False):
        """Adds a key-value pair to the GPU cache. Marks as critical if specified."""
        self.gpu_kv_cache[key] = value
        if is_critical:
            self.critical_nodes.add(key)
            print(f"[KVCache] Added critical node '{key}' to GPU cache.")
        else:
            print(f"[KVCache] Added node '{key}' to GPU cache.")

    def replicate_critical_nodes(self):
        """Replicates all critical nodes from GPU cache to host cache."""
        print("[KVCache] Replicating critical nodes to host memory...")
        for key in self.critical_nodes:
            if key in self.gpu_kv_cache:
                self.host_kv_cache[key] = self.gpu_kv_cache[key]
                print(f"[KVCache] Replicated critical node '{key}' to host cache.")
            else:
                print(f"[KVCache] Warning: Critical node '{key}' not found in GPU cache during replication.")
        print("[KVCache] Replication complete.")

    def simulate_gpu_failure(self):
        """Simulates a GPU failure by clearing the GPU KV cache."""
        print("\n[KVCache] Simulating GPU failure: Clearing GPU KV cache...")
        self.gpu_kv_cache.clear()
        print("[KVCache] GPU KV cache cleared. GPU is down.")

    def recover_from_host_cache(self):
        """Recovers critical nodes from the host cache to the GPU cache after a failure."""
        print("\n[KVCache] Recovering critical nodes from host memory...")
        for key in self.critical_nodes:
            if key in self.host_kv_cache:
                self.gpu_kv_cache[key] = self.host_kv_cache[key]
                print(f"[KVCache] Recovered critical node '{key}' from host cache to GPU.")
            else:
                print(f"[KVCache] Warning: Critical node '{key}' not found in host cache during recovery.")
        print("[KVCache] Recovery complete.")

    def get_kv_from_cache(self, key: str):
        """Attempts to retrieve a KV pair, first from GPU, then from Host if critical and GPU fails."""
        if key in self.gpu_kv_cache:
            return self.gpu_kv_cache[key], "GPU"
        elif key in self.critical_nodes and key in self.host_kv_cache:
            # This path would typically be taken after a recovery, but can also serve as a fallback if GPU misses
            return self.host_kv_cache[key], "Host"
        return None, "None"

    def get_gpu_cache_status(self):
        return list(self.gpu_kv_cache.keys())

    def get_host_cache_status(self):
        return list(self.host_kv_cache.keys())


# --- Demo / Example Usage for Medical Diagnosis Assistant --- #
if __name__ == "__main__":
    cache_manager = KVCacheManager()

    # 1. Initialize critical system prompts and frequently accessed medical profiles
    print("--- Initialization ---")
    cache_manager.add_to_gpu_cache("system_prompt_general_medical", "KV_data_for_general_medical_guidelines", is_critical=True)
    cache_manager.add_to_gpu_cache("disease_profile_diabetes", "KV_data_for_diabetes_mellitus", is_critical=True)
    cache_manager.add_to_gpu_cache("patient_context_john_doe_session_1", "KV_data_for_john_doe_session_context", is_critical=False)
    cache_manager.add_to_gpu_cache("medical_literature_summary_hypertension", "KV_data_for_hypertension_literature", is_critical=False)

    print("\nInitial GPU Cache:", cache_manager.get_gpu_cache_status())
    print("Initial Host Cache:", cache_manager.get_host_cache_status())

    # 2. Replicate critical nodes to host memory
    cache_manager.replicate_critical_nodes()
    print("\nAfter Replication GPU Cache:", cache_manager.get_gpu_cache_status())
    print("After Replication Host Cache:", cache_manager.get_host_cache_status())

    # 3. Simulate LLM inference using a critical KV node
    print("\n--- Pre-failure Inference (using GPU cache) ---")
    kv_data, source = cache_manager.get_kv_from_cache("system_prompt_general_medical")
    if kv_data:
        print(f"[LLM Inference] Retrieved '{kv_data}' from {source} for general medical prompt.")

    kv_data, source = cache_manager.get_kv_from_cache("patient_context_john_doe_session_1")
    if kv_data:
        print(f"[LLM Inference] Retrieved '{kv_data}' from {source} for patient context.")

    # 4. Simulate a GPU failure
    cache_manager.simulate_gpu_failure()
    print("\nAfter Failure GPU Cache:", cache_manager.get_gpu_cache_status())
    print("After Failure Host Cache:", cache_manager.get_host_cache_status())

    # 5. Attempt LLM inference after GPU failure - critical should fail to retrieve, non-critical will also fail
    print("\n--- Post-failure Inference Attempt ---")
    kv_data, source = cache_manager.get_kv_from_cache("system_prompt_general_medical")
    if kv_data:
        print(f"[LLM Inference] Retrieved '{kv_data}' from {source} for general medical prompt.")
    else:
        print("[LLM Inference] ERROR: Could not retrieve 'system_prompt_general_medical' from any cache. Requires recovery.")

    kv_data, source = cache_manager.get_kv_from_cache("patient_context_john_doe_session_1")
    if kv_data:
        print(f"[LLM Inference] Retrieved '{kv_data}' from {source} for patient context.")
    else:
        print("[LLM Inference] ERROR: Could not retrieve 'patient_context_john_doe_session_1' from any cache. This was non-critical and lost.")

    # 6. Recover critical nodes from host memory
    cache_manager.recover_from_host_cache()
    print("\nAfter Recovery GPU Cache:", cache_manager.get_gpu_cache_status())
    print("After Recovery Host Cache:", cache_manager.get_host_cache_status())

    # 7. Re-attempt LLM inference for critical nodes - should now succeed
    print("\n--- Post-recovery Inference ---")
    kv_data, source = cache_manager.get_kv_from_cache("system_prompt_general_medical")
    if kv_data:
        print(f"[LLM Inference] Retrieved '{kv_data}' from {source} for general medical prompt.")
    else:
        print("[LLM Inference] ERROR: 'system_prompt_general_medical' not retrieved after recovery. This should not happen.")

    kv_data, source = cache_manager.get_kv_from_cache("disease_profile_diabetes")
    if kv_data:
        print(f"[LLM Inference] Retrieved '{kv_data}' from {source} for diabetes profile.")

    kv_data, source = cache_manager.get_kv_from_cache("patient_context_john_doe_session_1")
    if kv_data:
        print(f"[LLM Inference] Retrieved '{kv_data}' from {source} for patient context.")
    else:
        print("[LLM Inference] Note: Non-critical 'patient_context_john_doe_session_1' was lost and not recovered.")