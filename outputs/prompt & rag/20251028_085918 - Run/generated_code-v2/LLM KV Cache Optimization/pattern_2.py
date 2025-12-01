import time
import random

class KVManager:
    def __init__(self):
        self.gpu_kv_cache = {}
        self.host_kv_cache = {}
        self.critical_keys = set()
        self.next_key_id = 0

    def _generate_kv_data(self, prompt):
        generated_data = {}
        # Simulate generating some KV pairs based on the prompt
        if "system_prompt" in prompt:
            key = f"system_prompt_{self.next_key_id}"
            value = f"system_guidelines_for_{self.next_key_id}_data"
            generated_data[key] = value
            self.critical_keys.add(key)
            self.next_key_id += 1
        elif "oncology_knowledge" in prompt:
            key = f"oncology_knowledge_{self.next_key_id}"
            value = f"treatment_efficacy_for_{self.next_key_id}_drug"
            generated_data[key] = value
            self.critical_keys.add(key)
            self.next_key_id += 1
        else:
            key = f"user_query_kv_{self.next_key_id}"
            value = f"intermediate_state_for_{self.next_key_id}_query"
            generated_data[key] = value
            # Randomly mark some non-critical data as critical for demonstration
            if random.random() < 0.2:
                self.critical_keys.add(key)
            self.next_key_id += 1
        
        # Add a few more random non-critical keys for each prompt
        for _ in range(random.randint(1, 3)):
            key = f"non_critical_kv_{self.next_key_id}"
            value = f"ephemeral_data_{self.next_key_id}"
            generated_data[key] = value
            self.next_key_id += 1

        return generated_data

    def simulate_llm_inference(self, prompt):
        print(f"\nSimulating LLM inference for prompt: '{prompt}'")
        new_kv_data = self._generate_kv_data(prompt)
        self.gpu_kv_cache.update(new_kv_data)
        print(f"Added {len(new_kv_data)} new KV pairs to GPU cache.")
        print(f"Current GPU cache size: {self.get_gpu_cache_size()}")
        print(f"Critical keys identified: {len(self.critical_keys)}")

    def replicate_critical_nodes(self):
        print("\nReplicating critical nodes from GPU to Host memory...")
        replicated_count = 0
        for key in self.critical_keys:
            if key in self.gpu_kv_cache:
                if self.host_kv_cache.get(key) != self.gpu_kv_cache[key]:
                    self.host_kv_cache[key] = self.gpu_kv_cache[key]
                    replicated_count += 1
            else:
                print(f"Warning: Critical key '{key}' not found in GPU cache during replication.")
        print(f"Replicated {replicated_count} critical nodes. Host cache size: {self.get_host_cache_size()}")

    def simulate_gpu_failure(self):
        print("\n*** Simulating GPU Failure! Clearing GPU cache... ***")
        self.gpu_kv_cache.clear()
        print(f"GPU cache size after failure: {self.get_gpu_cache_size()}")

    def recover_from_failure(self):
        print("\nAttempting to recover critical nodes from Host memory...")
        recovered_count = 0
        for key in self.critical_keys:
            if key in self.host_kv_cache:
                self.gpu_kv_cache[key] = self.host_kv_cache[key]
                recovered_count += 1
        print(f"Recovered {recovered_count} critical nodes to GPU cache.")
        print(f"GPU cache size after recovery: {self.get_gpu_cache_size()}")

    def get_gpu_cache_size(self):
        return len(self.gpu_kv_cache)

    def get_host_cache_size(self):
        return len(self.host_kv_cache)


if __name__ == "__main__":
    kv_manager = KVManager()

    print("--- Initial State ---")
    print(f"GPU cache size: {kv_manager.get_gpu_cache_size()}")
    print(f"Host cache size: {kv_manager.get_host_cache_size()}")

    # Step 1: Simulate initial LLM inferences with critical system prompts
    kv_manager.simulate_llm_inference("Initialize system_prompt: general medical guidelines")
    kv_manager.simulate_llm_inference("Load oncology_knowledge: common drug interactions")
    kv_manager.simulate_llm_inference("User query: patient symptoms")
    kv_manager.simulate_llm_inference("User query: treatment options for lung cancer")
    kv_manager.simulate_llm_inference("Load oncology_knowledge: latest clinical trials data")

    # Step 2: Replicate critical nodes
    kv_manager.replicate_critical_nodes()

    print("\n--- State after initial replication ---")
    print(f"GPU cache size: {kv_manager.get_gpu_cache_size()}")
    print(f"Host cache size: {kv_manager.get_host_cache_size()}")
    print(f"Critical keys: {list(kv_manager.critical_keys)}")
    # print("Host cache content:", kv_manager.host_kv_cache) # Uncomment to see host cache content

    # Step 3: Simulate more LLM inferences (some new critical, some not)
    kv_manager.simulate_llm_inference("User query: side effects of chemotherapy regimen")
    kv_manager.simulate_llm_inference("Initialize system_prompt: emergency protocols")
    kv_manager.simulate_llm_inference("User query: genetic markers for breast cancer")

    # Step 4: Replicate again to catch newly critical nodes
    kv_manager.replicate_critical_nodes()

    print("\n--- State after second replication ---")
    print(f"GPU cache size: {kv_manager.get_gpu_cache_size()}")
    print(f"Host cache size: {kv_manager.get_host_cache_size()}")
    print(f"Critical keys: {list(kv_manager.critical_keys)}")

    # Step 5: Simulate GPU failure
    kv_manager.simulate_gpu_failure()

    # Step 6: Recover from failure
    kv_manager.recover_from_failure()

    print("\n--- State after recovery ---")
    print(f"GPU cache size: {kv_manager.get_gpu_cache_size()}")
    print(f"Host cache size: {kv_manager.get_host_cache_size()}")
    print(f"Critical keys: {list(kv_manager.critical_keys)}")

    # Verify that recovered critical keys are indeed present in GPU cache
    print("\nVerification of recovered critical nodes:")
    for key in kv_manager.critical_keys:
        if key in kv_manager.gpu_kv_cache:
            print(f"- Critical key '{key}' is present in GPU cache after recovery.")
        else:
            print(f"- ERROR: Critical key '{key}' is MISSING from GPU cache after recovery!")

    print("\nSimulation complete.")
