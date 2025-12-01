class MedAssistantKVCache:
    def __init__(self, critical_keys_list):
        self.gpu_cache = {}
        self.host_cache = {}
        self.critical_keys = set(critical_keys_list)

    def _is_critical(self, key):
        return key in self.critical_keys

    def put(self, key, value, is_critical=False):
        self.gpu_cache[key] = value
        if is_critical or self._is_critical(key):
            self.host_cache[key] = value

    def get(self, key):
        if key in self.gpu_cache:
            return self.gpu_cache[key]
        elif self._is_critical(key) and key in self.host_cache:
            return self.host_cache[key]
        return None

    def replicate_critical_nodes(self):
        for key, value in self.gpu_cache.items():
            if self._is_critical(key):
                self.host_cache[key] = value

    def simulate_gpu_failure(self):
        print("\n--- Simulating GPU Failure (clearing gpu_cache) ---")
        self.gpu_cache = {}

    def recover_from_failure(self):
        print("\n--- Recovering from Failure (restoring critical nodes) ---")
        for key, value in self.host_cache.items():
            if self._is_critical(key):
                self.gpu_cache[key] = value

    def get_cache_status(self):
        return {
            "gpu_cache_keys": list(self.gpu_cache.keys()),
            "gpu_cache_size": len(self.gpu_cache),
            "host_cache_keys": list(self.host_cache.keys()),
            "host_cache_size": len(self.host_cache),
            "critical_keys": list(self.critical_keys)
        }

def main():
    critical_keys = ["system_prompt", "medical_guideline_fever", "patient_alert_diabetes"]
    med_assistant_cache = MedAssistantKVCache(critical_keys)

    print("--- Initial Cache Population ---")
    med_assistant_cache.put("system_prompt", "AI Medical Assistant: Provide accurate diagnostic support.", is_critical=True)
    med_assistant_cache.put("medical_guideline_fever", "High fever requires immediate attention.", is_critical=True)
    med_assistant_cache.put("patient_history_john", "John Doe, 45, no prior conditions.")
    med_assistant_cache.put("diagnostic_step_1", "Initial symptom analysis completed.")
    med_assistant_cache.put("patient_alert_diabetes", "Patient has type 2 diabetes.", is_critical=True)

    print("\n--- Cache Status After Initial Population ---")
    status = med_assistant_cache.get_cache_status()
    print(f"GPU Cache: {status['gpu_cache_keys']}, Size: {status['gpu_cache_size']}")
    print(f"Host Cache: {status['host_cache_keys']}, Size: {status['host_cache_size']}")

    print("\n--- Demonstrating Get Operations ---")
    print(f"Get 'system_prompt': {med_assistant_cache.get('system_prompt')}")
    print(f"Get 'patient_history_john': {med_assistant_cache.get('patient_history_john')}")
    print(f"Get 'non_existent_key': {med_assistant_cache.get('non_existent_key')}")

    med_assistant_cache.simulate_gpu_failure()

    print("\n--- Cache Status After GPU Failure ---")
    status = med_assistant_cache.get_cache_status()
    print(f"GPU Cache: {status['gpu_cache_keys']}, Size: {status['gpu_cache_size']}")
    print(f"Host Cache: {status['host_cache_keys']}, Size: {status['host_cache_size']}")

    print("\n--- Demonstrating Get Operations After Failure ---")
    print("(Note: Non-critical data in GPU cache is lost, critical data should still be retrievable from host cache)")
    print(f"Get 'system_prompt' (critical): {med_assistant_cache.get('system_prompt')}")
    print(f"Get 'patient_history_john' (non-critical): {med_assistant_cache.get('patient_history_john')}")

    med_assistant_cache.recover_from_failure()

    print("\n--- Cache Status After Recovery ---")
    status = med_assistant_cache.get_cache_status()
    print(f"GPU Cache: {status['gpu_cache_keys']}, Size: {status['gpu_cache_size']}")
    print(f"Host Cache: {status['host_cache_keys']}, Size: {status['host_cache_size']}")

    print("\n--- Demonstrating Get Operations After Recovery ---")
    print(f"Get 'system_prompt' (critical): {med_assistant_cache.get('system_prompt')}")
    print(f"Get 'patient_history_john' (non-critical): {med_assistant_cache.get('patient_history_john')}")
    print(f"Get 'diagnostic_step_1' (non-critical, should still be None as it wasn't critical): {med_assistant_cache.get('diagnostic_step_1')}")

if __name__ == "__main__":
    main()