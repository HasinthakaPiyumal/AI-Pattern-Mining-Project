class KVCacheManager:
    def __init__(self):
        self._gpu_cache = {}
        self._host_cache = {}

    def add_to_cache(self, key, value, is_critical=False):
        self._gpu_cache[key] = value
        if is_critical:
            self._host_cache[key] = value
        print(f"Added to GPU cache: {key} = {value}")
        if is_critical:
            print(f"Replicated to Host cache (critical): {key} = {value}")

    def get_from_cache(self, key):
        if key in self._gpu_cache:
            print(f"Retrieved from GPU cache: {key}")
            return self._gpu_cache[key]
        elif key in self._host_cache:
            print(f"Retrieved from Host cache (recovered): {key}")
            return self._host_cache[key]
        else:
            print(f"Key not found in cache: {key}")
            return None

    def simulate_gpu_failure(self):
        print("\n--- Simulating GPU failure ---")
        self._gpu_cache = {}
        print("GPU cache cleared.")

    def recover_from_failure(self):
        print("\n--- Initiating recovery from failure ---")
        for key, value in self._host_cache.items():
            self._gpu_cache[key] = value
        print("GPU cache restored from Host cache.")

class MedicalDiagnosticAssistant:
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager

    def load_medical_facts(self):
        print("\n--- Loading Medical Facts ---")
        self.cache_manager.add_to_cache("fever_symptom", "Elevated body temperature", is_critical=True)
        self.cache_manager.add_to_cache("cough_symptom", "Forceful expulsion of air from lungs", is_critical=False)
        self.cache_manager.add_to_cache("diabetes_info", "Chronic metabolic disease characterized by high blood sugar", is_critical=True)
        self.cache_manager.add_to_cache("aspirin_interaction", "Avoid with blood thinners", is_critical=True)
        self.cache_manager.add_to_cache("headache_treatment", "Pain relievers, rest", is_critical=False)

    def diagnose(self, symptom):
        print(f"\n--- Diagnosing symptom: {symptom} ---")
        fact = self.cache_manager.get_from_cache(f"{symptom}_symptom")
        if fact:
            print(f"Found fact for {symptom}: {fact}")
        else:
            print(f"No specific fact found for {symptom}.")

    def get_info(self, key):
        print(f"\n--- Retrieving information for: {key} ---")
        info = self.cache_manager.get_from_cache(key)
        if info:
            print(f"Information for {key}: {info}")
        else:
            print(f"No information found for {key}.")

if __name__ == "__main__":
    print("Starting Medical Diagnostic Assistant Simulation")
    cache_manager = KVCacheManager()
    assistant = MedicalDiagnosticAssistant(cache_manager)

    assistant.load_medical_facts()

    assistant.diagnose("fever")
    assistant.diagnose("cough")
    assistant.get_info("diabetes_info")
    assistant.get_info("unknown_disease")

    cache_manager.simulate_gpu_failure()

    print("\n--- Attempting to access data after GPU failure ---")
    assistant.diagnose("fever")
    assistant.get_info("aspirin_interaction")
    assistant.diagnose("headache")

    cache_manager.recover_from_failure()

    print("\n--- Attempting to access data after recovery ---")
    assistant.diagnose("fever")
    assistant.diagnose("cough")
    assistant.get_info("diabetes_info")
    assistant.get_info("headache_treatment")
