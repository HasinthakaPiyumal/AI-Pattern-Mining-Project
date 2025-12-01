import time

class GPUSimulatedKVStore:
    """Simulates a fast, volatile GPU memory KV cache."""
    def __init__(self):
        self._cache = {}
        print("GPU KV Cache initialized.")

    def store(self, key, value):
        self._cache[key] = value
        print(f"GPU KV Cache: Stored '{key}'.")

    def retrieve(self, key):
        if key in self._cache:
            print(f"GPU KV Cache: Retrieved '{key}'.")
            return self._cache[key]
        print(f"GPU KV Cache: '{key}' not found.")
        return None

    def simulate_failure(self):
        print("--- Simulating GPU failure: Clearing GPU KV Cache ---")
        self._cache.clear()

class HostSimulatedKVStore:
    """Simulates a slower, persistent Host memory KV cache."""
    def __init__(self):
        self._persistent_cache = {}
        print("Host KV Cache initialized (for replication).")

    def replicate(self, key, value):
        self._persistent_cache[key] = value
        print(f"Host KV Cache: Replicated critical key '{key}'.")

    def retrieve_backup(self, key):
        if key in self._persistent_cache:
            print(f"Host KV Cache: Retrieved backup for '{key}'.")
            return self._persistent_cache[key]
        print(f"Host KV Cache: Backup for '{key}' not found.")
        return None

class MedicalDiagnosticKVManager:
    """
    Manages the multi-level KV cache for the medical diagnostic assistant.
    Implements replication of critical nodes from GPU to Host memory.
    """
    def __init__(self):
        self.gpu_cache = GPUSimulatedKVStore()
        self.host_cache = HostSimulatedKVStore()
        self.critical_keys = set() # Keys designated for replication

    def add_critical_data(self, key, value):
        """Adds data to GPU cache and replicates it to host if critical."""
        self.gpu_cache.store(key, value)
        self.critical_keys.add(key)
        self.host_cache.replicate(key, value)
        print(f"Data for '{key}' added and marked as critical for replication.")

    def add_ephemeral_data(self, key, value):
        """Adds non-critical data only to GPU cache."""
        self.gpu_cache.store(key, value)
        print(f"Data for '{key}' added as ephemeral (GPU-only).")

    def get_data(self, key):
        """
        Retrieves data, prioritizing GPU cache.
        If GPU cache fails and key is critical, attempts retrieval from host backup.
        """
        data = self.gpu_cache.retrieve(key)
        if data is None:
            print(f"Data '{key}' not found in GPU cache. Checking for backup...")
            if key in self.critical_keys:
                data = self.host_cache.retrieve_backup(key)
                if data is not None:
                    print(f"Successfully retrieved critical data '{key}' from host backup.")
                    # Optionally, re-populate GPU cache with recovered data
                    self.gpu_cache.store(key, data)
                else:
                    print(f"Critical data '{key}' not found in host backup either.")
            else:
                print(f"Data '{key}' is not critical and not found in GPU. No backup exists.")
        return data

    def perform_gpu_failure_and_recover(self):
        """Simulates a GPU failure and attempts to recover critical data."""
        self.gpu_cache.simulate_failure()
        print("\n--- Initiating recovery of critical KV nodes ---")
        recovered_count = 0
        # Iterate over a copy of critical_keys to avoid modification issues if self.critical_keys were to be modified during iteration
        for key in list(self.critical_keys):
            backup_data = self.host_cache.retrieve_backup(key)
            if backup_data is not None:
                self.gpu_cache.store(key, backup_data)
                print(f"Recovered critical key '{key}' from host backup to GPU cache.")
                recovered_count += 1
            else:
                print(f"Could not recover critical key '{key}' from host backup (backup missing).")
        print(f"Recovery complete. {recovered_count} critical keys re-populated in GPU cache.")


# --- Example Usage ---
if __name__ == "__main__":
    print("Initializing Medical Diagnostic KV Cache Manager...\n")
    kv_manager = MedicalDiagnosticKVManager()

    # Add critical medical guidelines and patient context
    kv_manager.add_critical_data("sys_prompt_medical_guidelines", "Comprehensive guidelines for diagnosing common respiratory illnesses...")
    kv_manager.add_critical_data("patient_A_summary_history", "Patient A: 65, male, history of hypertension, presenting with cough and fever...")

    # Add ephemeral (non-critical) session-specific data
    kv_manager.add_ephemeral_data("session_A_current_query", "What are potential differential diagnoses for patient A's symptoms?")
    kv_manager.add_ephemeral_data("temp_scratchpad_notes", "Initial thoughts: influenza, pneumonia, bronchitis.")

    print("\n--- Initial state: Retrieving data ---")
    print(f"Retrieving 'sys_prompt_medical_guidelines': {kv_manager.get_data('sys_prompt_medical_guidelines')[:50]}...")
    print(f"Retrieving 'patient_A_summary_history': {kv_manager.get_data('patient_A_summary_history')[:50]}...")
    print(f"Retrieving 'session_A_current_query': {kv_manager.get_data('session_A_current_query')}")
    print(f"Retrieving 'temp_scratchpad_notes': {kv_manager.get_data('temp_scratchpad_notes')}")

    time.sleep(1) # Simulate some processing time

    # Simulate a GPU failure
    print("\n--- Simulating a GPU failure ---\n")
    kv_manager.perform_gpu_failure_and_recover()

    print("\n--- After failure and recovery: Attempting to retrieve data ---")
    # Critical data should be recovered
    print(f"Retrieving 'sys_prompt_medical_guidelines' after recovery: {kv_manager.get_data('sys_prompt_medical_guidelines')[:50]}...")
    print(f"Retrieving 'patient_A_summary_history' after recovery: {kv_manager.get_data('patient_A_summary_history')[:50]}...")

    # Ephemeral data should be lost
    print(f"Retrieving 'session_A_current_query' after recovery: {kv_manager.get_data('session_A_current_query')}")
    print(f"Retrieving 'temp_scratchpad_notes' after recovery: {kv_manager.get_data('temp_scratchpad_notes')}")

    # Try to retrieve a non-existent key
    print(f"Retrieving 'non_existent_key': {kv_manager.get_data('non_existent_key')}")
