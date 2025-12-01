class KVCacheNode:
    def __init__(self, key, value, is_critical=False):
        self.key = key
        self.value = value
        self.is_critical = is_critical
        self.access_frequency = 0

def add_to_cache(cache, key, value, is_critical=False):
    cache[key] = KVCacheNode(key, value, is_critical)

def access_node(cache, key):
    node = cache.get(key)
    if node:
        node.access_frequency += 1
        return node.value
    return None

def replicate_critical_nodes(gpu_cache, host_cache):
    for key, node in gpu_cache.items():
        if node.is_critical:
            host_cache[key] = KVCacheNode(node.key, node.value, node.is_critical) # Deep copy to avoid reference issues

def simulate_gpu_failure(gpu_cache):
    gpu_cache.clear()

def recover_from_host(gpu_cache, host_cache):
    for key, node in host_cache.items():
        gpu_cache[key] = KVCacheNode(node.key, node.value, node.is_critical) # Deep copy

# Application Flow Demonstration
if __name__ == "__main__":
    GPU_KV_Cache = {}
    Host_KV_Cache = {}

    print("--- Initializing Caches ---")
    add_to_cache(GPU_KV_Cache, "patient_id_123", {"name": "Alice", "age": 45}, is_critical=True)
    add_to_cache(GPU_KV_Cache, "diagnosis_alice", "Hypertension", is_critical=True)
    add_to_cache(GPU_KV_Cache, "allergies_alice", ["Penicillin"], is_critical=True)
    add_to_cache(GPU_KV_Cache, "historical_notes_alice_2022", "Routine check-up, no issues.", is_critical=False)
    add_to_cache(GPU_KV_Cache, "emergency_protocol_cardiac", "Administer nitroglycerin", is_critical=True)
    add_to_cache(GPU_KV_Cache, "common_medication_dosage_aspirin", "81mg daily", is_critical=False)

    print("\n--- GPU Cache after initial population ---")
    for key, node in GPU_KV_Cache.items():
        print(f"GPU: {key} -> Value: {node.value}, Critical: {node.is_critical}, Accesses: {node.access_frequency}")

    print("\n--- Accessing some nodes to simulate usage ---")
    access_node(GPU_KV_Cache, "diagnosis_alice")
    access_node(GPU_KV_Cache, "diagnosis_alice")
    access_node(GPU_KV_Cache, "patient_id_123")
    access_node(GPU_KV_Cache, "historical_notes_alice_2022")

    print("\n--- Replicating critical nodes to Host Cache ---")
    replicate_critical_nodes(GPU_KV_Cache, Host_KV_Cache)

    print("\n--- Caches after replication ---")
    print("GPU Cache:")
    for key, node in GPU_KV_Cache.items():
        print(f"  {key} -> Value: {node.value}, Critical: {node.is_critical}, Accesses: {node.access_frequency}")
    print("Host Cache:")
    for key, node in Host_KV_Cache.items():
        print(f"  {key} -> Value: {node.value}, Critical: {node.is_critical}, Accesses: {node.access_frequency}")

    print("\n--- Simulating GPU Failure ---")
    simulate_gpu_failure(GPU_KV_Cache)
    print("GPU Cache cleared.")

    print("\n--- Attempting to access critical node from GPU after failure (should be None) ---")
    alice_diagnosis = access_node(GPU_KV_Cache, "diagnosis_alice")
    print(f"Accessed diagnosis_alice from GPU: {alice_diagnosis}")

    print("\n--- Current state of Caches after GPU failure ---")
    print("GPU Cache (should be empty):")
    for key, node in GPU_KV_Cache.items():
        print(f"  {key} -> Value: {node.value}")
    print("Host Cache (critical nodes should still be there):")
    for key, node in Host_KV_Cache.items():
        print(f"  {key} -> Value: {node.value}, Critical: {node.is_critical}, Accesses: {node.access_frequency}")

    print("\n--- Recovering from Host Cache ---")
    recover_from_host(GPU_KV_Cache, Host_KV_Cache)

    print("\n--- GPU Cache after recovery (critical nodes restored) ---")
    for key, node in GPU_KV_Cache.items():
        print(f"GPU: {key} -> Value: {node.value}, Critical: {node.is_critical}, Accesses: {node.access_frequency}")

    print("\n--- Attempting to access critical node from GPU after recovery (should succeed) ---")
    alice_diagnosis_recovered = access_node(GPU_KV_Cache, "diagnosis_alice")
    print(f"Accessed diagnosis_alice from GPU: {alice_diagnosis_recovered}")
    alice_patient_id_recovered = access_node(GPU_KV_Cache, "patient_id_123")
    print(f"Accessed patient_id_123 from GPU: {alice_patient_id_recovered}")

    print("\n--- Attempting to access non-critical node from GPU after recovery (should still be None as it wasn't critical) ---")
    historical_notes_recovered = access_node(GPU_KV_Cache, "historical_notes_alice_2022")
    print(f"Accessed historical_notes_alice_2022 from GPU: {historical_notes_recovered}")
