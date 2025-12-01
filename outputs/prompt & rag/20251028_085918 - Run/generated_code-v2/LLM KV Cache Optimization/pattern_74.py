from fastapi import FastAPI
import uvicorn
import time

class KVNode:
    def __init__(self, key, value, is_critical=False):
        self.key = key
        self.value = value
        self.is_critical = is_critical
        self.timestamp = time.time()

class CacheManager:
    def __init__(self):
        self.gpu_cache = {}
        self.host_cache = {}
        self.gpu_failure_active = False

    def add_to_cache(self, key, value, is_critical=False):
        node = KVNode(key, value, is_critical)
        self.gpu_cache[key] = node
        if is_critical:
            self.host_cache[key] = node # Replicate immediately if critical
        print(f"Added/Updated KV Cache (GPU): {key}, Critical: {is_critical}")

    def get_from_cache(self, key):
        if not self.gpu_failure_active and key in self.gpu_cache:
            print(f"Retrieved from GPU Cache: {key}")
            return self.gpu_cache[key].value
        elif key in self.host_cache:
            print(f"Retrieved from Host Cache (GPU failure active or not found in GPU): {key}")
            return self.host_cache[key].value
        print(f"Key not found in any cache: {key}")
        return None

    def mark_critical(self, key):
        if key in self.gpu_cache:
            self.gpu_cache[key].is_critical = True
            self.host_cache[key] = self.gpu_cache[key] # Ensure replication if marked critical later
            print(f"Marked KV Cache node as critical: {key}")
        else:
            print(f"Key {key} not found in GPU cache to mark as critical.")

    def replicate_critical_nodes(self):
        for key, node in self.gpu_cache.items():
            if node.is_critical:
                self.host_cache[key] = node
        print("Replicated critical nodes from GPU to Host cache.")

    def simulate_gpu_failure(self):
        print("!!! Simulating GPU failure: Clearing GPU Cache !!!")
        self.gpu_cache.clear()
        self.gpu_failure_active = True

    def recover_from_failure(self):
        print("Initiating recovery from GPU failure...")
        if self.gpu_failure_active:
            self.gpu_cache.update(self.host_cache) # Restore critical nodes
            self.gpu_failure_active = False
            print("Recovery complete: Critical nodes restored to GPU cache.")
        else:
            print("No GPU failure active to recover from.")

    def get_cache_status(self):
        return {
            "gpu_cache_size": len(self.gpu_cache),
            "host_cache_size": len(self.host_cache),
            "gpu_failure_active": self.gpu_failure_active,
            "gpu_cache_keys": list(self.gpu_cache.keys()),
            "host_cache_keys": list(self.host_cache.keys())
        }

class LLMCore:
    def process_query(self, query, patient_context=None):
        print(f"LLM processing query: '{query}' with context: {patient_context}")
        # Simulate LLM's complex reasoning and intermediate state generation
        if "diagnosis for headache" in query.lower():
            return {
                "diagnosis_hypothesis": "Migraine vs. Tension Headache",
                "reasoning_steps_embedding": "[embedding_data_headache_123]",
                "recommended_tests": ["MRI", "Blood Test"],
                "intermediate_state_key": "headache_diag_state_001"
            }
        elif "patient history summary" in query.lower():
            return {
                "summary": "Patient presents with chronic fatigue and intermittent fever.",
                "intermediate_state_key": "patient_summary_state_002"
            }
        else:
            return {
                "response": f"LLM response for '{query}'",
                "intermediate_state_key": f"generic_state_{hash(query)}"
            }

class DiagnosticService:
    def __init__(self):
        self.cache_manager = CacheManager()
        self.llm_core = LLMCore()
        self._load_initial_critical_knowledge()

    def _load_initial_critical_knowledge(self):
        # Load initial critical medical guidelines
        self.cache_manager.add_to_cache("medical_guidelines_hypertension", "Comprehensive guidelines for hypertension management.", is_critical=True)
        self.cache_manager.add_to_cache("medical_guidelines_diabetes", "Latest protocols for diabetes care.", is_critical=True)
        print("Loaded initial critical medical knowledge into cache.")

    def diagnose_patient(self, patient_data: dict):
        patient_id = patient_data.get("patient_id", "unknown_patient")
        main_symptom = patient_data.get("main_symptom", "")
        query = f"Provide a diagnosis for {main_symptom} based on patient ID {patient_id}."

        # 1. Store patient context in cache (can be critical)
        patient_context_key = f"patient_context_{patient_id}"
        self.cache_manager.add_to_cache(patient_context_key, patient_data, is_critical=True)

        # 2. Retrieve critical medical guidelines (already in cache from init)
        hypertension_guidelines = self.cache_manager.get_from_cache("medical_guidelines_hypertension")
        diabetes_guidelines = self.cache_manager.get_from_cache("medical_guidelines_diabetes")

        # 3. LLM processes query with patient context and guidelines
        llm_input_context = {
            "patient_data": patient_data,
            "hypertension_guidelines": hypertension_guidelines,
            "diabetes_guidelines": diabetes_guidelines
        }
        llm_output = self.llm_core.process_query(query, llm_input_context)

        # 4. Store LLM's intermediate states/reasoning in cache (potentially critical)
        if "intermediate_state_key" in llm_output:
            intermediate_key = llm_output["intermediate_state_key"]
            self.cache_manager.add_to_cache(intermediate_key, llm_output, is_critical=True)
            # Explicitly mark if needed, though add_to_cache already handles is_critical=True
            self.cache_manager.mark_critical(intermediate_key)

        # 5. Periodically replicate critical nodes (or after specific steps)
        self.cache_manager.replicate_critical_nodes()

        return {"diagnosis_result": llm_output, "patient_id": patient_id}

app = FastAPI()
dx_service = DiagnosticService()

@app.post("/diagnose")
async def diagnose(patient_data: dict):
    print(f"API: Received diagnosis request for patient: {patient_data.get('patient_id')}")
    result = dx_service.diagnose_patient(patient_data)
    return result

@app.post("/simulate_gpu_failure")
async def simulate_failure():
    dx_service.cache_manager.simulate_gpu_failure()
    return {"status": "GPU failure simulated. GPU cache cleared."}

@app.post("/recover_from_failure")
async def recover_failure():
    dx_service.cache_manager.recover_from_failure()
    return {"status": "Recovery attempt initiated."}

@app.get("/cache_status")
async def get_status():
    return dx_service.cache_manager.get_cache_status()

@app.post("/replicate_cache")
async def replicate_cache_nodes():
    dx_service.cache_manager.replicate_critical_nodes()
    return {"status": "Critical nodes replicated to host cache."}

if __name__ == "__main__":
    # To run this, save it as main.py and then run: uvicorn main:app --reload
    print("Starting Medical Diagnosis Assistant API...\n")
    print("Example usage:")
    print("  POST /diagnose with body: {'patient_id': 'P001', 'main_symptom': 'severe headache', 'age': 45}")
    print("  POST /simulate_gpu_failure")
    print("  GET /cache_status")
    print("  POST /recover_from_failure")
    print("  POST /replicate_cache\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)