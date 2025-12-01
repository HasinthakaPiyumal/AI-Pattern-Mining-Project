from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

class KVCacheManager:
    def __init__(self):
        self.gpu_cache = {}
        self.host_cache = {}
        self.critical_keys = set()

    def update_cache(self, key: str, value: str, is_critical: bool = False):
        self.gpu_cache[key] = value
        if is_critical or key in self.critical_keys:
            self.host_cache[key] = value
            self.critical_keys.add(key)

    def get_cache(self, key: str):
        return self.gpu_cache.get(key)

    def simulate_gpu_failure(self):
        self.gpu_cache = {}

    def recover_from_host(self, key: str):
        if key in self.critical_keys and key in self.host_cache:
            self.gpu_cache[key] = self.host_cache[key]
            return True
        return False

    def add_critical_key(self, key: str):
        self.critical_keys.add(key)

class DiagnosisRequest(BaseModel):
    symptoms: str

class RecoveryRequest(BaseModel):
    key: str

class CacheStatusResponse(BaseModel):
    gpu_cache: dict
    host_cache: dict
    critical_keys: list

app = FastAPI()
k_v_cache_manager = KVCacheManager()

@app.post("/diagnose")
async def diagnose(request: DiagnosisRequest):
    # Simulate LLM processing and KV cache updates
    patient_id = "patient_123"
    k_v_cache_manager.update_cache(f"system_prompt_kv", "General medical knowledge base...", is_critical=True)
    k_v_cache_manager.update_cache(f"{patient_id}_symptoms_kv", request.symptoms, is_critical=True)
    k_v_cache_manager.update_cache(f"{patient_id}_context_step_1", "Initial symptom analysis...")

    simulated_diagnosis = f"Based on symptoms: {request.symptoms}, potential diagnosis: Fever. Further tests recommended."
    return {"diagnosis": simulated_diagnosis}

@app.post("/simulate_failure")
async def simulate_failure():
    k_v_cache_manager.simulate_gpu_failure()
    return {"message": "GPU cache simulated as failed (cleared)."}

@app.post("/recover_critical_cache")
async def recover_critical_cache(request: RecoveryRequest):
    if k_v_cache_manager.recover_from_host(request.key):
        return {"message": f"Critical KV cache for key '{request.key}' recovered from host memory."}
    return {"message": f"Failed to recover critical KV cache for key '{request.key}'. Key not found or not critical.", "status": "failed"}

@app.get("/get_cache_status", response_model=CacheStatusResponse)
async def get_cache_status():
    return CacheStatusResponse(
        gpu_cache=k_v_cache_manager.gpu_cache,
        host_cache=k_v_cache_manager.host_cache,
        critical_keys=list(k_v_cache_manager.critical_keys)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)