import logging
from fastapi import FastAPI
import uvicorn
import time
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

GPU_KV_CACHE = {}
HOST_KV_CACHE = {}

CRITICAL_KV_KEYS = ["system_prompt_kv", "common_greeting_kv", "product_info_prefix_kv"]

class LLMService:
    def __init__(self):
        logging.info("LLM Service initialized.")

    def _generate_simulated_kv(self, key_prefix, data):
        kv_data = {
            "key": f"{key_prefix}_key_{hash(data) % 1000}",
            "value": f"{key_prefix}_value_{len(data)}_{random.randint(1, 100)}",
            "timestamp": time.time()
        }
        return kv_data

    def process_query(self, query: str):
        logging.info(f"Processing query: '{query}'")
        new_kv_key = f"query_kv_{hash(query) % 1000}"
        GPU_KV_CACHE[new_kv_key] = self._generate_simulated_kv(new_kv_key, query)

        if "hello" in query.lower():
            GPU_KV_CACHE["common_greeting_kv"] = self._generate_simulated_kv("common_greeting_kv", "hello context")
        if "product" in query.lower():
            GPU_KV_CACHE["product_info_prefix_kv"] = self._generate_simulated_kv("product_info_prefix_kv", "product details")

        logging.info(f"GPU KV Cache updated. Current size: {len(GPU_KV_CACHE)}")
        return {"response": f"Simulated LLM response for: '{query}'", "kv_generated": new_kv_key}

class KVReplicationManager:
    def __init__(self):
        logging.info("KV Replication Manager initialized.")
        GPU_KV_CACHE["system_prompt_kv"] = {"key": "sys_p_k", "value": "sys_p_v", "timestamp": time.time()}
        GPU_KV_CACHE["common_greeting_kv"] = {"key": "g_k", "value": "g_v", "timestamp": time.time()}
        logging.info(f"Initial critical KV nodes in GPU cache: {list(GPU_KV_CACHE.keys())}")

    def replicate_critical_nodes(self):
        logging.info("Starting replication of critical KV cache nodes to Host Memory.")
        replicated_count = 0
        for key in CRITICAL_KV_KEYS:
            if key in GPU_KV_CACHE:
                HOST_KV_CACHE[key] = GPU_KV_CACHE[key]
                replicated_count += 1
                logging.debug(f"Replicated '{key}' to Host Memory.")
        logging.info(f"Finished replication. {replicated_count} critical nodes replicated to Host Memory. Host KV Cache size: {len(HOST_KV_CACHE)}")

    def simulate_gpu_failure(self):
        logging.warning("Simulating GPU failure: Clearing GPU KV Cache!")
        global GPU_KV_CACHE
        GPU_KV_CACHE = {}
        logging.info(f"GPU KV Cache cleared. Current size: {len(GPU_KV_CACHE)}")

    def recover_from_failure(self):
        logging.info("Starting recovery from failure: Restoring critical KV cache nodes from Host Memory.")
        recovered_count = 0
        for key in CRITICAL_KV_KEYS:
            if key in HOST_KV_CACHE:
                GPU_KV_CACHE[key] = HOST_KV_CACHE[key]
                recovered_count += 1
                logging.debug(f"Recovered '{key}' to GPU Memory.")
            else:
                logging.warning(f"Critical key '{key}' not found in Host Memory backup during recovery.")
        logging.info(f"Finished recovery. {recovered_count} critical nodes restored to GPU Memory. GPU KV Cache size: {len(GPU_KV_CACHE)}")

llm_service = LLMService()
kv_manager = KVReplicationManager()

@app.post("/query")
async def process_customer_query(query: str):
    llm_response = llm_service.process_query(query)

    kv_manager.replicate_critical_nodes()

    if random.random() < 0.1:
        logging.error("RANDOM GPU FAILURE DETECTED!")
        kv_manager.simulate_gpu_failure()
        kv_manager.recover_from_failure()
        return {"status": "recovered_after_failure", "llm_response": llm_response["response"], "gpu_cache_after_recovery_size": len(GPU_KV_CACHE)}

    return {"status": "success", "llm_response": llm_response["response"], "gpu_cache_size": len(GPU_KV_CACHE)}

@app.get("/status")
async def get_system_status():
    return {
        "gpu_kv_cache_size": len(GPU_KV_CACHE),
        "host_kv_cache_size": len(HOST_KV_CACHE),
        "critical_keys_in_gpu": [k for k in CRITICAL_KV_KEYS if k in GPU_KV_CACHE],
        "critical_keys_in_host": [k for k in CRITICAL_KV_KEYS if k in HOST_KV_CACHE]
    }

@app.post("/simulate_failure_and_recover")
async def simulate_failure_and_recover_endpoint():
    kv_manager.simulate_gpu_failure()
    kv_manager.recover_from_failure()
    return {"status": "failure_simulated_and_recovered", "gpu_cache_size": len(GPU_KV_CACHE)}

if __name__ == "__main__":
    logging.info("Starting FastAPI application...")
    uvicorn.run(app, host="0.0.0.0", port=8000)