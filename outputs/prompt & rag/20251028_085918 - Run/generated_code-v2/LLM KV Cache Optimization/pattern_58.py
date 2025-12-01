gpu_kv_cache = {}
host_kv_cache = {}

def replicate_critical_nodes(node_keys):
    for key in node_keys:
        if key in gpu_kv_cache:
            host_kv_cache[key] = gpu_kv_cache[key]

def simulate_gpu_failure():
    global gpu_kv_cache
    gpu_kv_cache = {}

def recover_from_host(node_keys):
    for key in node_keys:
        if key in host_kv_cache:
            gpu_kv_cache[key] = host_kv_cache[key]

def initialize_chatbot_context():
    gpu_kv_cache["system_prompt"] = "You are a helpful customer support assistant."
    gpu_kv_cache["greeting"] = "Hello! How can I assist you today?"
    gpu_kv_cache["product_info_faq_1"] = "Our premium plan includes unlimited support and advanced features."

def get_response(query):
    if "hello" in query.lower() or "hi" in query.lower():
        return gpu_kv_cache.get("greeting", "Hello there!")
    elif "plan" in query.lower() or "features" in query.lower():
        return gpu_kv_cache.get("product_info_faq_1", "Please tell me more about what you're looking for.")
    elif "system" in query.lower():
        return gpu_kv_cache.get("system_prompt", "I am an AI assistant.")
    return "I'm sorry, I don't have information on that. Can I help with anything else?"

if __name__ == "__main__":
    print("--- Chatbot Initialization ---")
    initialize_chatbot_context()
    critical_nodes = ["system_prompt", "greeting", "product_info_faq_1"]
    print(f"Initial GPU KV Cache: {gpu_kv_cache}")

    print("\n--- Replicating Critical Nodes to Host Memory ---")
    replicate_critical_nodes(critical_nodes)
    print(f"Host KV Cache after replication: {host_kv_cache}")

    print("\n--- Chatbot in Normal Operation ---")
    print(f"User: Hi there!")
    print(f"Chatbot: {get_response('Hi there!')}")
    print(f"User: What about your premium plan?")
    print(f"Chatbot: {get_response('What about your premium plan?')}")

    print("\n--- Simulating GPU Failure ---")
    simulate_gpu_failure()
    print(f"GPU KV Cache after simulated failure: {gpu_kv_cache}")
    print("Attempting to get response after failure (should fail for critical data):")
    print(f"User: Hi there!")
    print(f"Chatbot: {get_response('Hi there!')}")

    print("\n--- Recovering Critical Nodes from Host Memory ---")
    recover_from_host(critical_nodes)
    print(f"GPU KV Cache after recovery: {gpu_kv_cache}")

    print("\n--- Chatbot Operating After Recovery ---")
    print(f"User: Hi there again!")
    print(f"Chatbot: {get_response('Hi there again!')}")
    print(f"User: And the premium plan details?")
    print(f"Chatbot: {get_response('And the premium plan details?')}")