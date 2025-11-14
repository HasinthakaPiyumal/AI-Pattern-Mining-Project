import time

# 1. KV Cache Management Module
KV_CACHE = {}

def store_kv_tensors(prefix: str, kv_tensors: str):
    """Stores simulated KV tensors for a given prefix."""
    KV_CACHE[prefix] = kv_tensors
    print(f"[KV_CACHE] Stored KV tensors for prefix: '{prefix}'")

def retrieve_kv_tensors(prefix: str) -> str or None:
    """Retrieves simulated KV tensors for a given prefix if available."""
    if prefix in KV_CACHE:
        print(f"[KV_CACHE] Retrieved KV tensors for prefix: '{prefix}' (REUSE)")
        return KV_CACHE[prefix]
    print(f"[KV_CACHE] KV tensors not found for prefix: '{prefix}'")
    return None

# 2. Simulated LLM Inference Engine
def _simulate_llm_inference(prompt: str, prefix_kv_tensors: str or None = None) -> (str, str):
    """Simulates LLM inference, indicating whether KV tensors were reused or computed.
    Returns a simulated response and the newly computed/reused KV tensors (for the entire prompt).
    """
    simulated_kv_tensors = f"KV_TENSORS_FOR_{hash(prompt)}"

    if prefix_kv_tensors:
        print(f"[LLM_SIMULATION] Starting inference from cached KV tensors for prefix. Prompt: '{prompt[:min(len(prompt), 30)]}...' (REUSE MODE)")
        time.sleep(0.5) # Simulate faster inference
        response = f"(Cached prefix used) Here is the answer to: {prompt}"
    else:
        print(f"[LLM_SIMULATION] Computing KV tensors for full prompt. Prompt: '{prompt[:min(len(prompt), 30)]}...' (COMPUTE MODE)")
        time.sleep(1.5) # Simulate slower inference
        response = f"(New computation) This is the answer to: {prompt}"

    print(f"[LLM_SIMULATION] Inference complete for: '{prompt[:min(len(prompt), 30)]}...'\n")
    return response, simulated_kv_tensors

# 3. Chatbot Interaction Logic
def chatbot_response(user_query: str) -> str:
    """Generates a chatbot response, leveraging KV cache for prefixes."""
    print(f"[CHATBOT] User: {user_query}")

    # Simple prefix detection (can be more sophisticated in a real system)
    known_prefixes = [
        "What is the status of my order",
        "What is your return policy",
        "How do I change my shipping address"
    ]

    found_prefix = None
    for prefix in known_prefixes:
        if user_query.lower().startswith(prefix.lower()):
            found_prefix = prefix
            break

    prefix_kv_tensors = None
    if found_prefix:
        prefix_kv_tensors = retrieve_kv_tensors(found_prefix)

    # Simulate LLM inference
    response, current_kv_tensors = _simulate_llm_inference(user_query, prefix_kv_tensors)

    # Store KV tensors for the detected prefix if not already cached
    if found_prefix and not prefix_kv_tensors:
        # In a real scenario, you'd extract the KV tensors corresponding to the prefix
        # from `current_kv_tensors`. For this simulation, we'll store a placeholder
        # derived from the full query's KV tensors, implying a successful prefill.
        store_kv_tensors(found_prefix, f"KV_TENSORS_FOR_PREFIX_{hash(found_prefix)}")

    return response

# --- Example Usage ---
if __name__ == "__main__":
    print("\n--- Starting Chatbot Simulation ---")

    # First query - prefix not in cache, full computation
    resp1 = chatbot_response("What is the status of my order for product XYZ-123?")
    print(f"[CHATBOT] Bot: {resp1}\n")

    # Second query - same prefix, should reuse KV cache
    resp2 = chatbot_response("What is the status of my order for product ABC-456?")
    print(f"[CHATBOT] Bot: {resp2}\n")

    # Third query - different prefix, full computation
    resp3 = chatbot_response("What is your return policy for electronics?")
    print(f"[CHATBOT] Bot: {resp3}\n")

    # Fourth query - same prefix as third, should reuse KV cache
    resp4 = chatbot_response("What is your return policy if the item is damaged?")
    print(f"[CHATBOT] Bot: {resp4}\n")

    # Fifth query - completely new query, full computation
    resp5 = chatbot_response("Can I get a discount code?")
    print(f"[CHATBOT] Bot: {resp5}\n")

    print("--- Chatbot Simulation Finished ---")