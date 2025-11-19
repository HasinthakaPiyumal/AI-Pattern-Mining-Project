import time

class KVCacheManager:
    def __init__(self):
        self.cache = {}
        print("KV Cache Manager initialized.")

    def get_cache(self, prefix: str):
        """Retrieves KV tensors for a given prefix."""
        if prefix in self.cache:
            print(f"Cache hit for prefix: \"{prefix}\"")
            return self.cache[prefix]
        print(f"Cache miss for prefix: \"{prefix}\"")
        return None

    def set_cache(self, prefix: str, kv_tensors):
        """Stores KV tensors for a given prefix."""
        self.cache[prefix] = kv_tensors
        print(f"Cached KV tensors for prefix: \"{prefix}\"")

class SimplifiedLLM:
    def __init__(self, responses: dict):
        self.responses = responses
        print("Simplified LLM initialized.")

    def _simulate_computation(self, duration: float, task: str):
        """Simulates computational delay."""
        print(f"Simulating {task}... (delay: {duration:.2f}s)")
        time.sleep(duration)
        print(f"{task} completed.")

    def prefill(self, tokens: list):
        """Simulates the prefill phase, generating full KV tensors."""
        self._simulate_computation(len(tokens) * 0.1, "prefill operation") # More tokens, more prefill time
        # In a real LLM, this would generate actual KV tensors
        return {"keys": [f"k_{t}" for t in tokens], "values": [f"v_{t}" for t in tokens]}

    def decode(self, new_token: str, cached_kv):
        """Simulates the decode phase, processing a new token with cached KV tensors."""
        self._simulate_computation(0.05, "decode operation") # Less time per token
        # In a real LLM, this would append to and update KV tensors
        updated_keys = cached_kv["keys"] + [f"k_{new_token}"]
        updated_values = cached_kv["values"] + [f"v_{new_token}"]
        return {"keys": updated_keys, "values": updated_values}

    def generate(self, prompt: str, kv_cache_manager: KVCacheManager):
        """Generates a response using KV Cache Reuse."""
        prompt_tokens = prompt.lower().split()
        
        # Try to find the longest matching prefix in the cache
        matching_prefix = None
        cached_kv = None
        for i in range(len(prompt_tokens), 0, -1):
            prefix = " ".join(prompt_tokens[:i])
            temp_cache = kv_cache_manager.get_cache(prefix)
            if temp_cache:
                matching_prefix = prefix
                cached_kv = temp_cache
                break
        
        generated_response = ""

        if matching_prefix and cached_kv:
            print(f"Reusing KV cache for prefix: \"{matching_prefix}\"")
            prefix_length = len(matching_prefix.split())
            new_tokens = prompt_tokens[prefix_length:]
            current_kv = cached_kv

            # Simulate decoding for new tokens
            for token in new_tokens:
                current_kv = self.decode(token, current_kv)
            print(f"Decoded {len(new_tokens)} new tokens.")

            # Simulate generating a response based on the full prompt
            generated_response = self.responses.get(prompt.lower(), "I'm sorry, I don't understand that request. Can you please rephrase?")

        else:
            print("Performing full prefill for the entire prompt.")
            current_kv = self.prefill(prompt_tokens)
            kv_cache_manager.set_cache(prompt.lower(), current_kv) # Cache the full prompt

            # Simulate generating a response based on the full prompt
            generated_response = self.responses.get(prompt.lower(), "I'm sorry, I don't understand that request. Can you please rephrase?")
            
        return generated_response

# --- Main Application Logic ---
if __name__ == "__main__ commenced":
    # Predefined responses for the chatbot
    chatbot_responses = {
        "what is the status of my order?": "Your order #12345 is currently processing and expected to ship within 2 business days.",
        "i want to return an item": "Please visit our returns page at example.com/returns to initiate a return. You'll need your order number.",
        "how can i track my package?": "You can track your package using the tracking number provided in your shipping confirmation email.",
        "what products do you offer?": "We offer a wide range of products including electronics, apparel, and home goods. Please browse our categories!",
        "what is the status of my order for item x?": "Your order #12345 for item X is awaiting shipment and will be dispatched soon.",
        "what is the status of my order for item y?": "Your order #12345 for item Y has been shipped and should arrive within 3-5 business days.",
        "i want to return an item from order #67890": "For order #67890, please confirm the item(s) you wish to return at example.com/returns.",
    }

    kv_manager = KVCacheManager()
    llm = SimplifiedLLM(chatbot_responses)

    print("\n--- Scenario 1: Initial query (full prefill) ---")
    query1 = "What is the status of my order?"
    start_time = time.perf_counter()
    response1 = llm.generate(query1, kv_manager)
    end_time = time.perf_counter()
    print(f"User: {query1}")
    print(f"Chatbot: {response1}")
    print(f"Time taken: {end_time - start_time:.4f} seconds\n")

    print("\n--- Scenario 2: Repeat query (KV cache reuse) ---")
    query2 = "What is the status of my order?"
    start_time = time.perf_counter()
    response2 = llm.generate(query2, kv_manager)
    end_time = time.perf_counter()
    print(f"User: {query2}")
    print(f"Chatbot: {response2}")
    print(f"Time taken: {end_time - start_time:.4f} seconds\n")

    print("\n--- Scenario 3: Query with common prefix, new suffix (KV cache reuse + decode) ---")
    query3 = "What is the status of my order for item X?"
    start_time = time.perf_counter()
    response3 = llm.generate(query3, kv_manager)
    end_time = time.perf_counter()
    print(f"User: {query3}")
    print(f"Chatbot: {response3}")
    print(f"Time taken: {end_time - start_time:.4f} seconds\n")
    
    print("\n--- Scenario 4: Another query with common prefix, different suffix (KV cache reuse + decode) ---")
    query4 = "What is the status of my order for item Y?"
    start_time = time.perf_counter()
    response4 = llm.generate(query4, kv_manager)
    end_time = time.perf_counter()
    print(f"User: {query4}")
    print(f"Chatbot: {response4}")
    print(f"Time taken: {end_time - start_time:.4f} seconds\n")

    print("\n--- Scenario 5: New, unrelated query (full prefill) ---")
    query5 = "I want to return an item"
    start_time = time.perf_counter()
    response5 = llm.generate(query5, kv_manager)
    end_time = time.perf_counter()
    print(f"User: {query5}")
    print(f"Chatbot: {response5}")
    print(f"Time taken: {end_time - start_time:.4f} seconds\n")

    print("\n--- Scenario 6: Query with a prefix already cached from Scenario 5 ---")
    query6 = "I want to return an item from order #67890"
    start_time = time.perf_counter()
    response6 = llm.generate(query6, kv_manager)
    end_time = time.perf_counter()
    print(f"User: {query6}")
    print(f"Chatbot: {response6}")
    print(f"Time taken: {end_time - start_time:.4f} seconds\n")

    print("\n--- Scenario 7: Unrecognized query ---")
    query7 = "Do you have any discounts today?"
    start_time = time.perf_counter()
    response7 = llm.generate(query7, kv_manager)
    end_time = time.perf_counter()
    print(f"User: {query7}")
    print(f"Chatbot: {response7}")
    print(f"Time taken: {end_time - start_time:.4f} seconds\n")
