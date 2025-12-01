import time
import copy

class LLMCacheManager:
    def __init__(self):
        self.gpu_cache = {}
        self.host_cache = {}
        self.critical_nodes = set() # Stores keys of critical nodes

    def set_kv(self, key, value, is_critical=False):
        self.gpu_cache[key] = value
        if is_critical:
            self.critical_nodes.add(key)
            print(f"[KV Cache Manager] Set critical KV: '{key}' in GPU cache.")
        else:
            print(f"[KV Cache Manager] Set KV: '{key}' in GPU cache.")

    def get_kv(self, key):
        return self.gpu_cache.get(key)

    def replicate_critical_nodes(self):
        print("[KV Cache Manager] Replicating critical nodes to Host cache...")
        for key in self.critical_nodes:
            if key in self.gpu_cache:
                self.host_cache[key] = copy.deepcopy(self.gpu_cache[key])
                print(f"[KV Cache Manager] Replicated '{key}' to Host cache.")
        print("[KV Cache Manager] Replication complete.")

    def simulate_gpu_failure(self):
        print("\n--- SIMULATING GPU FAILURE ---")
        self.gpu_cache = {}
        print("[KV Cache Manager] GPU cache has been cleared due to simulated failure.")

    def recover_from_failure(self):
        print("\n--- INITIATING RECOVERY ---")
        print("[KV Cache Manager] Restoring critical nodes from Host cache...")
        for key in self.critical_nodes:
            if key in self.host_cache:
                self.gpu_cache[key] = copy.deepcopy(self.host_cache[key])
                print(f"[KV Cache Manager] Restored '{key}' to GPU cache.")
        print("[KV Cache Manager] Recovery complete. GPU cache state restored for critical nodes.")

class CustomerSupportChatbot:
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.conversation_history = []

        # Initialize critical system prompt in KV cache
        self.cache_manager.set_kv(
            "system_prompt_v1",
            {"role": "system", "content": "You are an AI assistant for E-commerce customer support. Be helpful and polite."},
            is_critical=True
        )
        self.cache_manager.set_kv(
            "faq_shipping_policy",
            "Our standard shipping takes 3-5 business days. Expedited options are available at checkout.",
            is_critical=True
        )

    def _get_llm_response(self, prompt, kv_state=None):
        # Simulate LLM processing and generating a response
        # In a real system, this would involve calling vllm or similar
        time.sleep(0.5)
        if prompt == "Hello, I need help with an order.":
            return "Hello! How can I assist you with your order today?"
        elif prompt == "What is your shipping policy?":
            shipping_faq = self.cache_manager.get_kv("faq_shipping_policy")
            if shipping_faq:
                return f"Our shipping policy is: {shipping_faq}"
            else:
                return "I need to look up the shipping policy. Please wait a moment."
        elif prompt == "Do you have information on product returns?":
            # This KV might not be critical or replicated, so it could be lost
            returns_info = self.cache_manager.get_kv("faq_returns_policy")
            if returns_info:
                return f"Yes, our returns policy is: {returns_info}"
            else:
                return "I don't have the returns policy readily available right now. Please check our website."
        return "I am sorry, I didn't understand that. Can you please rephrase?"

    def process_customer_query(self, query):
        print(f"\nCustomer: {query}")

        # Example: Try to retrieve system prompt from cache (always critical)
        system_prompt = self.cache_manager.get_kv("system_prompt_v1")
        if not system_prompt:
            print("[Chatbot] WARNING: System prompt not found in GPU cache!")
            # In a real scenario, this would trigger a re-initialization or further recovery logic

        # Simulate LLM interaction, using KV cache for intermediate states
        # Here, we directly use the query as a simplified prompt
        response = self._get_llm_response(query, system_prompt)

        # Simulate storing a new KV if a complex state is generated (not critical by default)
        if "returns" in query.lower() and not self.cache_manager.get_kv("faq_returns_policy"):
            self.cache_manager.set_kv("faq_returns_policy", "You can return items within 30 days for a full refund.")

        self.conversation_history.append((query, response))
        print(f"Chatbot: {response}")
        return response

def run_chatbot_simulation():
    print("### Starting Chatbot Fault Tolerance Simulation ###")

    cache_manager = LLMCacheManager()
    chatbot = CustomerSupportChatbot(cache_manager)

    # --- Initial State & Normal Operation ---
    print("\n--- Phase 1: Normal Operation ---")
    chatbot.process_customer_query("Hello, I need help with an order.")
    chatbot.process_customer_query("What is your shipping policy?")
    chatbot.process_customer_query("Do you have information on product returns?") # Non-critical KV added here
    print(f"Current GPU Cache: {cache_manager.gpu_cache.keys()}")

    # --- Replicate Critical Nodes ---
    cache_manager.replicate_critical_nodes()
    print(f"Current Host Cache: {cache_manager.host_cache.keys()}")

    # --- Simulate GPU Failure ---
    cache_manager.simulate_gpu_failure()
    print(f"GPU Cache after failure: {cache_manager.gpu_cache.keys()}")

    # --- Attempt Query After Failure (Critical nodes missing) ---
    print("\n--- Phase 2: Query after simulated failure (before recovery) ---")
    chatbot.process_customer_query("What is your shipping policy?") # This will likely fail to get from GPU cache
    chatbot.process_customer_query("Do you have information on product returns?") # This was non-critical and will be lost

    # --- Recover from Failure ---
    cache_manager.recover_from_failure()
    print(f"GPU Cache after recovery: {cache_manager.gpu_cache.keys()}")

    # --- Attempt Query After Recovery (Critical nodes restored) ---
    print("\n--- Phase 3: Query after recovery ---")
    chatbot.process_customer_query("Hello, I need help with an order.")
    chatbot.process_customer_query("What is your shipping policy?") # Should now work again
    chatbot.process_customer_query("Do you have information on product returns?") # Will still be lost as it wasn't critical

    print("\n### Simulation Complete ###")

if __name__ == "__main__":
    run_chatbot_simulation()