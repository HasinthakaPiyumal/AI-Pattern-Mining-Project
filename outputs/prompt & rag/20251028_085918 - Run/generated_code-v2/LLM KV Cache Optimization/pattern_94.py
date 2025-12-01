import random
import time

class KVNode:
    def __init__(self, key, value, is_critical=False):
        self.key = key
        self.value = value
        self.is_critical = is_critical

class GPUCache:
    def __init__(self, capacity=10):
        self.cache = {}
        self.capacity = capacity
        self.lru_order = []

    def put(self, key, node):
        if key in self.cache:
            self.lru_order.remove(key)
        elif len(self.cache) >= self.capacity:
            self._evict_lru()
        self.cache[key] = node
        self.lru_order.append(key)

    def get(self, key):
        if key in self.cache:
            self.lru_order.remove(key)
            self.lru_order.append(key)
            return self.cache[key].value
        return None

    def _evict_lru(self):
        if self.lru_order:
            lru_key = self.lru_order.pop(0)
            del self.cache[lru_key]

    def clear(self):
        self.cache = {}
        self.lru_order = []

    def __len__(self):
        return len(self.cache)


class HostCache:
    def __init__(self):
        self.cache = {}

    def put(self, key, node):
        self.cache[key] = node

    def get(self, key):
        if key in self.cache:
            return self.cache[key].value
        return None

    def __len__(self):
        return len(self.cache)


class CacheManager:
    def __init__(self, gpu_capacity=10):
        self.gpu_cache = GPUCache(capacity=gpu_capacity)
        self.host_cache = HostCache()

    def add_to_cache(self, key, value, is_critical=False):
        node = KVNode(key, value, is_critical)
        self.gpu_cache.put(key, node)
        if is_critical:
            self.host_cache.put(key, node)

    def get_from_cache(self, key):
        value = self.gpu_cache.get(key)
        if value is None:
            value = self.host_cache.get(key)
            if value is not None:
                # If found in host, put back to GPU (simulating recovery/promotion)
                self.gpu_cache.put(key, KVNode(key, value, True))
        return value

    def replicate_critical_node(self, key, value):
        node = KVNode(key, value, True)
        self.host_cache.put(key, node)

    def simulate_gpu_failure(self):
        print("\n--- Simulating GPU Failure ---")
        self.gpu_cache.clear()
        print("GPU Cache cleared.")

    def recover_from_failure(self):
        print("\n--- Initiating Recovery from Host Cache ---")
        recovered_count = 0
        for key, node in self.host_cache.cache.items():
            if node.is_critical:
                self.gpu_cache.put(key, node)
                recovered_count += 1
        print(f"Recovered {recovered_count} critical nodes to GPU Cache.")


class AICustomerSupportChatbot:
    def __init__(self):
        self.cache_manager = CacheManager(gpu_capacity=5)
        self.conversation_history = []
        self._initialize_system_prompt()

    def _initialize_system_prompt(self):
        system_prompt = "You are an e-commerce customer support assistant. Be helpful and polite."
        self.cache_manager.add_to_cache("system_prompt_kv", system_prompt, is_critical=True)
        print(f"System prompt added to cache (critical): {system_prompt[:50]}...")

    def _generate_embedding(self, text):
        # Simulate embedding generation
        return f"embedding_for_{text.replace(' ', '_')}_{random.randint(0, 999)}"

    def _llm_inference(self, prompt, context_embeddings):
        # Simulate LLM inference
        time.sleep(0.1) # Simulate some processing time
        response_templates = [
            "I understand you're looking for {product}. Let me check.",
            "Regarding your question about {topic}, here's what I found.",
            "I can help you with that. Could you provide more details about {query_part}?",
            "Our policy on {policy_topic} states..."
        ]
        selected_template = random.choice(response_templates)
        return selected_template.format(product="a product", topic="returns", query_part="your order", policy_topic="shipping") + f" (context: {len(context_embeddings)} embeddings)"

    def chat(self, user_input):
        print(f"\nUser: {user_input}")

        # Retrieve system prompt (critical node)
        system_prompt = self.cache_manager.get_from_cache("system_prompt_kv")
        if system_prompt:
            print(f"Retrieved system prompt from cache: {system_prompt[:50]}...")
        else:
            print("Error: System prompt not found in cache!")
            return "I am sorry, I am experiencing a temporary issue. Please try again later."

        # Generate embedding for user input
        user_embedding_key = f"user_query_emb_{len(self.conversation_history)}"
        user_embedding_value = self._generate_embedding(user_input)
        self.cache_manager.add_to_cache(user_embedding_key, user_embedding_value, is_critical=False)
        print(f"User query embedding added to GPU cache: {user_embedding_key}")

        # Add a critical conversation turn embedding (e.g., after a key decision or clarification)
        if "order status" in user_input.lower():
            critical_turn_key = f"critical_conv_emb_{len(self.conversation_history)}"
            critical_turn_value = self._generate_embedding(f"Order status query for '{user_input}'")
            self.cache_manager.add_to_cache(critical_turn_key, critical_turn_value, is_critical=True)
            print(f"Critical conversation turn embedding replicated: {critical_turn_key}")

        # Simulate RAG context retrieval (using cache)
        context_embeddings = []
        for i in range(len(self.conversation_history), -1, -1):
            key = f"user_query_emb_{i}"
            embedding = self.cache_manager.get_from_cache(key)
            if embedding:
                context_embeddings.append(embedding)
            # Also try to get critical turn embeddings if they exist
            critical_key = f"critical_conv_emb_{i}"
            critical_embedding = self.cache_manager.get_from_cache(critical_key)
            if critical_embedding:
                context_embeddings.append(critical_embedding)

        # Simulate LLM inference
        llm_response = self._llm_inference(system_prompt + " " + user_input, context_embeddings)
        self.conversation_history.append((user_input, llm_response))

        print(f"Bot: {llm_response}")
        return llm_response



if __name__ == "__main__":
    print("Initializing AI Customer Support Chatbot...")
    chatbot = AICustomerSupportChatbot()

    print("\n--- Initial Chat Interactions ---")
    chatbot.chat("Hi, I'd like to check the status of my order.")
    chatbot.chat("My order number is #12345.")
    chatbot.chat("What is your return policy?")

    print("\nGPU Cache size:", len(chatbot.cache_manager.gpu_cache))
    print("Host Cache size:", len(chatbot.cache_manager.host_cache))

    # Simulate a GPU failure
    chatbot.cache_manager.simulate_gpu_failure()

    # Try to chat again immediately after failure
    print("\n--- Chat Interaction after GPU Failure (before recovery) ---")
    chatbot.chat("Can you tell me about product XYZ?") # Should recover system prompt, but lose non-critical context

    # Recover critical nodes
    chatbot.cache_manager.recover_from_failure()

    # Chat again after recovery
    print("\n--- Chat Interaction after Recovery ---")
    chatbot.chat("What are the shipping options for product XYZ?")

    print("\nFinal GPU Cache size:", len(chatbot.cache_manager.gpu_cache))
    print("Final Host Cache size:", len(chatbot.cache_manager.host_cache))
