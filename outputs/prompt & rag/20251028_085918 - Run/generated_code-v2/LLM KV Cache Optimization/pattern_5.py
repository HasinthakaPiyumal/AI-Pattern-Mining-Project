import time
from collections import deque

class MockLLM:
    """A mock LLM to simulate prefill and generation with KV cache concepts."""
    def __init__(self, processing_speed=0.05):
        self.processing_speed = processing_speed # Time per token/step

    def _simulate_heavy_prefill(self, prompt_tokens):
        """Simulates the costly prefill phase for a new prompt."""
        time.sleep(len(prompt_tokens) * self.processing_speed * 2) # Prefill is more expensive
        # In a real LLM, this would generate initial KV tensors
        kv_state = f"KV_STATE_FOR:{'_'.join(prompt_tokens)}"
        initial_response_tokens = [f"LLM_RESP_P1_{token}" for token in prompt_tokens[:min(3, len(prompt_tokens))]]
        return kv_state, initial_response_tokens

    def _simulate_light_generation(self, kv_state, new_input_tokens, max_new_tokens=10):
        """Simulates generating new tokens using an existing KV state."""
        time.sleep(len(new_input_tokens) * self.processing_speed) # Less expensive than prefill
        generated_tokens = [f"LLM_RESP_G_{i}" for i in range(max_new_tokens)]
        new_kv_state = f"{kv_state}_UPDATED_WITH_{'_'.join(new_input_tokens)}"
        return new_kv_state, generated_tokens

    def generate(self, prompt: str, cached_kv_state=None):
        """Generates a response, potentially using a cached KV state."""
        prompt_tokens = prompt.lower().split()
        full_response_tokens = []

        if cached_kv_state:
            print(f"  [LLM] Generating from cached KV state: {cached_kv_state[:30]}...")
            new_kv_state, generated_tokens = self._simulate_light_generation(cached_kv_state, prompt_tokens)
            full_response_tokens.extend(generated_tokens)
            # In a real system, the prompt_tokens would be used to extend the context
            # from the cached_kv_state, and then generate further.
            print(f"  [LLM] Generated {len(generated_tokens)} tokens from cache.")
        else:
            print(f"  [LLM] Performing full prefill for new prompt: '{prompt[:30]}...' (expensive)")
            kv_state, initial_response_tokens = self._simulate_heavy_prefill(prompt_tokens)
            new_kv_state, generated_tokens = self._simulate_light_generation(kv_state, prompt_tokens[min(3, len(prompt_tokens)):])
            full_response_tokens.extend(initial_response_tokens)
            full_response_tokens.extend(generated_tokens)
            print(f"  [LLM] Completed full prefill and generated {len(initial_response_tokens) + len(generated_tokens)} tokens.")

        return new_kv_state, " ".join(full_response_tokens)


class KVCache:
    """Manages storage and retrieval of KV states for prefixes."""
    def __init__(self, max_size=100):
        self.cache = {}
        self.lru_queue = deque()
        self.max_size = max_size

    def _evict_oldest(self):
        if len(self.lru_queue) > self.max_size:
            oldest_prefix = self.lru_queue.popleft()
            if oldest_prefix in self.cache:
                del self.cache[oldest_prefix]
                print(f"  [KVCache] Evicted oldest prefix: '{oldest_prefix}'")

    def get(self, prefix: str):
        if prefix in self.cache:
            # Move to end to mark as recently used
            if prefix in self.lru_queue:
                self.lru_queue.remove(prefix)
            self.lru_queue.append(prefix)
            print(f"  [KVCache] Cache HIT for prefix: '{prefix[:30]}...' - Loading KV state.")
            return self.cache[prefix]
        print(f"  [KVCache] Cache MISS for prefix: '{prefix[:30]}...' - No KV state found.")
        return None

    def put(self, prefix: str, kv_state):
        if prefix not in self.cache:
            self._evict_oldest()
        self.cache[prefix] = kv_state
        if prefix in self.lru_queue:
            self.lru_queue.remove(prefix)
        self.lru_queue.append(prefix)
        print(f"  [KVCache] Stored KV state for prefix: '{prefix[:30]}...'")


class EcommerceChatbot:
    """An AI-powered customer support chatbot leveraging KV Cache Reuse."""
    def __init__(self, llm: MockLLM, kv_cache: KVCache, conversation_depth=5):
        self.llm = llm
        self.kv_cache = kv_cache
        self.conversation_history = deque(maxlen=conversation_depth)
        self.current_kv_state = None

    def _get_conversation_prefix(self, user_input: str) -> str:
        history_str = " ".join(self.conversation_history)
        if history_str:
            return f"{history_str} [SEP] {user_input}"
        return user_input

    def chat(self, user_query: str):
        print(f"\nUser: {user_query}")
        start_time = time.time()

        # 1. Determine the effective prefix for caching
        effective_prefix = self._get_conversation_prefix(user_query)

        # 2. Check KV Cache for the prefix
        cached_kv_state = self.kv_cache.get(effective_prefix)

        response_kv_state = None
        response_text = ""

        if cached_kv_state:
            # Cache hit: Use the loaded KV state to continue generation
            response_kv_state, response_text = self.llm.generate(user_query, cached_kv_state=cached_kv_state)
        else:
            # Cache miss: Perform a full LLM inference (prefill + generate)
            # and potentially cache the new KV state.
            response_kv_state, response_text = self.llm.generate(user_query)
            self.kv_cache.put(effective_prefix, response_kv_state) # Cache the new state

        end_time = time.time()
        print(f"Chatbot: {response_text}")
        print(f"  (Response took {end_time - start_time:.4f} seconds)")

        # Update conversation history for next turn
        self.conversation_history.append(user_query)
        # For multi-turn, we might update the current_kv_state for the *entire* conversation
        self.current_kv_state = response_kv_state
        return response_text

# --- Demonstration --- 
if __name__ == "__main__":
    mock_llm = MockLLM(processing_speed=0.01) # Faster for demo
    kv_cache = KVCache(max_size=5)
    chatbot = EcommerceChatbot(mock_llm, kv_cache)

    print("--- Scenario 1: Repeated FAQ (Cache Reuse) ---")
    chatbot.chat("Where is my order?")
    chatbot.chat("How do I track my delivery?") # New query, but potentially shares some context if system is smarter
    chatbot.chat("Where is my order?") # Repeated query, should be faster due to cache hit

    print("\n--- Scenario 2: Multi-turn Conversation (Prefix Reuse) ---")
    chatbot = EcommerceChatbot(mock_llm, kv_cache) # Reset for a fresh conversation
    chatbot.chat("I need help with a recent purchase.")
    chatbot.chat("My order number is #12345.")
    chatbot.chat("The item arrived damaged.")

    print("\n--- Scenario 3: Cache Eviction ---")
    chatbot = EcommerceChatbot(mock_llm, kv_cache) # Reset cache
    for i in range(1, 8):
        chatbot.chat(f"Query number {i}")
    chatbot.chat("Query number 1") # Should be a miss if it was evicted

    print("\n--- Scenario 4: Another Multi-turn Conversation with potential reuse ---")
    chatbot = EcommerceChatbot(mock_llm, kv_cache) # Reset for a fresh conversation
    chatbot.chat("Tell me about your return policy.")
    chatbot.chat("What is the time limit for returns?")
    chatbot.chat("Do I need the original packaging?")

    # Simulate a user asking a very similar question to a cached one, but not identical
    print("\n--- Scenario 5: Similar but not identical queries (potential miss) ---")
    chatbot = EcommerceChatbot(mock_llm, kv_cache)
    chatbot.chat("I want to know where my package is.")
    chatbot.chat("Can you tell me the status of my shipment?") # Might be a miss if matching is exact
