import hashlib
import json
import time

class KVCacheManager:
    def __init__(self):
        self.cache = {}

    def generate_prefix_hash(self, prompt_text):
        return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()

    def get(self, prefix_hash):
        return self.cache.get(prefix_hash)

    def put(self, prefix_hash, kv_tensors):
        self.cache[prefix_hash] = kv_tensors

class LLMInferenceWrapper:
    def __init__(self, kv_cache_manager):
        self.kv_cache_manager = kv_cache_manager

    def _simulate_kv_generation(self, text_segment):
        # Simulate generating KV tensors. In a real LLM, this would be actual tensor computation.
        # For simplicity, we use a placeholder string representation of KV tensors.
        time.sleep(0.1) # Simulate computation time
        return f"KV_TENSORS_FOR_'{text_segment}'"

    def _simulate_llm_decode(self, new_tokens_segment, cached_kv_tensors=None):
        # Simulate the decoding phase for new tokens, potentially using cached KV tensors.
        time.sleep(0.05) # Simulate decoding time
        if cached_kv_tensors:
            return f"[Decoded with Cache ({cached_kv_tensors}): Responding to '{new_tokens_segment}']"
        else:
            return f"[Decoded without Cache: Responding to '{new_tokens_segment}']"

    def infer(self, user_message, conversation_history, system_prompt):
        # Construct the full prefix for cache lookup
        prefix_parts = [system_prompt]
        if conversation_history:
            for entry in conversation_history:
                prefix_parts.append(f"User: {entry['user']}")
                prefix_parts.append(f"Assistant: {entry['assistant']}")
        
        prefix_text = "\n".join(prefix_parts)
        prefix_hash = self.kv_cache_manager.generate_prefix_hash(prefix_text)

        cached_kv_tensors = self.kv_cache_manager.get(prefix_hash)

        response_text = ""
        if cached_kv_tensors:
            print(f"[KV Cache Hit for prefix hash: {prefix_hash[:8]}...] Using cached KV tensors.")
            # Simulate decoding for the user's new message
            response_text = self._simulate_llm_decode(user_message, cached_kv_tensors)
        else:
            print(f"[KV Cache Miss for prefix hash: {prefix_hash[:8]}...] Performing full prefill.")
            # Simulate full prefill for the entire prefix
            generated_kv = self._simulate_kv_generation(prefix_text)
            self.kv_cache_manager.put(prefix_hash, generated_kv)
            print(f"[KV Tensors stored in cache for prefix hash: {prefix_hash[:8]}...]")
            # Simulate decoding for the user's new message
            response_text = self._simulate_llm_decode(user_message)
        
        # In a real scenario, the LLM would generate a complete response based on the full input.
        # Here, we'll just simulate a more elaborate response based on the simulated decoding.
        simulated_llm_response = f"Hello! You said: '{user_message}'. My simulated LLM response is: {response_text}"
        return simulated_llm_response

# Chatbot Logic (Main Application)
if __name__ == "__main__":
    kv_cache_manager = KVCacheManager()
    llm_inference_wrapper = LLMInferenceWrapper(kv_cache_manager)

    system_prompt = "You are a helpful customer support assistant. Provide concise and accurate answers."
    conversation_history = []

    print("Intelligent Customer Support Chatbot (type 'exit' to quit)")
    print(f"System Prompt: '{system_prompt}'")
    print("\n--- Start Conversation ---")

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break

        response = llm_inference_wrapper.infer(user_input, conversation_history, system_prompt)
        print(f"Assistant: {response}")

        # Update conversation history for the next turn
        conversation_history.append({"user": user_input, "assistant": response})
        print("\n")

    print("--- Conversation Ended ---")
    print(f"Total KV cache entries: {len(kv_cache_manager.cache)}")
