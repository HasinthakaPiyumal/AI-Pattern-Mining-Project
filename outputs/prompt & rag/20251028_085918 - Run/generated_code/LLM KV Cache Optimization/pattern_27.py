class LLMInferenceModule:
    def infer(self, prompt: str, use_kv_cache: bool = False):
        if use_kv_cache:
            response = f"[FAST - KV CACHE REUSED] Bot's answer to: {prompt}"
            return response, True
        else:
            response = f"[FULL INFERENCE] Bot's answer to: {prompt}"
            return response, False

class KVCMCacheManager:
    def __init__(self):
        self.kv_cache = {}

    def store_kv_tensors(self, prefix: str, kv_tensors: str):
        self.kv_cache[prefix] = kv_tensors

    def retrieve_kv_tensors(self, prefix: str):
        return self.kv_cache.get(prefix)

    def get_all_prefixes(self):
        return list(self.kv_cache.keys())


class PrefixMatcher:
    def find_longest_common_prefix(self, query: str, cached_prefixes: list):
        longest_prefix = ""
        for cached_prefix in cached_prefixes:
            if query.startswith(cached_prefix) and len(cached_prefix) > len(longest_prefix):
                longest_prefix = cached_prefix
        return longest_prefix


class ChatbotCoreLogic:
    def __init__(self, prefix_length_for_cache: int = 5):
        self.llm_inference_module = LLMInferenceModule()
        self.kv_cache_manager = KVCMCacheManager()
        self.prefix_matcher = PrefixMatcher()
        self.prefix_length_for_cache = prefix_length_for_cache

    def handle_query(self, query: str):
        cached_prefixes = self.kv_cache_manager.get_all_prefixes()
        common_prefix = self.prefix_matcher.find_longest_common_prefix(query, cached_prefixes)

        if common_prefix:
            print(f"DEBUG: Common prefix found: '{common_prefix}'. Reusing KV cache.")
            response, cache_reused = self.llm_inference_module.infer(query, use_kv_cache=True)
        else:
            print(f"DEBUG: No common prefix found for '{query}'. Performing full inference.")
            response, cache_reused = self.llm_inference_module.infer(query, use_kv_cache=False)
            # If no cache hit, and query is long enough, store a new prefix
            if len(query) >= self.prefix_length_for_cache:
                new_prefix = query[:self.prefix_length_for_cache]
                # Simulate storing KV tensors for the new prefix
                self.kv_cache_manager.store_kv_tensors(new_prefix, f"simulated_kv_for_{new_prefix}")
                print(f"DEBUG: Stored new prefix '{new_prefix}' in KV cache.")

        return response, cache_reused


if __name__ == "__main__":
    chatbot = ChatbotCoreLogic(prefix_length_for_cache=10)

    print("\n--- First set of queries (establishing cache) ---")
    query1 = "What is your refund policy for digital products?"
    response1, cached1 = chatbot.handle_query(query1)
    print(f"Customer: {query1}\nChatbot: {response1}\n")

    query2 = "What is your refund policy for physical items?"
    response2, cached2 = chatbot.handle_query(query2)
    print(f"Customer: {query2}\nChatbot: {response2}\n")

    query3 = "How can I reset my password?"
    response3, cached3 = chatbot.handle_query(query3)
    print(f"Customer: {query3}\nChatbot: {response3}\n")

    print("\n--- Second set of queries (leveraging cache) ---")
    query4 = "What is your refund policy regarding software licenses?"
    response4, cached4 = chatbot.handle_query(query4)
    print(f"Customer: {query4}\nChatbot: {response4}\n")

    query5 = "What is your refund policy on sale items?"
    response5, cached5 = chatbot.handle_query(query5)
    print(f"Customer: {query5}\nChatbot: {response5}\n")

    query6 = "How can I reset my account password if I forgot it?"
    response6, cached6 = chatbot.handle_query(query6)
    print(f"Customer: {query6}\nChatbot: {response6}\n")

    query7 = "I need help with something else."
    response7, cached7 = chatbot.handle_query(query7)
    print(f"Customer: {query7}\nChatbot: {response7}\n")

    print("\n--- Inspecting KV Cache (simulated) ---")
    print("Cached prefixes:", chatbot.kv_cache_manager.get_all_prefixes())
