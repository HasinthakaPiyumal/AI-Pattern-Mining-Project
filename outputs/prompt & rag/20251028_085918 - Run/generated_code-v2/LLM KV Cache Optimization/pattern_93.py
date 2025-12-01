class LLMInference:
    def __init__(self):
        self._token_map = {}
        self._next_token_id = 0

    def _get_token_id(self, char):
        if char not in self._token_map:
            self._token_map[char] = self._next_token_id
            self._next_token_id += 1
        return self._token_map[char]

    def _get_char_from_id(self, token_id):
        for char, tid in self._token_map.items():
            if tid == token_id:
                return char
        return ""

    def tokenize(self, text):
        return [self._get_token_id(char) for char in text]

    def detokenize(self, tokens):
        return "".join([self._get_char_from_id(t_id) for t_id in tokens])

    def generate(self, input_tokens, past_kv=None):
        computed_kv_tensors = []
        for token_id in input_tokens:
            computed_kv_tensors.append(f"K_{token_id}")
            computed_kv_tensors.append(f"V_{token_id}")

        if past_kv:
            updated_kv = past_kv + computed_kv_tensors
        else:
            updated_kv = computed_kv_tensors
        
        # Mock LLM response generation
        response_text = "Bot response to: " + self.detokenize(input_tokens)
        response_tokens = self.tokenize(response_text)
        
        return response_tokens, updated_kv


class KVCacheManager:
    def __init__(self):
        self._cache = {}

    def store(self, prefix_tokens, kv_tensors):
        self._cache[tuple(prefix_tokens)] = kv_tensors

    def retrieve(self, prefix_tokens):
        return self._cache.get(tuple(prefix_tokens))

    def get_longest_prefix_match(self, input_tokens):
        longest_match_tokens = []
        longest_match_kv = None

        for i in range(len(input_tokens), 0, -1):
            prefix_to_check = input_tokens[:i]
            cached_kv = self.retrieve(prefix_to_check)
            if cached_kv:
                longest_match_tokens = prefix_to_check
                longest_match_kv = cached_kv
                break
        return longest_match_tokens, longest_match_kv


class RAGSystem:
    def __init__(self, knowledge_base_data):
        self.knowledge_base = knowledge_base_data

    def retrieve_documents(self, query, top_k=1):
        relevant_docs = []
        query_words = set(query.lower().split())

        for doc in self.knowledge_base:
            if any(word in doc.lower() for word in query_words if len(word) > 2):
                relevant_docs.append(doc)

        return "\n".join(relevant_docs[:top_k])


class Chatbot:
    def __init__(self):
        self.kv_cache_manager = KVCacheManager()
        self.llm = LLMInference()
        self.rag_system = RAGSystem([
            "Our operating hours are Monday to Friday, 9 AM to 5 PM.",
            "You can reset your password by visiting our website and clicking 'Forgot Password'.",
            "We offer a 30-day money-back guarantee on all products.",
            "For technical support, please call us at 1-800-TECH-HELP."
        ])
        self.conversation_history = [] # List of (user_message, bot_response)

    def _format_conversation_history(self):
        formatted_history = ""
        for user_msg, bot_resp in self.conversation_history:
            formatted_history += f"User: {user_msg}\nBot: {bot_resp}\n"
        return formatted_history

    def process_message(self, user_message):
        # 1. RAG Retrieval
        retrieved_docs = self.rag_system.retrieve_documents(user_message, top_k=1)
        rag_prefix = f"\n[Knowledge Base]:\n{retrieved_docs}\n" if retrieved_docs else ""

        # 2. Construct Full Prompt
        history_prefix = self._format_conversation_history()
        full_prompt_text = f"{history_prefix}{rag_prefix}User: {user_message}\nBot: "
        full_prompt_tokens = self.llm.tokenize(full_prompt_text)

        # 3. KV Cache Lookup
        cached_prefix_tokens, cached_kv_tensors = self.kv_cache_manager.get_longest_prefix_match(full_prompt_tokens)
        
        # Identify new tokens to process
        new_tokens_start_idx = len(cached_prefix_tokens)
        new_tokens_to_process = full_prompt_tokens[new_tokens_start_idx:]

        print(f"--- Processing Message ---\nUser: {user_message}")
        print(f"Full Prompt Length (tokens): {len(full_prompt_tokens)}")
        print(f"Cached Prefix Length (tokens): {len(cached_prefix_tokens)}")
        print(f"New Tokens to Process Length: {len(new_tokens_to_process)}")
        if cached_prefix_tokens:
            print(f"Reusing KV cache for prefix: '{self.llm.detokenize(cached_prefix_tokens)}'")
        else:
            print("No KV cache hit, processing full prompt.")

        # 4. LLM Inference
        response_tokens, updated_kv_tensors = self.llm.generate(new_tokens_to_process, past_kv=cached_kv_tensors)

        # 5. Update KV Cache (store the full prompt's KV tensors)
        self.kv_cache_manager.store(full_prompt_tokens, updated_kv_tensors)

        # 6. Generate Response
        bot_response = self.llm.detokenize(response_tokens)
        
        # 7. Update conversation history
        self.conversation_history.append((user_message, bot_response))

        print(f"Bot: {bot_response}\n--------------------------")
        return bot_response


if __name__ == "__main__":
    chatbot = Chatbot()

    print("\n--- Chatbot Session Started ---")
    chatbot.process_message("Hello, what are your operating hours?")
    chatbot.process_message("How can I reset my password?")
    chatbot.process_message("What is your return policy?")
    chatbot.process_message("Can I get help with a technical issue?")
    chatbot.process_message("Thank you, that was helpful.")
    chatbot.process_message("What about the hours again?") # Should reuse prefix + history
    print("\n--- Chatbot Session Ended ---")
