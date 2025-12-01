import json
import os
import time

class GPUMemoryKVCache:
    def __init__(self):
        self.cache = {}
        self.critical_keys = set() # Keys marked for replication

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value, critical=False):
        self.cache[key] = value
        if critical:
            self.critical_keys.add(key)

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
        if key in self.critical_keys:
            self.critical_keys.remove(key)

    def get_critical_kv_pairs(self):
        return {k: self.cache[k] for k in self.critical_keys if k in self.cache}

    def clear(self):
        self.cache.clear()
        self.critical_keys.clear()

class HostMemoryKVCache:
    def __init__(self, filename="host_cache.json"):
        self.filename = filename
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.filename, "w") as f:
            json.dump(self.cache, f, indent=4)

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value
        self._save_cache()

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
            self._save_cache()

    def get_all(self):
        return self.cache.copy()

class ReplicationManager:
    def __init__(self, gpu_cache: GPUMemoryKVCache, host_cache: HostMemoryKVCache):
        self.gpu_cache = gpu_cache
        self.host_cache = host_cache

    def replicate_critical_nodes(self):
        critical_kv = self.gpu_cache.get_critical_kv_pairs()
        for key, value in critical_kv.items():
            self.host_cache.set(key, value)
        print("Replication: Critical KV nodes replicated to host memory.")

class Chatbot:
    def __init__(self):
        self.gpu_cache = GPUMemoryKVCache()
        self.host_cache = HostMemoryKVCache()
        self.replication_manager = ReplicationManager(self.gpu_cache, self.host_cache)
        self.conversation_history_key = "conversation_history"
        self.system_prompt_key = "system_prompt"
        self._initialize_caches()

    def _initialize_caches(self):
        # Set initial critical system prompt
        initial_system_prompt = "You are a helpful customer support AI. Provide concise and accurate answers."
        self.gpu_cache.set(self.system_prompt_key, initial_system_prompt, critical=True)

        # Simulate frequently accessed product info as critical
        self.gpu_cache.set("product_A_info", "Details for Product A: High-performance, 1-year warranty.", critical=True)
        self.gpu_cache.set("faq_shipping", "Shipping usually takes 3-5 business days.", critical=True)

        # Replicate initial critical items immediately
        self.replication_manager.replicate_critical_nodes()

    def _simulate_llm_response(self, context, query):
        if "GPU_FAILURE" in query: # Special query to simulate failure
            raise RuntimeError("Simulated GPU failure!")

        # Simple LLM simulation: combine context and query
        response = f"(LLM processed with context: {context[:50]}...) Based on your query '{query}', I can say..."

        if "product A" in query.lower():
            response += " Product A is known for its high-performance and comes with a 1-year warranty."
        elif "shipping" in query.lower():
            response += " Shipping typically takes 3-5 business days. Do you have a tracking number?"
        elif "hello" in query.lower() or "hi" in query.lower():
            response = "Hello! How can I assist you today?"
        elif "thank you" in query.lower():
            response = "You're welcome! Is there anything else?"
        elif "system prompt" in query.lower():
            response = f"My current system prompt guides me to be: '{self.gpu_cache.get(self.system_prompt_key)}'"
        else:
            response += " I'm sorry, I don't have enough information on that. Could you please rephrase or provide more details?"

        return response

    def _simulate_retriever(self, query):
        retrieved_docs = []
        # Simple keyword matching for demonstration
        if "product A" in query.lower():
            retrieved_docs.append(self.gpu_cache.get("product_A_info"))
        if "shipping" in query.lower():
            retrieved_docs.append(self.gpu_cache.get("faq_shipping"))
        if not retrieved_docs:
            retrieved_docs.append("General knowledge base information.")
        return " ".join([doc for doc in retrieved_docs if doc is not None])

    def _update_conversation_cache(self, user_input, chatbot_response):
        history = self.gpu_cache.get(self.conversation_history_key)
        if history is None:
            history = []
        history.append({"user": user_input, "bot": chatbot_response})
        self.gpu_cache.set(self.conversation_history_key, history)

    def _recover_gpu_cache(self):
        print("Attempting GPU cache recovery from host memory...")
        self.gpu_cache.clear() # Clear potentially corrupted GPU cache
        for key, value in self.host_cache.get_all().items():
            # Mark all recovered items as critical again
            self.gpu_cache.set(key, value, critical=True)
        print("GPU cache recovered successfully from host memory.")

    def chat(self, user_input):
        try:
            # Simulate potential GPU cache update during active conversation
            current_history = self.gpu_cache.get(self.conversation_history_key) or []
            current_context = f"System Prompt: {self.gpu_cache.get(self.system_prompt_key)}. Conversation History: {current_history[-2:]}"

            # Retrieve relevant information
            retrieved_info = self._simulate_retriever(user_input)
            full_context = f"{current_context}. Retrieved Info: {retrieved_info}"

            # Get LLM response
            chatbot_response = self._simulate_llm_response(full_context, user_input)

            # Update conversation history in GPU cache
            self._update_conversation_cache(user_input, chatbot_response)

            # Periodically replicate critical nodes (simulate a background process)
            if len(self.gpu_cache.cache) % 5 == 0: # Replicate every 5 user turns
                self.replication_manager.replicate_critical_nodes()

            return chatbot_response

        except RuntimeError as e:
            print(f"Error: {e}. Initiating recovery process.")
            self._recover_gpu_cache()
            # After recovery, retry the current query or inform the user
            return "A temporary issue occurred, but critical systems have been restored. Please repeat your last query."

    def run_cli(self):
        print("Welcome to the Reliable Customer Support Chatbot! (Type 'exit' to quit or 'GPU_FAILURE' to simulate a GPU crash)")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            response = self.chat(user_input)
            print(f"Bot: {response}")

if __name__ == "__main__":
    # Clean up previous host cache for a fresh start
    if os.path.exists("host_cache.json"):
        os.remove("host_cache.json")

    chatbot_app = Chatbot()
    chatbot_app.run_cli()