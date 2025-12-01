class KnowledgeBase:
    def __init__(self):
        self.documents = {
            "doc_1": "Information about product returns: 30-day policy, original packaging required.",
            "doc_2": "Troubleshooting guide for login issues: check username, reset password, clear cache.",
            "doc_3": "Details on subscription plans: basic, premium, enterprise, features, pricing.",
            "doc_4": "Contact support via email, phone, or live chat during business hours."
        }

    def retrieve(self, query):
        relevant_docs = []
        query_lower = query.lower()
        if "return" in query_lower:
            relevant_docs.append(self.documents["doc_1"])
        if "login" in query_lower or "password" in query_lower:
            relevant_docs.append(self.documents["doc_2"])
        if "subscription" in query_lower or "plan" in query_lower:
            relevant_docs.append(self.documents["doc_3"])
        if "contact" in query_lower or "support" in query_lower:
            relevant_docs.append(self.documents["doc_4"])
        return " ".join(relevant_docs) if relevant_docs else "No specific relevant information found."


class LLMSimulator:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt

    def process_query(self, query, context, kv_cache):
        response_parts = []

        if "system_prompt_kv" in kv_cache:
            response_parts.append(f"[Using cached system prompt state: {kv_cache['system_prompt_kv']}] ")
        else:
            kv_cache["system_prompt_kv"] = f"KV for: {self.system_prompt}"
            response_parts.append(f"[Generated system prompt state for: {self.system_prompt}] ")

        context_prefix_key = "context_prefix_kv_" + str(hash(context[:20]))
        if context_prefix_key in kv_cache:
            response_parts.append(f"[Using cached context prefix state: {kv_cache[context_prefix_key]}] ")
        else:
            kv_cache[context_prefix_key] = f"KV for context prefix: {context[:20]}"
            response_parts.append(f"[Generated context prefix state for: {context[:20]}] ")

        full_input = f"System: {self.system_prompt}\nContext: {context}\nQuery: {query}"
        response_parts.append(f"Simulated LLM response for '{query}' based on context.\n")

        return "".join(response_parts), kv_cache


class KVCacheManager:
    def __init__(self, critical_kv_keys):
        self.gpu_kv_cache = {}
        self.host_kv_cache = {}
        self.critical_kv_keys = critical_kv_keys

    def get_gpu_cache(self):
        return self.gpu_kv_cache

    def replicate_critical_nodes(self):
        for key in self.critical_kv_keys:
            if key in self.gpu_kv_cache:
                self.host_kv_cache[key] = self.gpu_kv_cache[key]
        # print(f"Critical nodes replicated. Host cache size: {len(self.host_kv_cache)}")

    def simulate_gpu_failure(self):
        # print("Simulating GPU failure: Clearing GPU KV cache.")
        self.gpu_kv_cache.clear()

    def recover_from_host(self):
        # print("Recovering critical nodes from Host KV cache to GPU KV cache.")
        for key in self.critical_kv_keys:
            if key in self.host_kv_cache:
                self.gpu_kv_cache[key] = self.host_kv_cache[key]


class CustomerSupportAgent:
    def __init__(self, system_prompt):
        self.knowledge_base = KnowledgeBase()
        self.llm_simulator = LLMSimulator(system_prompt)
        self.kv_cache_manager = KVCacheManager(critical_kv_keys=["system_prompt_kv"])
        self.system_prompt = system_prompt

    def handle_query(self, query, simulate_failure_before_query=False, simulate_replication=True):
        if simulate_failure_before_query:
            # print("--- Simulating GPU failure BEFORE processing query ---")
            self.kv_cache_manager.simulate_gpu_failure()
            # print("--- Attempting recovery ---")
            self.kv_cache_manager.recover_from_host()

        context = self.knowledge_base.retrieve(query)
        gpu_cache = self.kv_cache_manager.get_gpu_cache()
        response, updated_gpu_cache = self.llm_simulator.process_query(query, context, gpu_cache)
        self.kv_cache_manager.gpu_kv_cache = updated_gpu_cache

        if simulate_replication:
            self.kv_cache_manager.replicate_critical_nodes()

        return response


if __name__ == "__main__":
    system_prompt_text = "You are a helpful customer support assistant. Provide concise and accurate answers."
    agent = CustomerSupportAgent(system_prompt=system_prompt_text)

    # Scenario 1: Normal operation, initial query
    print("\n--- Scenario 1: Initial Query (Normal Operation) ---")
    query1 = "How do I return a product?"
    response1 = agent.handle_query(query1)
    print(f"User: {query1}\nAgent: {response1}")
    print(f"GPU Cache after query 1: {agent.kv_cache_manager.gpu_kv_cache}")
    print(f"Host Cache after query 1: {agent.kv_cache_manager.host_kv_cache}")

    # Scenario 2: Subsequent query, utilizing cached system prompt and context prefix
    print("\n--- Scenario 2: Subsequent Query (Utilizing Cache) ---")
    query2 = "What are the steps for troubleshooting login issues?"
    response2 = agent.handle_query(query2)
    print(f"User: {query2}\nAgent: {response2}")
    print(f"GPU Cache after query 2: {agent.kv_cache_manager.gpu_kv_cache}")
    print(f"Host Cache after query 2: {agent.kv_cache_manager.host_kv_cache}")

    # Scenario 3: Simulate GPU failure and recovery before a new query
    print("\n--- Scenario 3: GPU Failure and Recovery ---")
    query3 = "Tell me about subscription plans."
    response3 = agent.handle_query(query3, simulate_failure_before_query=True)
    print(f"User: {query3}\nAgent: {response3}")
    print(f"GPU Cache after recovery and query 3: {agent.kv_cache_manager.gpu_kv_cache}")
    print(f"Host Cache after recovery and query 3: {agent.kv_cache_manager.host_kv_cache}")

    # Scenario 4: Another query after recovery, showing restored critical cache usage
    print("\n--- Scenario 4: Post-Recovery Query (Utilizing Restored Cache) ---")
    query4 = "How can I contact support?"
    response4 = agent.handle_query(query4)
    print(f"User: {query4}\nAgent: {response4}")
    print(f"GPU Cache after query 4: {agent.kv_cache_manager.gpu_kv_cache}")
    print(f"Host Cache after query 4: {agent.kv_cache_manager.host_kv_cache}")
