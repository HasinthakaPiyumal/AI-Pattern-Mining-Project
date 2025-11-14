import collections

class PagedKVStore:
    """
    Simulates a paged KV cache, managing KV "tensors" at page granularity.
    In a real scenario, 'pages' would refer to actual memory blocks
    and 'kv_data' would be complex tensor structures. Here, it's simplified
    to store lists of KV entries. PagedAttention concept is applied by
    managing memory in fixed-size "pages" and allowing non-contiguous storage.
    """
    def __init__(self, page_size=2): # Reduced page size for easier demonstration
        self.page_size = page_size
        self.pages = collections.OrderedDict()  # {page_id: list_of_kv_entries}
        self.next_page_id = 0
        self.page_to_prefix = {} # {page_id: prefix} - simplified mapping
        self.prefix_to_pages = collections.defaultdict(list) # {prefix: [page_ids]}

    def _allocate_page(self, prefix):
        page_id = self.next_page_id
        self.next_page_id += 1
        self.pages[page_id] = []
        self.page_to_prefix[page_id] = prefix
        self.prefix_to_pages[prefix].append(page_id)
        return page_id

    def add_kv_entry(self, prefix, kv_entry):
        # Try to find a page for this prefix that isn't full
        for page_id in self.prefix_to_pages[prefix]:
            if len(self.pages[page_id]) < self.page_size:
                self.pages[page_id].append(kv_entry)
                return page_id

        # If no suitable page, allocate a new one
        page_id = self._allocate_page(prefix)
        self.pages[page_id].append(kv_entry)
        return page_id

    def get_kv_entries(self, prefix):
        found_entries = []
        for page_id in self.prefix_to_pages[prefix]:
            found_entries.extend(self.pages[page_id])
        return found_entries if found_entries else None

    def evict_page(self, page_id):
        if page_id in self.pages:
            evicted_data = self.pages.pop(page_id)
            prefix = self.page_to_prefix.pop(page_id)
            self.prefix_to_pages[prefix].remove(page_id)
            if not self.prefix_to_pages[prefix]:
                del self.prefix_to_pages[prefix]
            return prefix, evicted_data
        return None, None

    def load_page(self, page_id, prefix, data):
        self.pages[page_id] = data
        self.page_to_prefix[page_id] = prefix
        self.prefix_to_pages[prefix].append(page_id)


class FaultTolerantKVManager:
    """
    Manages GPU and Host KV caches, implementing Swap-Out-Only-Once and
    Replication of Critical KV Cache Nodes.
    """
    def __init__(self, gpu_cache_max_pages=4, critical_node_prefixes=None):
        self.gpu_cache = PagedKVStore()
        self.gpu_cache_max_pages = gpu_cache_max_pages # max number of pages in GPU
        self.host_cache = {}  # {page_id: (prefix, data)} - for swap-out-only-once
        self.persistent_host_memory = {} # {prefix: {page_id: data}} - for critical node replication
        self.page_lru = collections.deque() # Simple LRU of page_ids in GPU cache
        self.critical_node_prefixes = set(critical_node_prefixes or [])
        self.next_unique_page_id = 0 # To ensure unique page IDs even across reloads

    def _get_next_unique_page_id(self):
        self.next_unique_page_id += 1
        return self.next_unique_page_id - 1

    def _manage_gpu_capacity(self):
        while len(self.gpu_cache.pages) >= self.gpu_cache_max_pages:
            # Evict LRU page
            if not self.page_lru: # Should not happen if capacity is full
                break
            lru_page_id = self.page_lru.popleft()
            prefix, data = self.gpu_cache.evict_page(lru_page_id)

            if data:
                # Swap-Out-Only-Once: Copy to host memory if not already there
                if lru_page_id not in self.host_cache:
                    self.host_cache[lru_page_id] = (prefix, data)
                    print(f"INFO: Swapped out page {lru_page_id} (prefix: '{prefix}') to host memory (first eviction).")
                else:
                    print(f"INFO: Page {lru_page_id} (prefix: '{prefix}') already in host memory, no re-copy needed.")

    def get_kv(self, prefix):
        # Try GPU cache first (KV Cache Reuse)
        kv_entries = self.gpu_cache.get_kv_entries(prefix)
        if kv_entries:
            # Update LRU for all pages associated with this prefix
            for page_id in list(self.gpu_cache.prefix_to_pages[prefix]):
                if page_id in self.page_lru:
                    self.page_lru.remove(page_id)
                    self.page_lru.append(page_id)
            print(f"HIT: KV cache hit for prefix '{prefix}' in GPU. Reusing.")
            return kv_entries

        # Try host cache (if swapped out before)
        host_kv_entries = []
        pages_to_load = []
        for page_id, (p, data) in self.host_cache.items():
            if p == prefix:
                host_kv_entries.extend(data)
                pages_to_load.append((page_id, p, data))

        if host_kv_entries:
            print(f"MISS: KV cache miss in GPU, HIT in host cache for prefix '{prefix}'. Loading back to GPU.")
            for page_id, p, data in pages_to_load:
                self._manage_gpu_capacity()
                self.gpu_cache.load_page(page_id, p, data)
                if page_id not in self.page_lru: # Only add if not already in LRU (could be on other pages)
                    self.page_lru.append(page_id)
            return host_kv_entries

        # Try persistent host memory for critical nodes
        if prefix in self.critical_node_prefixes and prefix in self.persistent_host_memory:
            persistent_kv_entries = []
            for page_id, data in self.persistent_host_memory[prefix].items():
                persistent_kv_entries.extend(data)
                print(f"MISS: KV cache miss, HIT in persistent host memory (critical node) for prefix '{prefix}' (page {page_id}). Loading to GPU.")
                self._manage_gpu_capacity()
                self.gpu_cache.load_page(page_id, prefix, data)
                if page_id not in self.page_lru:
                    self.page_lru.append(page_id)
            return persistent_kv_entries

        print(f"MISS: KV cache miss for prefix '{prefix}'. Must recompute (simulate LLM inference).")
        return None

    def store_kv(self, prefix, kv_entry):
        self._manage_gpu_capacity() # Ensure capacity before adding
        page_id = self.gpu_cache.add_kv_entry(prefix, kv_entry)
        # Update LRU: move to end if exists, add to end if new
        if page_id in self.page_lru:
            self.page_lru.remove(page_id)
        self.page_lru.append(page_id)


        # Replication of Critical KV Cache Nodes
        if prefix in self.critical_node_prefixes:
            # We store the *current* state of all KV entries for this prefix
            # in persistent memory, potentially across multiple pages.
            current_prefix_kv_data = self.gpu_cache.get_kv_entries(prefix)
            if prefix not in self.persistent_host_memory:
                self.persistent_host_memory[prefix] = {}
            # Simplified: just store all current data associated with the prefix on a "virtual" page ID
            # In a real system, this would involve more sophisticated mapping and snapshotting
            critical_page_id = self._get_next_unique_page_id() # A unique ID for the critical snapshot
            self.persistent_host_memory[prefix][critical_page_id] = current_prefix_kv_data
            print(f"INFO: Replicated critical KV for prefix '{prefix}' to persistent host memory (snapshot ID: {critical_page_id}).")

    def simulate_failure_recovery(self):
        """Simulates a GPU failure and recovery using persistent host memory."""
        print("\nSIMULATION: GPU failure detected! Clearing GPU cache.")
        # Reset GPU cache and LRU
        self.gpu_cache = PagedKVStore()
        self.page_lru.clear()
        print("SIMULATION: Attempting to recover critical nodes from persistent host memory...")
        recovered_count = 0
        for prefix, pages_data in self.persistent_host_memory.items():
            print(f"SIMULATION: Recovering critical prefix '{prefix}'.")
            for virtual_page_id, data in pages_data.items(): # Iterate through snapshots/pages
                self._manage_gpu_capacity()
                # When recovering, we need to give it a new actual page ID in GPU
                # if the original one was associated with the failed GPU state.
                # For simplicity, let's reuse original virtual_page_id as actual_page_id
                # and assume it's unique enough for this simulation.
                # A more robust solution would re-allocate and re-map.
                actual_page_id_for_recovery = self._get_next_unique_page_id() # Use a new unique ID
                self.gpu_cache.load_page(actual_page_id_for_recovery, prefix, data)
                self.page_lru.append(actual_page_id_for_recovery)
                # Also restore to host_cache for consistency if it was there
                if actual_page_id_for_recovery not in self.host_cache:
                     self.host_cache[actual_page_id_for_recovery] = (prefix, data)
                recovered_count += 1
        print(f"SIMULATION: Recovered {recovered_count} critical nodes/pages.")
        return recovered_count


class LLMChatbot:
    """
    A simplified LLM Chatbot integrating the fault-tolerant KV cache.
    """
    def __init__(self, name="SupportBot", gpu_cache_max_pages=4, critical_prefixes=None):
        self.name = name
        self.kv_manager = FaultTolerantKVManager(gpu_cache_max_pages, critical_prefixes)
        self.context_history = []

    def _simulate_llm_inference(self, prompt, context_kv=None):
        """
        A placeholder for actual LLM inference.
        If context_kv is provided, it means KV cache was reused.
        """
        response_template = "As your {name}, I understand that '{prompt}' is an important query. "
        if context_kv:
            response_template += f"Leveraging cached context (entries: {len(context_kv)}), I can tell you: {context_kv[0]}... " # Show first entry
            # In a real LLM, context_kv would be actual tensors fed to the model
        else:
            response_template += "I'm processing this fresh, generating new context. "

        response_template += "Let me provide a comprehensive answer shortly."
        full_response = response_template.format(name=self.name, prompt=prompt)
        # Simulate generating a new KV entry (e.g., embedding of the prompt + initial response)
        new_kv_entry = f"KV for '{prompt}' @{self.kv_manager.next_unique_page_id}"
        return full_response, new_kv_entry

    def chat(self, user_query):
        print(f"\n{'='*50}\nUser: {user_query}")

        # Determine prefix for KV cache lookup (simplified: full query as prefix)
        current_prefix = user_query

        # Try to retrieve KV from cache (demonstrates KV Cache Reuse)
        cached_kv = self.kv_manager.get_kv(current_prefix)

        # Simulate LLM inference, using cached_kv if available
        response, new_kv_data = self._simulate_llm_inference(user_query, context_kv=cached_kv)

        # Store new KV data into the cache (demonstrates PagedAttention by adding to a page,
        # and implicitly involves Swap-Out-Only-Once and Replication for critical nodes)
        self.kv_manager.store_kv(current_prefix, new_kv_data)

        self.context_history.append((user_query, response))
        print(f"{self.name}: {response}")
        print(f"Current Cache State: {self.get_current_kv_state()}")
        return response

    def get_current_kv_state(self):
        return {
            "gpu_cache_pages": list(self.kv_manager.gpu_cache.pages.keys()),
            "host_cache_pages": list(self.kv_manager.host_cache.keys()),
            "persistent_host_prefixes": list(self.kv_manager.persistent_host_memory.keys()),
            "gpu_lru": list(self.kv_manager.page_lru)
        }

if __name__ == "__main__":
    # Example Usage
    print("Initializing Chatbot with Fault-Tolerant KV Cache...")
    critical_queries = ["what are your services", "how do I reset my password"]
    chatbot = LLMChatbot(gpu_cache_max_pages=2, critical_prefixes=critical_queries)
    print(f"Chatbot initialized. GPU cache capacity: {chatbot.kv_manager.gpu_cache_max_pages} pages.")
    print(f"Critical prefixes set: {chatbot.kv_manager.critical_node_prefixes}")

    # --- Scenario 1: KV Cache Reuse and PagedAttention --- 
    # Query 1: New query, will result in LLM inference and caching
    chatbot.chat("Hi, what are your services?")

    # Query 2: Similar query prefix, should hit KV cache (KV Cache Reuse)
    chatbot.chat("Can you tell me more about your services?")

    # Query 3: Different query, but a critical one. Will be stored and replicated.
    chatbot.chat("How do I reset my password?")

    # Query 4: Another new query, will likely evict older non-critical data
    chatbot.chat("What are your business hours?")

    # Query 5: Repeated non-critical query that was likely evicted, should hit host cache (Swap-Out-Only-Once)
    chatbot.chat("Hi, what are your services?")

    # --- Scenario 2: Simulate GPU Failure and Recovery (Replication of Critical KV Cache Nodes) ---
    print(f"\n{'='*50}\nSIMULATION START: Testing Fault Tolerance")
    recovered_nodes = chatbot.kv_manager.simulate_failure_recovery()
    print(f"SIMULATION END: Fault Tolerance Test complete. Recovered {recovered_nodes} nodes.")
    print(f"Current Cache State after recovery: {chatbot.get_current_kv_state()}")

    # Query 6: Try a critical query after recovery. Should be recovered from persistent memory.
    chatbot.chat("How do I reset my password?")

    # Query 7: Try a non-critical query that was evicted and not critical. Will be recomputed.
    chatbot.chat("What are your business hours?")

    print(f"\n{'='*50}\nDemo Complete.")