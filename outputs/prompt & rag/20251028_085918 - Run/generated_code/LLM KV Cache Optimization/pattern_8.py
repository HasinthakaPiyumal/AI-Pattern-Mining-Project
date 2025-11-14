import collections
import hashlib

class QueryProcessor:
    """
    Pre-processes incoming customer queries for the LLM.
    Extracts relevant prefixes for KV cache reuse and tokenizes the input.
    """
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer if tokenizer else self._default_tokenizer

    def _default_tokenizer(self, text):
        return text.split()

    def process_query(self, query: str):
        tokens = self.tokenizer(query)
        # For simulation, a prefix is just the first few tokens
        prefix_length = min(len(tokens), 3) # Example: first 3 tokens as prefix
        prefix = " ".join(tokens[:prefix_length])
        return tokens, prefix


class LLMKVCacheManager:
    """
    Manages the Key-Value (KV) cache for the LLM, implementing core optimization strategies:
    KV Cache Reuse, PagedAttention, Swap-Out-Only-Once, and Replication of Critical KV Cache Nodes.
    """
    def __init__(self, gpu_cache_size_pages=10, host_cache_size_pages=100):
        self.gpu_cache_size_pages = gpu_cache_size_pages
        self.host_cache_size_pages = host_cache_size_pages

        # Simulated GPU Cache: { (prefix_hash, page_index): kv_page_data }
        self.gpu_kv_tensors = {}
        # LRU order for GPU cache eviction: [ (prefix_hash, page_index), ... ]
        self.gpu_lru_order = collections.deque()

        # Simulated Host Cache (Swap-Out-Only-Once): { (prefix_hash, page_index): kv_page_data }
        self.host_kv_tensors = {}

        # Critical KV Cache Nodes for fault tolerance: { (prefix_hash, page_index) }
        self.critical_kv_nodes = set()

        print(f"LLMKVCacheManager initialized: GPU cache size={gpu_cache_size_pages} pages, Host cache size={host_cache_size_pages} pages")

    def _generate_kv_page_key(self, prefix: str, page_index: int):
        # Use a hash of the prefix to represent its identity
        prefix_hash = hashlib.sha256(prefix.encode()).hexdigest()
        return (prefix_hash, page_index)

    def _simulate_kv_page_data(self, prefix: str, page_index: int, tokens_in_page: list):
        # Simulate KV page data as a simple string for demonstration
        return f"KV_PAGE_{prefix}_{page_index}_{'_'.join(tokens_in_page)}"

    def retrieve_kv(self, prefix: str, page_index: int):
        key = self._generate_kv_page_key(prefix, page_index)

        if key in self.gpu_kv_tensors:
            # Move to front of LRU for recently used
            if key in self.gpu_lru_order:
                self.gpu_lru_order.remove(key)
            self.gpu_lru_order.appendleft(key)
            print(f"  Cache Hit (GPU) for {key[0][:6]}... page {page_index}")
            return self.gpu_kv_tensors[key]

        if key in self.host_kv_tensors:
            print(f"  Cache Hit (Host) for {key[0][:6]}... page {page_index}. Moving to GPU cache.")
            kv_data = self.host_kv_tensors[key]
            self.store_kv(prefix, page_index, kv_data, from_host=True)
            return kv_data

        print(f"  Cache Miss for {key[0][:6]}... page {page_index}")
        return None

    def store_kv(self, prefix: str, page_index: int, kv_data, is_critical=False, from_host=False):
        key = self._generate_kv_page_key(prefix, page_index)

        if key in self.gpu_kv_tensors and not from_host:
            # Already in GPU cache, just update LRU
            if key in self.gpu_lru_order:
                self.gpu_lru_order.remove(key)
            self.gpu_lru_order.appendleft(key)
            print(f"  KV {key[0][:6]}... page {page_index} already in GPU cache, updated LRU.")
            return

        # If GPU cache is full, evict the least recently used item
        while len(self.gpu_kv_tensors) >= self.gpu_cache_size_pages:
            self._evict_from_gpu()

        self.gpu_kv_tensors[key] = kv_data
        self.gpu_lru_order.appendleft(key)
        print(f"  Stored KV {key[0][:6]}... page {page_index} in GPU cache.")

        if is_critical:
            self.critical_kv_nodes.add(key)
            # Replicate critical nodes to host memory immediately
            if key not in self.host_kv_tensors:
                self.host_kv_tensors[key] = kv_data
                print(f"  Replicated critical KV {key[0][:6]}... page {page_index} to host cache.")
        elif from_host and key in self.host_kv_tensors:
            # If moved from host, remove from host cache only if not critical
            # This is a simplification; a real system might keep non-critical data in host cache for longer
            # For 'Swap-Out-Only-Once' we *only* write to host upon eviction from GPU
            pass

    def _evict_from_gpu(self):
        if not self.gpu_lru_order:
            return

        key_to_evict = self.gpu_lru_order.pop()
        kv_data = self.gpu_kv_tensors.pop(key_to_evict)
        print(f"  Evicting KV {key_to_evict[0][:6]}... page {key_to_evict[1]} from GPU cache.")

        # Swap-Out-Only-Once strategy: copy to host memory only upon initial eviction
        if key_to_evict not in self.host_kv_tensors:
            # Ensure host cache has space if not critical
            if key_to_evict not in self.critical_kv_nodes and len(self.host_kv_tensors) >= self.host_cache_size_pages:
                # For simplicity, if host cache is full for non-critical, we just drop it.
                # A real system would have its own eviction policy for host cache.
                print(f"  Host cache full, cannot store non-critical KV {key_to_evict[0][:6]}... page {key_to_evict[1]} (Swap-Out-Only-Once).")
            else:
                self.host_kv_tensors[key_to_evict] = kv_data
                print(f"  Copied KV {key_to_evict[0][:6]}... page {key_to_evict[1]} to host cache (Swap-Out-Only-Once).")
        else:
            print(f"  KV {key_to_evict[0][:6]}... page {key_to_evict[1]} already in host cache (Swap-Out-Only-Once).")


class LLMInferenceEngine:
    """
    A simulated LLM inference engine that uses the KV cache manager.
    It processes tokens and conceptually generates responses, leveraging cached KV tensors.
    """
    def __init__(self, kv_cache_manager: LLMKVCacheManager):
        self.kv_cache_manager = kv_cache_manager

    def generate_response(self, tokens: list, prefix: str, max_new_tokens=5):
        print(f"\nLLM Inference Engine processing query: {' '.join(tokens)}")
        kv_hits = 0
        kv_misses = 0
        generated_tokens = []

        # Simulate processing tokens in pages
        page_size = 4 # tokens per page
        for i in range(0, len(tokens), page_size):
            current_page_tokens = tokens[i:i+page_size]
            page_index = i // page_size

            # Attempt to retrieve KV data for the current prefix and page
            kv_data = self.kv_cache_manager.retrieve_kv(prefix, page_index)

            if kv_data:
                kv_hits += 1
                print(f"    Using cached KV data for page {page_index}: {kv_data}")
            else:
                kv_misses += 1
                # Simulate generating KV data if not found in cache
                new_kv_data = self.kv_cache_manager._simulate_kv_page_data(prefix, page_index, current_page_tokens)
                # Mark the first page of a critical prefix as critical
                is_critical_page = (page_index == 0) and (hashlib.sha256(prefix.encode()).hexdigest() in [hashlib.sha256("How can I help you?".encode()).hexdigest(), hashlib.sha256("What is your refund policy?".encode()).hexdigest()]) # Example critical prefixes
                self.kv_cache_manager.store_kv(prefix, page_index, new_kv_data, is_critical=is_critical_page)
                print(f"    Generated new KV data for page {page_index}: {new_kv_data}")

            # Simulate generating a response token for this page
            if current_page_tokens:
                generated_tokens.append(f"resp_{current_page_tokens[-1]}")

        # Simulate generating a few more tokens based on the prompt
        for _ in range(max_new_tokens):
            generated_tokens.append(f"LLM_gen_token_{len(generated_tokens)}")

        print(f"  KV Cache Stats: Hits={kv_hits}, Misses={kv_misses}")
        return " ".join(generated_tokens)


class VirtualAssistant:
    """
    Orchestrates the QueryProcessor, LLMKVCacheManager, and LLMInferenceEngine
    to process customer queries and generate intelligent responses.
    """
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.kv_cache_manager = LLMKVCacheManager(gpu_cache_size_pages=3, host_cache_size_pages=10)
        self.llm_inference_engine = LLMInferenceEngine(self.kv_cache_manager)
        print("\nVirtualAssistant initialized and ready to serve customers.")

    def handle_query(self, query: str):
        print(f"\n--- Handling new query: \"{query}\" ---")
        tokens, prefix = self.query_processor.process_query(query)
        print(f"  Processed Query: Prefix='{prefix}', Tokens={tokens}")

        response = self.llm_inference_engine.generate_response(tokens, prefix)
        print(f"  Virtual Assistant Response: {response}")
        return response

# --- Demonstration --- #
if __name__ == "__main__":
    assistant = VirtualAssistant()

    # Scenario 1: First query - all cache misses
    assistant.handle_query("How can I help you with your order status?")

    # Scenario 2: Similar query with shared prefix - KV Cache Reuse
    assistant.handle_query("How can I help you to track my package?")

    # Scenario 3: Different query, potentially causing eviction and swap-out
    assistant.handle_query("What is your refund policy for digital goods?")

    # Scenario 4: Another query, testing host cache retrieval (Swap-Out-Only-Once) and PagedAttention
    assistant.handle_query("I want to know about your refund policy on physical products.")

    # Scenario 5: Re-accessing a previously evicted item from host cache that might have become critical
    # (Simulated for demonstration, marking a prefix as critical if it matches a predefined one)
    assistant.handle_query("How can I help you to get a refund for my order?")

    print("\n--- Final Cache State ---")
    print(f"GPU Cache ({len(assistant.kv_cache_manager.gpu_kv_tensors)} items): {list(assistant.kv_cache_manager.gpu_kv_tensors.keys())}")
    print(f"Host Cache ({len(assistant.kv_cache_manager.host_kv_tensors)} items): {list(assistant.kv_cache_manager.host_kv_tensors.keys())}")
    print(f"Critical KV Nodes ({len(assistant.kv_cache_manager.critical_kv_nodes)} items): {list(assistant.kv_cache_manager.critical_kv_nodes)}")
