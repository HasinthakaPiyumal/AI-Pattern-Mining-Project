import torch
import hashlib
import time
import os
import joblib
from collections import OrderedDict

# --- Configuration ---
HOST_CACHE_DIR = "bot_cache_host"
CRITICAL_NODES_FILE = os.path.join(HOST_CACHE_DIR, "critical_kv_cache.joblib")
GPU_CACHE_CAPACITY = 3  # Simulated GPU cache capacity (number of KV entries)

# --- 1. KVCacheEntry Class ---
class KVCacheEntry:
    def __init__(self, kv_tensors, is_critical=False):
        self.kv_tensors = kv_tensors  # Mock KV tensors (e.g., a torch.Tensor)
        self.is_critical = is_critical
        self.last_accessed = time.time()

    def update_access_time(self):
        self.last_accessed = time.time()

# --- 2. KVCacheManager Class ---
class KVCacheManager:
    def __init__(self, gpu_cache_capacity=GPU_CACHE_CAPACITY, host_cache_dir=HOST_CACHE_DIR, critical_nodes_file=CRITICAL_NODES_FILE):
        self._gpu_cache_capacity = gpu_cache_capacity
        self._host_cache_dir = host_cache_dir
        self._critical_nodes_file = critical_nodes_file

        os.makedirs(self._host_cache_dir, exist_ok=True)

        # _gpu_cache: Stores KVCacheEntry objects for active GPU cache (OrderedDict for LRU)
        self._gpu_cache = OrderedDict()

        # _host_kv_tensors_paths: Maps prefix_hash to its path in host memory, for "Swap-Out-Only-Once" tracking
        self._host_kv_tensors_paths = {}

        self.load_critical_nodes()

    def _generate_prefix_hash(self, prefix_text: str) -> str:
        """Generates a SHA256 hash for the given prefix text."""
        return hashlib.sha256(prefix_text.encode('utf-8')).hexdigest()

    def _get_host_kv_path(self, prefix_hash: str) -> str:
        """Returns the file path for a KV tensor in host memory."""
        return os.path.join(self._host_cache_dir, f"{prefix_hash}_kv.joblib")

    def add_or_update(self, prefix_text: str, kv_tensors, is_critical: bool = False):
        """
        Adds or updates a KV cache entry. If cache exceeds capacity, evicts LRU.
        kv_tensors: can be a torch.Tensor or a tuple of tensors.
        """
        prefix_hash = self._generate_prefix_hash(prefix_text)

        if prefix_hash in self._gpu_cache:
            entry = self._gpu_cache[prefix_hash]
            entry.kv_tensors = kv_tensors  # Update tensors
            entry.is_critical = is_critical # Update criticality
            entry.update_access_time()
            self._gpu_cache.move_to_end(prefix_hash)
            print(f"DEBUG: Updated KV cache entry for prefix '{prefix_text[:30]}...'")
        else:
            entry = KVCacheEntry(kv_tensors, is_critical)
            self._gpu_cache[prefix_hash] = entry
            print(f"DEBUG: Added new KV cache entry for prefix '{prefix_text[:30]}...'")

        # Evict if capacity exceeded
        if len(self._gpu_cache) > self._gpu_cache_capacity:
            self._evict_lru_from_gpu()

    def get(self, prefix_text: str):
        """
        Retrieves KV tensors for a given prefix.
        Handles loading from host if not in GPU cache and previously swapped out.
        """
        prefix_hash = self._generate_prefix_hash(prefix_text)

        if prefix_hash in self._gpu_cache:
            entry = self._gpu_cache[prefix_hash]
            entry.update_access_time()
            self._gpu_cache.move_to_end(prefix_hash)
            print(f"DEBUG: KV cache HIT (GPU) for prefix '{prefix_text[:30]}...'")
            return entry.kv_tensors
        elif prefix_hash in self._host_kv_tensors_paths:
            # Not in GPU, but was swapped out to host
            host_path = self._host_kv_tensors_paths[prefix_hash]
            try:
                kv_tensors = joblib.load(host_path)
                # Load back to GPU cache (this might trigger another eviction)
                # We re-add it, retaining its criticality if it was critical
                is_critical_on_host = (prefix_hash in self._gpu_cache and self._gpu_cache[prefix_hash].is_critical) # Check if already in _gpu_cache as critical (from critical nodes load)
                self.add_or_update(prefix_text, kv_tensors, is_critical=is_critical_on_host)
                print(f"DEBUG: KV cache HIT (Host load) for prefix '{prefix_text[:30]}...'")
                return kv_tensors
            except Exception as e:
                print(f"ERROR: Failed to load KV tensors from host '{host_path}': {e}")
                # Remove from host tracking if file is corrupted/missing
                if prefix_hash in self._host_kv_tensors_paths:
                    del self._host_kv_tensors_paths[prefix_hash]
                return None
        else:
            print(f"DEBUG: KV cache MISS for prefix '{prefix_text[:30]}...'")
            return None

    def _evict_lru_from_gpu(self):
        """
        Evicts the Least Recently Used entry from the GPU cache.
        Applies "Swap-Out-Only-Once" strategy for non-critical nodes.
        """
        if not self._gpu_cache:
            return

        # Pop the LRU item (first item in OrderedDict)
        lru_hash, lru_entry = self._gpu_cache.popitem(last=False)
        print(f"DEBUG: Evicting LRU entry (hash: {lru_hash[:8]}...) from GPU cache.")

        if not lru_entry.is_critical:
            # Apply Swap-Out-Only-Once for non-critical nodes
            if lru_hash not in self._host_kv_tensors_paths:
                host_path = self._get_host_kv_path(lru_hash)
                try:
                    joblib.dump(lru_entry.kv_tensors, host_path)
                    self._host_kv_tensors_paths[lru_hash] = host_path
                    print(f"DEBUG: Swapped out KV tensors for '{lru_hash[:8]}...' to host (first time).")
                except Exception as e:
                    print(f"ERROR: Failed to swap out KV tensors to host '{host_path}': {e}")
            else:
                print(f"DEBUG: KV tensors for '{lru_hash[:8]}...' already exist on host. Skipping swap-out.")
        else:
            print(f"DEBUG: Critical node '{lru_hash[:8]}...' evicted from GPU. Its replication is handled separately (via '{self._critical_nodes_file}').")
            # Critical nodes are already assumed to be replicated by `replicate_critical_nodes`
            # and their presence in `_host_kv_tensors_paths` is managed by replication/load process.

    def replicate_critical_nodes(self):
        """
        Replicates all currently critical KV cache nodes (from GPU cache) to a single persistent file.
        Also ensures these critical nodes are tracked in `_host_kv_tensors_paths`.
        """
        critical_kv_data = {}
        for prefix_hash, entry in self._gpu_cache.items():
            if entry.is_critical:
                critical_kv_data[prefix_hash] = entry.kv_tensors
                # Ensure critical nodes are also tracked as existing on host (their default individual path)
                if prefix_hash not in self._host_kv_tensors_paths:
                    self._host_kv_tensors_paths[prefix_hash] = self._get_host_kv_path(prefix_hash)

        try:
            joblib.dump(critical_kv_data, self._critical_nodes_file)
            print(f"INFO: Replicated {len(critical_kv_data)} critical KV nodes to '{self._critical_nodes_file}'.")
        except Exception as e:
            print(f"ERROR: Failed to replicate critical KV nodes: {e}")

    def load_critical_nodes(self):
        """
        Loads critical KV cache nodes from persistent host memory into the GPU cache.
        Updates `_host_kv_tensors_paths` for these loaded critical nodes.
        """
        if os.path.exists(self._critical_nodes_file):
            try:
                critical_kv_data = joblib.load(self._critical_nodes_file)
                for prefix_hash, kv_tensors in critical_kv_data.items():
                    # Add to GPU cache and mark as critical
                    # This might trigger eviction of other non-critical nodes if capacity is low
                    self._gpu_cache[prefix_hash] = KVCacheEntry(kv_tensors, is_critical=True)
                    self._gpu_cache.move_to_end(prefix_hash) # Mark as recently used
                    
                    # Ensure they are also tracked as existing on host (as they were loaded from host)
                    self._host_kv_tensors_paths[prefix_hash] = self._get_host_kv_path(prefix_hash)
                        
                print(f"INFO: Loaded {len(critical_kv_data)} critical KV nodes from '{self._critical_nodes_file}'.")
                # Evict if capacity exceeded after loading critical nodes
                while len(self._gpu_cache) > self._gpu_cache_capacity:
                    self._evict_lru_from_gpu()

            except Exception as e:
                print(f"ERROR: Failed to load critical KV nodes: {e}")
        else:
            print(f"INFO: No critical KV nodes file found at '{self._critical_nodes_file}'.")

# --- 3. Mock LLM Inference Service ---
class MockLLM:
    def __init__(self, model_name="MockLLM"):
        self.model_name = model_name
        # A very simplified tokenizer and model for demonstration
        self.vocab = {"hello": 1, "world": 2, "how": 3, "are": 4, "you": 5, "i": 6, "am": 7, "fine": 8, "what": 9, "is": 10, "your": 11, "name": 12, "bot": 13, "the": 14, "weather": 15, "today": 16, "customer": 17, "support": 18, "product": 19, "x": 20, "password": 21, "reset": 22, "features": 23, "<|endoftext|>": 24, "new":25, "topic":26, "about":27, "work":28, "cached":29, "response":30, "for":31}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def _tokenize(self, text: str):
        return [self.vocab.get(word.lower(), 0) for word in text.split() if word.lower() in self.vocab]

    def _detokenize(self, tokens):
        return " ".join([self.reverse_vocab.get(t, "<unk>") for t in tokens])

    def generate(self, input_ids: list, past_key_values=None, max_new_tokens: int = 10):
        """
        Simulates LLM generation.
        input_ids: List of integer tokens for the *new* part of the sequence.
        past_key_values: Mock KV tensors for the *previous* part of the sequence.
        Returns: Tuple of (generated_ids, new_past_key_values).
        """
        print(f"LLM: Processing input_ids: {input_ids}")
        
        current_kv_length = 0
        if past_key_values is not None:
            print(f"LLM: Reusing past_key_values with shape: {past_key_values.shape}")
            current_kv_length = past_key_values.shape[1] if past_key_values.ndim > 1 else 0
            # For simulation, just concatenate. In a real LLM, past_key_values would be passed
            # to the model's forward pass, and new_kv would be computed.
            mock_new_kv_for_input = torch.randn(1, len(input_ids), 4) # Mock KV for current input
            new_past_key_values_base = torch.cat([past_key_values, mock_new_kv_for_input], dim=1)
        else:
            print("LLM: No past_key_values, generating from scratch.")
            new_past_key_values_base = torch.randn(1, len(input_ids), 4) # (batch, sequence_length, hidden_dim)

        generated_ids = []
        final_kv_tensors = new_past_key_values_base

        for i in range(max_new_tokens):
            # Simulate generating a new token
            if not input_ids:
                 next_token_id = (i + 1) % (len(self.vocab)-1) + 1 # Simple next token logic if no input
            else:
                next_token_id = (input_ids[-1] + i + 1) % (len(self.vocab)-1) + 1 # Simple next token logic based on last input

            if next_token_id == self.vocab["<|endoftext|>" or 0]:
                break
            generated_ids.append(next_token_id)
            
            # In a real LLM, new_past_key_values would be updated with each generated token.
            # Here, we just extend the mock KV tensors.
            final_kv_tensors = torch.cat([final_kv_tensors, torch.randn(1, 1, 4)], dim=1)

        print(f"LLM: Generated {len(generated_ids)} tokens.")
        return generated_ids, final_kv_tensors


# --- 4. Intelligent Customer Support Bot (Main Application Logic) ---
class IntelligentCustomerSupportBot:
    def __init__(self, model_name="MockLLM", gpu_cache_capacity=GPU_CACHE_CAPACITY):
        self.llm = MockLLM(model_name)
        self.kv_cache_manager = KVCacheManager(gpu_cache_capacity)
        # Tracks the *last complete conversation prefix text* and its *KV tensors* for each customer
        self.customer_last_kv_context = {} # {customer_id: {'prefix_text': str, 'kv_tensors': Any}}

    def process_customer_query(self, customer_id: str, query: str) -> str:
        """
        Processes a customer query, leveraging KV cache.
        """
        print(f"\n--- Customer {customer_id} Query: '{query}' ---")
        
        # Get the previous context (full prefix text and its KV tensors) for this customer
        last_context = self.customer_last_kv_context.get(customer_id)
        
        current_query_ids = self.llm._tokenize(query)

        llm_input_ids_for_generation = current_query_ids
        past_kv_tensors_for_generation = None
        full_sequence_text_for_new_kv = query # This will be the key for the KV cache

        if last_context:
            # We have previous KV tensors, so the LLM will continue from there.
            # The LLM's 'generate' will take the current query's tokens and the past KVs.
            past_kv_tensors_for_generation = last_context['kv_tensors']
            # The full sequence that the *new* KV tensors will represent is (last_context_prefix + current_query_text)
            full_sequence_text_for_new_kv = last_context['prefix_text'] + " " + query
            print(f"DEBUG: Continuing conversation for customer {customer_id}. Full sequence for new KV: '{full_sequence_text_for_new_kv[:50]}...'")
        else:
            print(f"DEBUG: Starting new conversation for customer {customer_id}. Full sequence for new KV: '{full_sequence_text_for_new_kv[:50]}...'")

        # 1. Try to get KV tensors for the *full_sequence_text_for_new_kv* from cache
        # This would be a direct hit if this *exact entire sequence* was processed before.
        cached_kv_for_full_sequence = self.kv_cache_manager.get(full_sequence_text_for_new_kv)
        
        generated_ids = []
        new_kv_tensors = None

        if cached_kv_for_full_sequence is not None:
            # Direct cache hit for the entire sequence. No need for LLM generation.
            print(f"DEBUG: Direct KV Cache hit for full sequence '{full_sequence_text_for_new_kv[:50]}...'. Reusing stored KV.")
            new_kv_tensors = cached_kv_for_full_sequence
            generated_ids = self.llm._tokenize("Cached response for " + query) # Mock a response from cache
        else:
            # 2. LLM Inference (using past_kv_tensors_for_generation if available)
            generated_ids, new_kv_tensors = self.llm.generate(llm_input_ids_for_generation, past_key_values=past_kv_tensors_for_generation)
        
        # 3. Update KV cache with new tensors (for the *full_sequence_text_for_new_kv*)
        # Decide if this full sequence should be marked as critical (e.g., for common FAQs that complete a convo)
        is_critical = "password reset" in full_sequence_text_for_new_kv.lower() or "product x features" in full_sequence_text_for_new_kv.lower()
        self.kv_cache_manager.add_or_update(full_sequence_text_for_new_kv, new_kv_tensors, is_critical=is_critical)
        
        # 4. Update customer's last KV context for next turn
        self.customer_last_kv_context[customer_id] = {
            'prefix_text': full_sequence_text_for_new_kv,
            'kv_tensors': new_kv_tensors
        }
        
        bot_response = self.llm._detokenize(generated_ids)
        return f"Bot Response for Customer {customer_id}: {bot_response}"

    def replicate_critical_data(self):
        """Initiates replication of critical KV cache nodes."""
        self.kv_cache_manager.replicate_critical_nodes()

# --- Main Execution / Demonstration ---
if __name__ == "__main__":
    # Clean up previous cache for a fresh start
    if os.path.exists(HOST_CACHE_DIR):
        print(f"INFO: Clearing previous cache directory: {HOST_CACHE_DIR}")
        try:
            for f in os.listdir(HOST_CACHE_DIR):
                os.remove(os.path.join(HOST_CACHE_DIR, f))
            os.rmdir(HOST_CACHE_DIR)
        except OSError as e:
            print(f"WARNING: Could not completely clear cache directory: {e}")
    
    bot = IntelligentCustomerSupportBot(gpu_cache_capacity=GPU_CACHE_CAPACITY)

    print("\n--- SCENARIO 1: KV Cache Reuse and Eviction ---")
    
    # Query 1 (new prefix)
    print(bot.process_customer_query("cust_A", "Hello how are you?")) 
    time.sleep(0.1)

    # Query 2 (another new prefix)
    print(bot.process_customer_query("cust_B", "What is the weather today?"))
    time.sleep(0.1)

    # Query 3 (another new prefix, will cause eviction of LRU from Q1)
    print(bot.process_customer_query("cust_C", "Tell me about product x features.")) 
    time.sleep(0.1)
    
    # Query 4 (continuation of cust_A, should use past_kv for "Hello how are you?")
    # The full sequence for new KV will be "Hello how are you? I am fine."
    print(bot.process_customer_query("cust_A", "I am fine.")) 
    time.sleep(0.1)
    
    # Query 5 (New customer, exact same initial query as Q3 - direct KV cache reuse for "Tell me about product x features.")
    # This should trigger a direct KV cache hit for the full sequence "Tell me about product x features."
    print(bot.process_customer_query("cust_D", "Tell me about product x features.")) 
    time.sleep(0.1)

    # Query 6 (Another new query to force more eviction)
    print(bot.process_customer_query("cust_E", "New topic for E."))
    time.sleep(0.1)

    # Query 7 (Access a very old conversation prefix, possibly loaded from host if evicted)
    # The full sequence "Hello how are you? I am fine." might have been evicted and swapped out.
    # This will continue from that and create a new full sequence: "Hello how are you? I am fine. What about your name?"
    print(bot.process_customer_query("cust_A", "What about your name?"))
    time.sleep(0.1)


    print("\n--- SCENARIO 2: Critical Node Replication and Recovery ---")
    
    # Simulate marking a specific interaction as critical and replicating
    critical_query = "How do I reset my password?"
    print(bot.process_customer_query("cust_CRITICAL", critical_query))
    
    # Manually ensure the last stored prefix for cust_CRITICAL is marked critical and then replicate
    critical_prefix_text_for_cust_critical = bot.customer_last_kv_context["cust_CRITICAL"]['prefix_text']
    critical_prefix_hash_for_cust_critical = bot.kv_cache_manager._generate_prefix_hash(critical_prefix_text_for_cust_critical)
    
    if critical_prefix_hash_for_cust_critical in bot.kv_cache_manager._gpu_cache:
        bot.kv_cache_manager._gpu_cache[critical_prefix_hash_for_cust_critical].is_critical = True
        print(f"DEBUG: Manually marked prefix for '{critical_query[:30]}...' as critical.")

    bot.replicate_critical_data() # Replicate critical nodes

    # Simulate a crash and restart of the bot (clearing in-memory cache)
    print("\n--- Simulating Bot Restart / Crash Recovery ---")
    del bot # Destroy old bot instance
    
    # Re-initialize bot - it should load critical nodes from disk
    new_bot = IntelligentCustomerSupportBot(gpu_cache_capacity=GPU_CACHE_CAPACITY)
    
    # Query using the critical prefix again - it should be in the cache due to recovery
    # This should result in a direct KV cache hit for the full critical sequence.
    print(new_bot.process_customer_query("cust_CRITICAL", critical_query)) 
    
    # Try another query that was not critical, to see if it's still available (if not evicted/swapped out)
    # This prefix might have been evicted and potentially swapped out to host from the previous bot instance.
    print(new_bot.process_customer_query("cust_A", "How do you work?")) 

    print("\n--- SCENARIO 3: Swap-Out-Only-Once Detailed ---")
    # Fill cache to force evictions
    print(new_bot.process_customer_query("cust_F", "Query F1")) # Add 1
    print(new_bot.process_customer_query("cust_G", "Query G1")) # Add 2
    print(new_bot.process_customer_query("cust_H", "Query H1")) # Add 3 (capacity reached)

    print(new_bot.process_customer_query("cust_I", "Query I1")) # Add 4, evicts F1 (F1 swapped out once)

    # Now, if we try to access F1 (its original prefix), it will be loaded from host
    print(new_bot.process_customer_query("cust_F", "Query F2")) # F1 loaded, then F1's new KV is added.
    
    # If F1's continued conversation (F1 + F2) gets evicted again, it won't be swapped out again for F1's original prefix.
    # The new combined prefix "Query F1 Query F2" will be handled. If it gets evicted, it will be swapped out *once*.
    print(new_bot.process_customer_query("cust_J", "Query J1")) # Evicts G1
    print(new_bot.process_customer_query("cust_K", "Query K1")) # Evicts H1
    print(new_bot.process_customer_query("cust_L", "Query L1")) # Evicts (potentially) F1's continued conversation, but won't re-swap F1's original prefix.
    print("\n--- End of Demonstration ---")
