import random
import time

class CacheEntry:
    def __init__(self, key, kv_tensor, size, cost):
        self.key = key
        self.kv_tensor = kv_tensor
        self.size = size
        self.frequency = 1
        self.last_accessed_clock = 0
        self.cost = cost

class PGDSFCache:
    def __init__(self, gpu_capacity_tokens, host_capacity_tokens):
        self.gpu_capacity = gpu_capacity_tokens
        self.host_capacity = host_capacity_tokens
        self.gpu_cache = {}
        self.host_cache = {}
        self.current_gpu_size = 0
        self.current_host_size = 0
        self.current_clock = 0

    def _calculate_priority(self, entry):
        # Priority = Clock * Frequency * Cost / Size
        # Use a small epsilon to avoid division by zero if Size is somehow 0
        return (self.current_clock - entry.last_accessed_clock + 1) * entry.frequency * entry.cost / (entry.size + 1e-6)

    def _get_cost(self, key, prefix_length):
        # Simulated prefix-aware recomputation cost
        # Higher cost for shorter prefixes (more recomputation needed)
        # Specific keys could also have different base costs
        base_cost = hash(key) % 10 + 1 # Simple key-based variation
        return max(1, base_cost * (100 - prefix_length) / 100) # Longer prefixes reduce cost

    def get(self, key, prefix_length):
        self.current_clock += 1
        
        # Check GPU cache
        if key in self.gpu_cache:
            entry = self.gpu_cache[key]
            entry.frequency += 1
            entry.last_accessed_clock = self.current_clock
            entry.cost = self._get_cost(key, prefix_length) # Re-estimate cost on access if needed
            # print(f"GPU Hit: {key}, Priority: {self._calculate_priority(entry):.2f}")
            return entry.kv_tensor

        # Check Host cache
        if key in self.host_cache:
            entry = self.host_cache[key]
            entry.frequency += 1
            entry.last_accessed_clock = self.current_clock
            entry.cost = self._get_cost(key, prefix_length) # Re-estimate cost on access if needed
            # print(f"Host Hit: {key}, Priority: {self._calculate_priority(entry):.2f}")

            # Promote to GPU if space allows
            if self.current_gpu_size + entry.size <= self.gpu_capacity:
                del self.host_cache[key]
                self.current_host_size -= entry.size
                self.gpu_cache[key] = entry
                self.current_gpu_size += entry.size
                # print(f"Promoted {key} to GPU.")
            else:
                # If GPU is full, try to make space by evicting lowest priority from GPU
                # and then promoting.
                # This is a simplification; a full GDSF would integrate promotion directly into eviction
                # For this demo, we'll just return from host if no immediate GPU space.
                pass # For simplicity, if GPU full, it stays in host until explicit put
            return entry.kv_tensor

        # Cache miss
        # print(f"Cache Miss: {key}")
        return None

    def put(self, key, kv_tensor, size, prefix_length):
        self.current_clock += 1
        cost = self._get_cost(key, prefix_length)
        
        if key in self.gpu_cache:
            entry = self.gpu_cache[key]
            self.current_gpu_size -= entry.size # Update size if tensor content/size changed
            entry.kv_tensor = kv_tensor
            entry.size = size
            entry.frequency += 1
            entry.last_accessed_clock = self.current_clock
            entry.cost = cost
            self.current_gpu_size += entry.size
        elif key in self.host_cache:
            entry = self.host_cache[key]
            self.current_host_size -= entry.size
            del self.host_cache[key]
            entry.kv_tensor = kv_tensor
            entry.size = size
            entry.frequency += 1
            entry.last_accessed_clock = self.current_clock
            entry.cost = cost

            # Attempt to put into GPU even if it was in host
            if self.current_gpu_size + size > self.gpu_capacity:
                self._evict_from_gpu()
            self.gpu_cache[key] = entry
            self.current_gpu_size += entry.size

        else: # New entry
            entry = CacheEntry(key, kv_tensor, size, cost)
            entry.last_accessed_clock = self.current_clock # First access sets the clock

            if self.current_gpu_size + size > self.gpu_capacity:
                self._evict_from_gpu()
            
            if self.current_gpu_size + size <= self.gpu_capacity: # Check again after eviction
                self.gpu_cache[key] = entry
                self.current_gpu_size += size
                # print(f"Added {key} to GPU cache. GPU size: {self.current_gpu_size}/{self.gpu_capacity}")
            else:
                # If it still doesn't fit in GPU after eviction, try host
                if self.current_host_size + size > self.host_capacity:
                    self._evict_from_host()
                
                if self.current_host_size + size <= self.host_capacity: # Check again after eviction
                    self.host_cache[key] = entry
                    self.current_host_size += size
                    # print(f"Added {key} to Host cache. Host size: {self.current_host_size}/{self.host_capacity}")
                else:
                    # print(f"Warning: {key} could not fit in either cache after eviction.")
                    pass # Item dropped if it doesn't fit anywhere


    def _evict_from_gpu(self):
        if not self.gpu_cache:
            return
        
        # Find the entry with the lowest priority
        lowest_priority_key = None
        lowest_priority = float('inf')

        for key, entry in self.gpu_cache.items():
            priority = self._calculate_priority(entry)
            if priority < lowest_priority:
                lowest_priority = priority
                lowest_priority_key = key
        
        if lowest_priority_key:
            evicted_entry = self.gpu_cache[lowest_priority_key]
            del self.gpu_cache[lowest_priority_key]
            self.current_gpu_size -= evicted_entry.size
            # print(f"Evicting {lowest_priority_key} from GPU (Priority: {lowest_priority:.2f}). GPU size: {self.current_gpu_size}/{self.gpu_capacity}")

            # Move to host cache
            if self.current_host_size + evicted_entry.size > self.host_capacity:
                self._evict_from_host()

            if self.current_host_size + evicted_entry.size <= self.host_capacity: # Check again after host eviction
                self.host_cache[evicted_entry.key] = evicted_entry
                self.current_host_size += evicted_entry.size
                # print(f"Moved {evicted_entry.key} to Host cache. Host size: {self.current_host_size}/{self.host_capacity}")
            else:
                # print(f"Warning: Evicted {evicted_entry.key} from GPU but couldn't fit in Host.")
                pass # Dropped if it doesn't fit in host


    def _evict_from_host(self):
        if not self.host_cache:
            return
        
        lowest_priority_key = None
        lowest_priority = float('inf')

        for key, entry in self.host_cache.items():
            priority = self._calculate_priority(entry)
            if priority < lowest_priority:
                lowest_priority = priority
                lowest_priority_key = key
        
        if lowest_priority_key:
            evicted_entry = self.host_cache[lowest_priority_key]
            del self.host_cache[lowest_priority_key]
            self.current_host_size -= evicted_entry.size
            # print(f"Evicting {lowest_priority_key} from Host (Priority: {lowest_priority:.2f}). Host size: {self.current_host_size}/{self.host_capacity}")


class RAGSystem:
    def __init__(self, gpu_cache_capacity_tokens, host_cache_capacity_tokens):
        self.cache = PGDSFCache(gpu_cache_capacity_tokens, host_cache_capacity_tokens)
        self.total_queries = 0
        self.cache_hits = 0

    def _retrieve_document_segment(self, query, document_id, segment_id):
        # Simulate fetching a document segment's KV tensor and size from a knowledge base
        # In a real system, this would involve embedding retrieval, LLM processing etc.
        kv_tensor = f"KV_TENSOR_DOC{{document_id}}_SEG{{segment_id}}_for_query:_{query[:10]}"
        size = random.randint(30, 150) # Simulate variable sizes
        # print(f"Simulated retrieval for {document_id}-{segment_id}. Size: {size}")
        return kv_tensor, size

    def _generate_response_from_kv(self, kv_tensor, query):
        # Simulate LLM generation based on the retrieved KV tensor and query
        return f"Response generated using {kv_tensor} for query: '{query}'"

    def query_system(self, query, document_id, segment_id, prefix_length):
        self.total_queries += 1
        key = f"doc_{document_id}_seg_{segment_id}"
        
        # Try to get KV tensor from cache
        kv_tensor = self.cache.get(key, prefix_length)
        
        if kv_tensor is None: # Cache miss
            # print(f"RAG Miss: Fetching {key} from knowledge base.")
            retrieved_kv_tensor, retrieved_size = self._retrieve_document_segment(query, document_id, segment_id)
            self.cache.put(key, retrieved_kv_tensor, retrieved_size, prefix_length)
            kv_tensor = retrieved_kv_tensor
        else: # Cache hit
            self.cache_hits += 1
            # print(f"RAG Hit: Using cached {key}.")
        
        return self._generate_response_from_kv(kv_tensor, query)


# --- Simulation --- 
if __name__ == "__main__":
    GPU_CAPACITY = 500 # tokens
    HOST_CAPACITY = 1000 # tokens

    rag_system = RAGSystem(GPU_CAPACITY, HOST_CAPACITY)

    queries = [
        ("What is product A feature X?", 1, 1, 20),
        ("How to troubleshoot error Y in product B?", 2, 1, 15),
        ("What are the specifications of product A?", 1, 2, 25),
        ("Product B installation guide step 3.", 2, 2, 10),
        ("Product A feature X advanced usage.", 1, 1, 30), # Repeat, higher prefix_length
        ("Error Y resolution steps.", 2, 1, 12),       # Repeat
        ("Compare product A and C.", 3, 1, 5),
        ("Product B safety guidelines.", 2, 3, 8),
        ("Product C warranty information.", 3, 2, 18),
        ("Product A troubleshooting common issues.", 1, 3, 15),
        ("What is product A feature X?", 1, 1, 22), # Repeat with slightly different prefix
        ("Details about product B warranty.", 2, 3, 10), # Repeat
        ("Product C advanced features.", 3, 1, 7),
        ("Product A performance metrics.", 1, 2, 40)
    ]

    print(f"--- Starting RAG System Simulation with PGDSF Caching ---")
    print(f"GPU Cache Capacity: {GPU_CAPACITY} tokens, Host Cache Capacity: {HOST_CAPACITY} tokens\n")

    for i, (query, doc_id, seg_id, prefix_len) in enumerate(queries):
        print(f"Query {i+1}: '{query[:30]}...' (Doc {doc_id}, Seg {seg_id}, Prefix {prefix_len})")
        response = rag_system.query_system(query, doc_id, seg_id, prefix_len)
        # print(f"Response: {response[:50]}...")
        print(f"  Current GPU Cache Size: {rag_system.cache.current_gpu_size}/{rag_system.cache.gpu_capacity}")
        print(f"  Current Host Cache Size: {rag_system.cache.current_host_size}/{rag_system.cache.host_capacity}")
        print("  GPU Cache Keys:", list(rag_system.cache.gpu_cache.keys()))
        print("  Host Cache Keys:", list(rag_system.cache.host_cache.keys()))
        print("---------------------------------------------------")
        time.sleep(0.1) # Simulate some processing time

    print(f"\n--- Simulation Complete ---")
    print(f"Total Queries: {rag_system.total_queries}")
    print(f"Cache Hits: {rag_system.cache_hits}")
    print(f"Cache Hit Rate: {(rag_system.cache_hits / rag_system.total_queries * 100):.2f}%")

    print("\nFinal GPU Cache Contents:")
    for key, entry in rag_system.cache.gpu_cache.items():
        print(f"  Key: {key}, Size: {entry.size}, Freq: {entry.frequency}, Recency: {rag_system.cache.current_clock - entry.last_accessed_clock}, Cost: {entry.cost:.2f}, Priority: {rag_system.cache._calculate_priority(entry):.2f}")

    print("\nFinal Host Cache Contents:")
    for key, entry in rag_system.cache.host_cache.items():
        print(f"  Key: {key}, Size: {entry.size}, Freq: {entry.frequency}, Recency: {rag_system.cache.current_clock - entry.last_accessed_clock}, Cost: {entry.cost:.2f}, Priority: {rag_system.cache._calculate_priority(entry):.2f}")