import time
import uuid
import heapq
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np

class KnowledgeBase:
    """
    Manages the document knowledge base, including embedding and retrieval.
    """
    def __init__(self, documents: list[str]):
        self.documents = documents
        # Using a smaller, faster embedding model for demonstration
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_embeddings = self.encoder.encode(documents, convert_to_tensor=False)
        self.doc_map = {i: doc for i, doc in enumerate(documents)}
        print("Knowledge Base initialized and documents embedded.")

    def retrieve(self, query: str, top_k: int = 2) -> tuple[list[str], list[int]]:
        """
        Retrieves the top_k most relevant documents for a given query.
        Returns the document texts and their original indices.
        """
        query_embedding = self.encoder.encode([query], convert_to_tensor=False)[0]
        # Calculate cosine similarity
        similarities = np.dot(query_embedding, self.doc_embeddings.T) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(self.doc_embeddings, axis=1)
        )
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        retrieved_docs = [self.doc_map[i] for i in top_k_indices]
        return retrieved_docs, top_k_indices

class RAGProcessor:
    """
    Simulates a RAG LLM processor with a simplified KV cache awareness.
    In a real system, a specialized inference server (e.g., vLLM) would manage the KV cache.
    """
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        # Using a small, readily available LLM for tokenization and *simulated* generation.
        # Actual generation is commented out to keep the demo lightweight.
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        # self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it") # Commented out for lighter demo
        
        self.kv_cache_hits = 0
        self.total_processed_requests = 0

        # SIMPLIFIED KV CACHE: We track recently used document *indices*.
        # A real KV cache would store actual key-value tensors for LLM attention.
        self.recent_kv_cache_docs = deque(maxlen=5) # Stores document indices of recently processed context

    def process_request(self, query: str, retrieved_docs_indices: list[int]) -> str:
        """
        Processes a RAG request, simulating LLM inference and KV cache interaction.
        """
        self.total_processed_requests += 1

        cached_length = 0
        computation_length = 0
        # retrieved_text = "\n".join([self.kb.doc_map[idx] for idx in retrieved_docs_indices])

        # Simulate cache hit/miss based on document indices
        for doc_idx in retrieved_docs_indices:
            doc_len = len(self.kb.doc_map[doc_idx]) # Approximate length by character count
            if doc_idx in self.recent_kv_cache_docs:
                cached_length += doc_len
                self.kv_cache_hits += 1
            else:
                computation_length += doc_len
        
        # Update simplified KV cache with current request's document indices
        for doc_idx in retrieved_docs_indices:
            if doc_idx not in self.recent_kv_cache_docs:
                self.recent_kv_cache_docs.append(doc_idx)

        # Simulate LLM inference time based on new computation needed
        # In a real system, you'd call model.generate()
        # prompt = f"Context: {retrieved_text}\n\nQuestion: {query}\n\nAnswer:"
        # input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        # output = self.model.generate(input_ids, max_new_tokens=100)
        # response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        time.sleep(0.05 + (computation_length / 2000.0)) # Base time + time proportional to new computation
        
        return f"Simulated RAG Response for '{query[:40]}...' (KB Docs: {retrieved_docs_indices}). Cached: {cached_length} chars, New Comp: {computation_length} chars."

class Request:
    """
    Represents an incoming RAG request with metadata for scheduling.
    """
    def __init__(self, query: str, request_id: str, arrival_time: float, retrieved_docs_indices: list[int]):
        self.query = query
        self.request_id = request_id
        self.arrival_time = arrival_time
        self.retrieved_docs_indices = retrieved_docs_indices
        self.order_priority = 0.0  # OrderPriority = Cached Length / Computation Length
        self.last_priority_update = arrival_time # For fairness window tracking

    def __lt__(self, other): # For min-heap, we want higher priority to be considered 'smaller'
        # Prioritize higher OrderPriority. If equal, prioritize older requests (FIFO).
        if self.order_priority != other.order_priority:
            return self.order_priority > other.order_priority # Max-heap behavior for priority
        return self.arrival_time < other.arrival_time # Min-heap behavior for arrival time (older first)

class CacheAwareRequestScheduler:
    """
    Manages incoming requests using a priority queue, reordering them based on cache awareness.
    """
    def __init__(self, rag_processor: RAGProcessor, fairness_window: float = 5.0):
        self.priority_queue = []  # A min-heap to store Request objects
        self.rag_processor = rag_processor
        self.fairness_window = fairness_window # Time in seconds after which a request's priority is boosted

    def _calculate_priority_metric(self, query: str, retrieved_docs_indices: list[int]) -> tuple[float, float]:
        """
        Calculates the OrderPriority metric (Cached Length / Computation Length).
        Returns the calculated priority and the computation length for logging/debugging.
        """
        cached_length = 0.0
        computation_length = 0.0

        for doc_idx in retrieved_docs_indices:
            doc_len = len(self.rag_processor.kb.doc_map[doc_idx])
            if doc_idx in self.rag_processor.recent_kv_cache_docs:
                cached_length += doc_len
            else:
                computation_length += doc_len

        if computation_length == 0:
            # If no new computation is needed (e.g., all context is cached), give highest priority.
            # If both are 0 (empty request/no docs), priority is 0.
            return float('inf') if cached_length > 0 else 0.0, computation_length

        priority = cached_length / computation_length
        return priority, computation_length

    def add_request(self, query: str):
        """
        Adds a new request to the scheduler, calculates its initial priority, and pushes it to the queue.
        """
        request_id = str(uuid.uuid4())
        arrival_time = time.time()
        
        # First, retrieve documents to determine cache potential
        _, retrieved_docs_indices = self.rag_processor.kb.retrieve(query)

        request = Request(query, request_id, arrival_time, retrieved_docs_indices)
        request.order_priority, _ = self._calculate_priority_metric(query, retrieved_docs_indices)
        
        heapq.heappush(self.priority_queue, request)
        print(f"[SCHEDULER] Added request {request.request_id[:4]} for '{query[:20]}...' with initial priority: {request.order_priority:.2f}")

    def process_next_request(self):
        """
        Pulls the highest priority request from the queue and processes it.
        Applies fairness boosting to old requests before popping.
        """
        if not self.priority_queue:
            # print("[SCHEDULER] No requests in queue. Waiting...")
            time.sleep(0.1) # Wait if queue is empty
            return

        # FAIRNESS WINDOW IMPLEMENTATION:
        # Periodically re-evaluate and boost priority for requests that have been waiting too long.
        current_time = time.time()
        re_heapify_needed = False
        for req in list(self.priority_queue): # Iterate over a copy if modifying during iteration
            if current_time - req.arrival_time > self.fairness_window and \
               current_time - req.last_priority_update > (self.fairness_window / 2): # Prevent boosting too frequently
                
                old_priority = req.order_priority
                # Simple boost: increase priority, ensuring it gets processed sooner.
                # A more sophisticated approach might add a time-based component to the original priority metric.
                req.order_priority = req.order_priority * 1.5 + 1.0 # Boost by a factor and add a constant
                req.last_priority_update = current_time
                print(f"[SCHEDULER] Fairness boost for {req.request_id[:4]}, old priority {old_priority:.2f} -> new {req.order_priority:.2f}")
                re_heapify_needed = True
        
        if re_heapify_needed:
            heapq.heapify(self.priority_queue) # Re-heapify if priorities were changed

        # Pop the highest priority request
        next_request = heapq.heappop(self.priority_queue)
        print(f"\n[SCHEDULER] Processing request {next_request.request_id[:4]} for '{next_request.query[:20]}...' (P: {next_request.order_priority:.2f}, Wait: {time.time() - next_request.arrival_time:.2f}s)")

        # Process the request using the RAG processor
        response = self.rag_processor.process_request(next_request.query, next_request.retrieved_docs_indices)
        print(f"[RAG] Finished request {next_request.request_id[:4]}. Output: {response[:100]}...")