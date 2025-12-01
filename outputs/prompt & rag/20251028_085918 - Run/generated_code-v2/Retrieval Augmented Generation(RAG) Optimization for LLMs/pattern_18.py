import asyncio
import heapq
import time
import uuid
from collections import deque
from typing import List, Dict, Set, Optional, Tuple

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class MockFAISS:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.chunk_ids = []
        self.index_built = False

    def add(self, embeddings: List[List[float]], chunk_ids: List[str], texts: List[str]):
        if not self.index_built:
            self.embeddings.extend(embeddings)
            self.chunk_ids.extend(chunk_ids)
            self.documents.extend(texts)
            if len(self.embeddings) > 0:
                self.index_built = True

    def search(self, query_embedding: List[float], k: int) -> Tuple[List[str], List[float]]:
        if not self.index_built or not self.embeddings:
            return [], []

        query_tensor = torch.tensor(query_embedding).unsqueeze(0)
        doc_embeddings_tensor = torch.tensor(self.embeddings)

        similarities = torch.nn.functional.cosine_similarity(query_tensor, doc_embeddings_tensor)
        top_k_indices = torch.topk(similarities, min(k, len(similarities))).indices.tolist()

        retrieved_chunk_ids = [self.chunk_ids[i] for i in top_k_indices]
        retrieved_scores = [similarities[i].item() for i in top_k_indices]
        return retrieved_chunk_ids, retrieved_scores

    def get_document_text(self, chunk_id: str) -> Optional[str]:
        try:
            idx = self.chunk_ids.index(chunk_id)
            return self.documents[idx]
        except ValueError:
            return None


class CustomerRequest(BaseModel):
    id: str
    query: str
    retrieved_chunk_ids: List[str]
    timestamp: float
    order_priority: float = 0.0

    def __lt__(self, other):
        return self.order_priority > other.order_priority


class CacheAwareScheduler:
    def __init__(self, fairness_window_threshold: int = 5, min_computation_length: float = 0.1):
        self._priority_queue: List[CustomerRequest] = []
        self._fairness_window: deque[CustomerRequest] = deque()
        self._kv_cache: Set[str] = set()
        self._processed_count: int = 0
        self.FAIRNESS_WINDOW_THRESHOLD = fairness_window_threshold
        self.MIN_COMPUTATION_LENGTH = min_computation_length
        self._waiting_requests_by_id: Dict[str, CustomerRequest] = {}

    def _calculate_priority(self, request: CustomerRequest) -> float:
        cached_length = len(self._kv_cache.intersection(set(request.retrieved_chunk_ids)))
        total_length = len(request.retrieved_chunk_ids)
        computation_length = total_length - cached_length
        
        if computation_length <= 0:
            return float('inf')
        
        priority = cached_length / computation_length
        return priority

    def add_request(self, request: CustomerRequest):
        request.order_priority = self._calculate_priority(request)
        heapq.heappush(self._priority_queue, request)
        self._waiting_requests_by_id[request.id] = request
        self._fairness_window.append(request)

    def get_next_request(self) -> Optional[CustomerRequest]:
        selected_request: Optional[CustomerRequest] = None
        
        if self._processed_count % self.FAIRNESS_WINDOW_THRESHOLD == 0 and self._fairness_window:
            temp_fairness_window = deque()
            while self._fairness_window:
                candidate = self._fairness_window.popleft()
                if candidate.id in self._waiting_requests_by_id:
                    selected_request = candidate
                    break
                else:
                    pass
            while self._fairness_window:
                temp_fairness_window.append(self._fairness_window.popleft())
            self._fairness_window = temp_fairness_window


        if selected_request is None:
            while self._priority_queue:
                candidate = heapq.heappop(self._priority_queue)
                if candidate.id in self._waiting_requests_by_id:
                    selected_request = candidate
                    break

        if selected_request:
            del self._waiting_requests_by_id[selected_request.id]
            self._processed_count += 1
            return selected_request
        
        return None

    def update_kv_cache(self, processed_request: CustomerRequest):
        self._kv_cache.update(set(processed_request.retrieved_chunk_ids))
        
    def reset_cache(self):
        self._kv_cache.clear()
        self._processed_count = 0
        self._fairness_window.clear()
        self._priority_queue.clear()
        self._waiting_requests_by_id.clear()


class RAGSystem:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm_tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
        self.llm_model = AutoModelForCausalLM.from_pretrained('distilgpt2')
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.document_store = MockFAISS()
        self._initialize_document_store()

    def _initialize_document_store(self):
        documents_data = [
            ("doc_1", "Our return policy states that items can be returned within 30 days of purchase, provided they are in their original condition with a receipt."),
            ("doc_2", "To track your order, please visit our website and enter your order number in the 'Track Order' section."),
            ("doc_3", "Shipping usually takes 3-5 business days for standard delivery. Expedited shipping options are available at checkout."),
            ("doc_4", "You can contact customer support via live chat on our website or by calling us at 1-800-SHOP-NOW."),
            ("doc_5", "Our loyalty program offers exclusive discounts and early access to sales for members. Sign up on our homepage."),
            ("doc_6", "Product warranty covers manufacturing defects for 1 year from the purchase date. Accidental damage is not covered."),
            ("doc_7", "We accept major credit cards (Visa, Mastercard, Amex) and PayPal for payments."),
            ("doc_8", "If an item is out of stock, you can sign up for email notifications to be alerted when it's back."),
            ("doc_9", "Our return policy for electronics is 15 days, sealed and unused."),
            ("doc_10", "How to track my order?"),
        ]
        
        texts = [data[1] for data in documents_data]
        chunk_ids = [data[0] for data in documents_data]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        self.document_store.add(embeddings, chunk_ids, texts)
        print(f"Initialized RAGSystem with {len(chunk_ids)} document chunks.")

    def retrieve_documents(self, query: str, k: int = 3) -> List[str]:
        query_embedding = self.embedding_model.encode(query).tolist()
        retrieved_chunk_ids, _ = self.document_store.search(query_embedding, k=k)
        return retrieved_chunk_ids

    def generate_response(self, query: str, context_chunk_ids: List[str]) -> str:
        context_texts = [self.document_store.get_document_text(cid) for cid in context_chunk_ids if self.document_store.get_document_text(cid) is not None]
        context = "\n".join(context_texts)
        
        if not context:
            input_text = f"Customer query: {query}. Assistant:"
        else:
            input_text = f"Context: {context}\nCustomer query: {query}. Assistant:"
        
        inputs = self.llm_tokenizer(input_text, return_tensors='pt')
        
        outputs = self.llm_model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            attention_mask=inputs.attention_mask
        )
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_start_index = response.find("Assistant:")
        if response_start_index != -1:
            response = response[response_start_index + len("Assistant:"):]
        else:
            response = response
            
        response = response.strip()
            
        if not response:
            response = "I'm sorry, I couldn't find a specific answer to that. Please try rephrasing your question or contact our live support."

        time.sleep(0.5 + len(response) * 0.005)
        return response


app = FastAPI()
rag_system: RAGSystem
scheduler: CacheAwareScheduler
request_results: Dict[str, str] = {}
_processing_task: Optional[asyncio.Task] = None


@app.on_event("startup")
async def startup_event():
    global rag_system, scheduler, _processing_task
    print("Initializing RAG System and Scheduler...")
    rag_system = RAGSystem()
    scheduler = CacheAwareScheduler(fairness_window_threshold=3)
    _processing_task = asyncio.create_task(process_requests_background())
    print("RAG System and Scheduler initialized. Background processing started.")


@app.on_event("shutdown")
async def shutdown_event():
    global _processing_task
    if _processing_task:
        _processing_task.cancel()
        try:
            await _processing_task
        except asyncio.CancelledError:
            print("Background processing task cancelled.")


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def handle_query(query_request: QueryRequest):
    request_id = str(uuid.uuid4())
    current_time = time.time()

    print(f"Received query [{request_id}]: {query_request.query}")

    retrieved_chunk_ids = rag_system.retrieve_documents(query_request.query)
    print(f"[{request_id}] Retrieved chunks: {retrieved_chunk_ids}")

    customer_req = CustomerRequest(
        id=request_id,
        query=query_request.query,
        retrieved_chunk_ids=retrieved_chunk_ids,
        timestamp=current_time
    )
    scheduler.add_request(customer_req)

    return {"request_id": request_id, "status": "queued"}


@app.get("/result/{request_id}")
async def get_result(request_id: str):
    if request_id in request_results:
        response = request_results.pop(request_id)
        return {"request_id": request_id, "status": "completed", "response": response}
    else:
        return {"request_id": request_id, "status": "processing", "response": None}


async def process_requests_background():
    global request_results
    while True:
        try:
            await asyncio.sleep(0.1)

            next_request = scheduler.get_next_request()
            if next_request:
                print(f"[{next_request.id}] Processing request (Priority: {next_request.order_priority:.2f}). Query: {next_request.query[:50]}...")
                
                response = rag_system.generate_response(next_request.query, next_request.retrieved_chunk_ids)
                
                scheduler.update_kv_cache(next_request)
                request_results[next_request.id] = response
                print(f"[{next_request.id}] Finished processing. KV Cache size: {len(scheduler._kv_cache)}")
            else:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            print("Background processing task received cancellation signal.")
            break
        except Exception as e:
            print(f"Error in background processing task: {e}")
            await asyncio.sleep(1)
