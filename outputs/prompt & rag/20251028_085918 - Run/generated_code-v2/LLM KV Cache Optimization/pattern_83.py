from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import random
import uuid
import asyncio

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    kv_cache_status: dict
    explanation: str

class KVCacheManager:
    def __init__(self, page_size: int, total_memory_pages: int):
        self.page_size = page_size
        self.total_memory_pages = total_memory_pages
        self.available_pages = set(range(total_memory_pages))
        self.allocated_blocks = {} 
        self.fragmentation_log = []

    def allocate_pages(self, request_id: str, num_pages: int) -> list[int] | None:
        if num_pages > len(self.available_pages):
            self.fragmentation_log.append(f"Request {request_id}: Memory exhaustion! Cannot allocate {num_pages} pages. Only {len(self.available_pages)} pages available. This scenario is significantly reduced by PagedAttention but can still occur under extreme load.")
            return None
        
        allocated = []
        for _ in range(num_pages):
            page_id = self.available_pages.pop()
            allocated.append(page_id)
        
        self.allocated_blocks[request_id] = allocated
        self.fragmentation_log.append(f"Request {request_id}: Allocated {num_pages} non-contiguous pages ({allocated}). Remaining available: {len(self.available_pages)}")
        return allocated

    def free_pages(self, request_id: str):
        if request_id in self.allocated_blocks:
            pages_to_free = self.allocated_blocks.pop(request_id)
            self.available_pages.update(pages_to_free)
            self.fragmentation_log.append(f"Request {request_id}: Freed {len(pages_to_free)} pages ({pages_to_free}). New available: {len(self.available_pages)}")
        else:
            self.fragmentation_log.append(f"Attempted to free pages for non-existent request_id: {request_id}")

    def get_status(self) -> dict:
        return {
            "total_pages": self.total_memory_pages,
            "available_pages_count": len(self.available_pages),
            "allocated_requests_count": len(self.allocated_blocks),
            "total_allocated_pages": sum(len(v) for v in self.allocated_blocks.values()),
            "fragmentation_notes": self.fragmentation_log[-5:] 
        }

class LLMServingCore:
    def __init__(self, kv_cache_manager: KVCacheManager, llm_model_name: str = "Conceptual Llama"):
        self.kv_cache_manager = kv_cache_manager
        self.llm_model_name = llm_model_name
        self.simulated_token_to_page_ratio = 4 

    async def process_query(self, query: str, request_id: str) -> tuple[str, str]:
        simulated_tokens = len(query.split()) + random.randint(5, 15) 
        
        num_kv_pages_needed = (simulated_tokens // self.simulated_token_to_page_ratio) + 1
        
        explanation_steps = []
        explanation_steps.append(f"Request {request_id}: User query '{query}' simulated to require ~{simulated_tokens} tokens.")
        explanation_steps.append(f"Based on token count, {num_kv_pages_needed} KV cache pages are requested from KVCacheManager.")

        allocated_pages = self.kv_cache_manager.allocate_pages(request_id, num_kv_pages_needed)

        if not allocated_pages:
            explanation_steps.append(f"Failed to allocate {num_kv_pages_needed} KV cache pages for request {request_id}. This indicates system-wide memory exhaustion, which PagedAttention significantly delays but cannot entirely prevent under extreme load.")
            return "Apologies, the system is currently under heavy load. Please try again shortly.", " ".join(explanation_steps)
        
        explanation_steps.append(f"Successfully allocated {len(allocated_pages)} KV cache pages ({allocated_pages}) for request {request_id}.")
        explanation_steps.append("PagedAttention ensures these pages can be non-contiguous in physical GPU memory, preventing external fragmentation and maximizing GPU memory utilization.")
        explanation_steps.append("These allocated pages would dynamically form the 'block table' for this request, mapping logical KV cache blocks to physical memory pages, much like virtual memory.")

        inference_time = simulated_tokens * 0.01 + random.uniform(0.05, 0.15)
        await asyncio.sleep(inference_time)
        
        simulated_llm_response = f"Hello! I am a conceptual e-commerce chatbot. For your query about '{query}', I found some relevant information. This response was generated using our highly optimized LLM serving layer, which leverages PagedAttention principles to efficiently manage memory for numerous concurrent requests like yours. The KV cache for your request took {len(allocated_pages)} pages." 
        
        self.kv_cache_manager.free_pages(request_id)
        explanation_steps.append(f"KV cache pages for request {request_id} have been freed after generating the response. This allows immediate reuse of memory for other requests, further enhancing throughput.")
        
        return simulated_llm_response, " ".join(explanation_steps)

app = FastAPI(
    title="E-commerce Chatbot with Conceptual PagedAttention",
    description="A conceptual FastAPI application demonstrating an AI customer support chatbot that highlights the benefits of PagedAttention for efficient KV cache management in LLM serving.",
    version="0.1.0"
)

CONCEPTUAL_PAGE_SIZE = 256 
TOTAL_CONCEPTUAL_MEMORY_PAGES = 500 

kv_cache_manager = KVCacheManager(
    page_size=CONCEPTUAL_PAGE_SIZE,
    total_memory_pages=TOTAL_CONCEPTUAL_MEMORY_PAGES
)
llm_serving_core = LLMServingCore(kv_cache_manager=kv_cache_manager)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    request_id = str(uuid.uuid4())
    print(f"Received request {request_id} for query: {request.query}")

    try:
        llm_response, explanation_text = await llm_serving_core.process_query(request.query, request_id)
        current_kv_status = kv_cache_manager.get_status()
        
        return ChatResponse(
            response=llm_response,
            kv_cache_status=current_kv_status,
            explanation=explanation_text
        )
    except Exception as e:
        print(f"Error processing request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/status")
async def get_system_status():
    return {
        "kv_cache_manager_status": kv_cache_manager.get_status(),
        "llm_serving_layer": {
            "model": llm_serving_core.llm_model_name,
            "simulated_token_to_page_ratio": llm_serving_core.simulated_token_to_page_ratio
        },
        "description": "This endpoint provides conceptual status of the PagedAttention-like KV cache management. 'fragmentation_notes' illustrate memory events."
    }