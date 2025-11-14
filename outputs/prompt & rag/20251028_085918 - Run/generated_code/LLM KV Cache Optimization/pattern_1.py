
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import os
import time

# Mock vLLM client for demonstration. In a real scenario, you'd use vllm.completion.Completion or similar.
# vLLM inherently handles KV Cache Reuse.
class MockVLLMClient:
    def __init__(self, model: str = "mock-llm"):
        self.model = model
        self.kv_cache_hits = 0
        self.kv_cache_misses = 0
        print(f"MockVLLMClient initialized for model: {model}. KV Cache Reuse enabled internally.")

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> str:
        # Simulate LLM generation with some delay and KV cache hit/miss logic
        print(f"[VLLM Mock] Generating response for: \'{prompt[:50]}...\' (Model: {self.model})")
        time.sleep(0.05) # Simulate processing time

        # Simple heuristic for demonstration of KV cache reuse for common prefixes
        # In a real vLLM setup, this is handled automatically based on token IDs and prefix trees.
        common_prefixes = ["hello", "hi", "what is", "tell me about"]
        is_kv_cache_hit = False
        for prefix in common_prefixes:
            if prompt.lower().startswith(prefix):
                is_kv_cache_hit = True
                break
        
        # Also check if the conversation history implies a shared prefix
        if len(prompt.split()) > 10 and any(turn["user"] in prompt for turn in self._last_conversation_history):
            is_kv_cache_hit = True

        if is_kv_cache_hit:
            self.kv_cache_hits += 1
            print("[VLLM Mock] KV Cache Hit (simulated for common prefix or history reuse)")
        else:
            self.kv_cache_misses += 1
            print("[VLLM Mock] KV Cache Miss (simulated)")
        
        self._last_conversation_history = self._extract_history(prompt) # Update history for next simulation

        if "what is your name" in prompt.lower():
            return "I am a customer support AI assistant."
        if "features of product a" in prompt.lower():
            return "Product A has features X, Y, and Z. It costs $100."
        if "price of product a" in prompt.lower():
            return "Product A costs $100."
        return f"This is a simulated response to: \'{prompt}\'. I can assist with product queries. (KV {'Hit' if is_kv_cache_hit else 'Miss'})"

    def _extract_history(self, prompt: str) -> List[Dict[str, str]]:
        # Very simplistic extraction for mock history tracking
        history = []
        lines = prompt.split('\n')
        for line in lines:
            if line.startswith("User: "):
                history.append({"user": line[len("User: "):]})
            elif line.startswith("Assistant: "):
                if history: # If there's a user turn before it
                    history[-1]["assistant"] = line[len("Assistant: "):]
        return history

# Mock LlamaIndex for RAG (Retrieval Augmented Generation)
class MockLlamaIndex:
    def __init__(self, vector_store_name: str = "Chroma"):
        self.vector_store_name = vector_store_name
        self.knowledge_base = {
            "product_a": {
                "features": "Product A has features X, Y, and Z. It is known for its high durability and sleek design.",
                "price": "$100",
                "warranty": "1-year limited warranty."
            },
            "shipping_policy": "Standard shipping takes 3-5 business days. Express shipping is available for an extra fee. International shipping rates vary.",
            "return_policy": "Returns are accepted within 30 days of purchase with a valid receipt. Items must be in original condition."
        }
        print(f"MockLlamaIndex initialized with {vector_store_name} for RAG.")

    def retrieve_context(self, query: str) -> List[str]:
        context = []
        query_lower = query.lower()

        if "product a" in query_lower:
            if "features" in query_lower:
                context.append(self.knowledge_base["product_a"]["features"])
            elif "price" in query_lower:
                context.append(self.knowledge_base["product_a"]["price"])
            elif "warranty" in query_lower:
                context.append(self.knowledge_base["product_a"]["warranty"])
            else:
                context.append(f"Product A Information: {self.knowledge_base['product_a']['features']} Price: {self.knowledge_base['product_a']['price']}.")
        
        if "shipping" in query_lower:
            context.append(self.knowledge_base["shipping_policy"])
        if "return" in query_lower:
            context.append(self.knowledge_base["return_policy"])
        
        return context

    def query_with_rag(self, user_query: str, llm_client: MockVLLMClient, conversation_history: List[Dict[str, str]]) -> str:
        print(f"[LlamaIndex Mock] Performing RAG for query: \'{user_query}\'")
        retrieved_context = self.retrieve_context(user_query)
        
        rag_prompt_parts = []
        for turn in conversation_history:
            rag_prompt_parts.append(f"User: {turn['user']}")
            rag_prompt_parts.append(f"Assistant: {turn['assistant']}")
        
        if retrieved_context:
            rag_prompt_parts.append(f"Context from knowledge base: {' '.join(retrieved_context)}")
            rag_prompt_parts.append(f"User's current question: {user_query}")
            rag_prompt_parts.append("Based on the context and conversation, answer the user's question concisely.")
        else:
            rag_prompt_parts.append(f"User: {user_query}")
            rag_prompt_parts.append("No specific context found, generate a helpful general response.")
            
        full_rag_prompt = "\n".join(rag_prompt_parts)
        return llm_client.generate(full_rag_prompt)


app = FastAPI(title="Customer Support Chatbot with KV Cache Reuse")

# Initialize VLLM and LlamaIndex mocks
vllm_client = MockVLLMClient()
lama_index_rag = MockLlamaIndex(vector_store_name="MockChroma")

# Internal storage for mock history in VLLM (for better KV cache simulation)
vllm_client._last_conversation_history = []

# --- Pydantic Models ---
class Message(BaseModel):
    user: str
    assistant: str

class ChatRequest(BaseModel):
    user_message: str
    conversation_history: List[Message] = [] # For multi-turn context

class ChatResponse(BaseModel):
    response: str
    kv_cache_stats: Dict[str, int]
    
@app.on_event("startup")
async def startup_event():
    print("Starting up customer support chatbot service...")

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    print(f"\nReceived chat request: \'{request.user_message}\'")

    # Decide whether to use RAG based on keywords
    use_rag = any(keyword in request.user_message.lower() for keyword in ["product", "shipping", "return", "price", "warranty", "features"])

    if use_rag:
        llm_response = lama_index_rag.query_with_rag(request.user_message, vllm_client, request.conversation_history)
    else:
        # Construct the full prompt including conversation history for direct LLM interaction
        full_prompt_parts = []
        for turn in request.conversation_history:
            full_prompt_parts.append(f"User: {turn.user}")
            full_prompt_parts.append(f"Assistant: {turn.assistant}")
        full_prompt_parts.append(f"User: {request.user_message}")
        
        full_prompt = "\n".join(full_prompt_parts)
        llm_response = vllm_client.generate(full_prompt)

    return ChatResponse(
        response=llm_response,
        kv_cache_stats={
            "hits": vllm_client.kv_cache_hits,
            "misses": vllm_client.kv_cache_misses
        }
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "vllm_mock_status": "running", "kv_cache_hits": vllm_client.kv_cache_hits, "kv_cache_misses": vllm_client.kv_cache_misses}

# To run the FastAPI application using uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# How to run this application:
# 1. Save the code as `chatbot_service.py`
# 2. Install necessary libraries: `pip install fastapi uvicorn pydantic`
# 3. Run from your terminal: `uvicorn chatbot_service:app --reload --port 8000`
# 4. Access the API at http://127.0.0.1:8000/docs for Swagger UI.
#    Example POST request to /chat (initial query):
#    {
#        "user_message": "Hello, what are the features of Product A?",
#        "conversation_history": []
#    }
#    Subsequent request with shared prefix (or part of the same conversation):
#    {
#        "user_message": "What about its price?",
#        "conversation_history": [
#            {"user": "Hello, what are the features of Product A?", "assistant": "Product A has features X, Y, and Z. It costs $100. (KV Hit)"}
#        ]
#    }
#    Observe the "KV Cache Hit" or "KV Cache Miss" in the console output and the `kv_cache_stats` in the response.
