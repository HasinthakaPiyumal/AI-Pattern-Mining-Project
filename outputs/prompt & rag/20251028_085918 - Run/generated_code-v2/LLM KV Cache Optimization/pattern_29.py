from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import time

# --- 1. FastAPI Setup ---
app = FastAPI(
    title="Smart E-commerce Chatbot with KV Cache Reuse",
    description="A chatbot leveraging KV Cache Reuse for optimized LLM inference."
)

# --- Data Models ---
class ChatRequest(BaseModel):
    session_id: str
    user_message: str

class ChatResponse(BaseModel):
    session_id: str
    bot_response: str
    kv_cache_status: str

# --- 2. LLM Serving Layer (Mocked vLLM with KV Cache Reuse Simulation) ---
class MockVLLM:
    def __init__(self):
        self.kv_cache: Dict[str, str] = {}
        self.model_name = "mock-llama-2-7b"

    async def generate(self, prompt: str, prefix_hint: Optional[str] = None) -> str:
        """Simulates LLM generation with KV cache reuse."""
        print(f"MockVLLM received prompt: {prompt[:100]}...")
        generation_time = 2.0 # Default generation time
        kv_cache_status = "cache_miss"
        
        if prefix_hint and prefix_hint in self.kv_cache:
            print(f"KV Cache HIT for prefix: {prefix_hint[:50]}...")
            # Simulate faster generation due to cache hit
            generation_time = 0.5
            kv_cache_status = "cache_hit"
        elif prefix_hint:
            print(f"KV Cache MISS for prefix: {prefix_hint[:50]}...")
            # Store prefix in cache (simplified: just store the prefix itself as a 'key')
            self.kv_cache[prefix_hint] = "cached_kv_tensors_placeholder"
            
        time.sleep(generation_time)
        
        # Simple response generation based on prompt
        if "product details for" in prompt.lower():
            product_name = prompt.lower().split("product details for")[-1].strip().split("\n")[0].strip()
            if product_name:
                return f"(Cached: {kv_cache_status}) Here are some details about {product_name}: It's a high-quality item with great reviews. Current price: $XX.XX. Available in multiple colors." 
        elif "shipping policy" in prompt.lower():
            return f"(Cached: {kv_cache_status}) Our shipping policy is free for orders over $50, otherwise a flat rate of $5 applies. Delivery typically takes 3-5 business days." 
        elif "return policy" in prompt.lower():
            return f"(Cached: {kv_cache_status}) Our return policy allows returns within 30 days of purchase with the original receipt and packaging." 
        elif "hello" in prompt.lower() or "hi" in prompt.lower():
            return f"(Cached: {kv_cache_status}) Hello! How can I assist you with your e-commerce needs today?"
        elif "thank you" in prompt.lower():
            return f"(Cached: {kv_cache_status}) You're welcome! Is there anything else I can help you with?"
        
        return f"(Cached: {kv_cache_status}) I'm not sure about that, but I can help you with product details, shipping, or returns. What else can I help you with?"

mock_vllm_service = MockVLLM()

# --- 3. Retrieval-Augmented Generation (RAG) Component ---
class RAGSystem:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="ecommerce_faq_products")
        self.populate_vector_db()

    def populate_vector_db(self):
        documents = [
            "Product A: A fantastic smartphone with a long-lasting battery and an amazing camera. Price: $799. Available in black and silver.",
            "Product B: A comfortable ergonomic office chair, perfect for long working hours. Features adjustable lumbar support. Price: $249.",
            "Shipping Policy: Free shipping on all orders over $50. Standard delivery takes 3-5 business days. Express shipping options are available at an extra cost.",
            "Return Policy: Items can be returned within 30 days of purchase, provided they are in their original condition with all packaging and tags. Refunds are processed within 7-10 business days.",
            "Payment Methods: We accept major credit cards (Visa, Mastercard, Amex), PayPal, and Apple Pay.",
            "Order Tracking: You can track your order using the tracking number provided in your shipping confirmation email on our website's 'Order Status' page."
        ]
        metadatas = [
            {"type": "product", "name": "Product A"},
            {"type": "product", "name": "Product B"},
            {"type": "policy", "name": "Shipping"},
            {"type": "policy", "name": "Returns"},
            {"type": "policy", "name": "Payment"},
            {"type": "support", "name": "Order Tracking"},
        ]
        ids = [f"doc_{i}" for i in range(len(documents))]

        if self.collection.count() == 0:
            self.collection.add(
                embeddings=self.embedding_model.encode(documents).tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"ChromaDB populated with {len(documents)} documents.")
        else:
            print(f"ChromaDB already contains {self.collection.count()} documents.")

    def retrieve_context(self, query: str, n_results: int = 2) -> List[str]:
        """Retrieves relevant documents from the vector database."""
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents']
        )
        return [doc for sublist in results['documents'] for doc in sublist]

rag_system = RAGSystem()

# --- 4. Prefix Management / Conversation History ---
conversation_history: Dict[str, List[str]] = {}

def identify_common_prefix(session_id: str, current_message: str) -> Optional[str]:
    """Identifies a common prefix between current message and recent history."""
    history = conversation_history.get(session_id, [])
    if not history:
        return None
    
    # For simplicity, check if the current message starts with any part of the previous bot response or user input
    # A more sophisticated approach would involve token-level analysis and a prefix tree
    
    last_interaction = history[-1]
    
    # Simple check: Does the current message start with a common phrase?
    common_phrases = [
        "what about", "tell me more about", "can you explain", 
        "and the", "how about the", "regarding the", "what is the"
    ]

    for phrase in common_phrases:
        if current_message.lower().startswith(phrase):
            # Take the last relevant piece of information from the history as a potential prefix hint
            # This is a very basic heuristic
            for i in range(len(history) -1, -1, -1):
                if "product details for" in history[i].lower() or \
                   "policy" in history[i].lower() or \
                   "about the" in history[i].lower():
                    return history[i] # Return the last informative turn as a prefix
            return None # No informative prefix found in history

    return None

# --- 5. FastAPI Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(chat_request: ChatRequest):
    session_id = chat_request.session_id
    user_message = chat_request.user_message
    
    # Update conversation history
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    conversation_history[session_id].append(f"User: {user_message}")

    # --- RAG Component Integration ---
    retrieved_context = []
    # Simple heuristic to decide if RAG is needed
    if any(keyword in user_message.lower() for keyword in ["product", "shipping", "return", "policy", "payment", "order tracking"]):
        retrieved_context = rag_system.retrieve_context(user_message)
        if retrieved_context:
            print(f"RAG retrieved context: {retrieved_context}")

    # Construct the prompt for the LLM
    full_prompt_parts = [
        "You are an AI customer support assistant for an e-commerce platform.",
        "Answer user questions based on the provided context and conversation history."
    ]
    
    if retrieved_context:
        full_prompt_parts.append("\n--- Context from E-commerce Knowledge Base ---")
        full_prompt_parts.extend(retrieved_context)
        full_prompt_parts.append("----------------------------------------------\n")
    
    full_prompt_parts.append("\n--- Conversation History ---")
    full_prompt_parts.extend(conversation_history[session_id])
    full_prompt_parts.append("---------------------------")
    full_prompt_parts.append(f"Assistant:")

    full_prompt = "\n".join(full_prompt_parts)

    # Identify common prefixes for KV cache reuse
    prefix_hint = identify_common_prefix(session_id, user_message)
    if prefix_hint:
        print(f"Identified potential prefix for KV cache: {prefix_hint[:50]}...")

    # Call the mocked vLLM service
    bot_response = await mock_vllm_service.generate(full_prompt, prefix_hint=prefix_hint)
    kv_cache_status = "cache_miss" # Default, will be updated by mock_vllm_service output
    
    if bot_response.startswith("(Cached: cache_hit)"):
        kv_cache_status = "cache_hit"
    
    # Clean up the status tag for the final response
    bot_response = bot_response.replace("(Cached: cache_hit) ", "").replace("(Cached: cache_miss) ", "")

    # Update history with bot's response
    conversation_history[session_id].append(f"Assistant: {bot_response}")

    return ChatResponse(
        session_id=session_id,
        bot_response=bot_response,
        kv_cache_status=kv_cache_status
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Chatbot service is running."}

# Example usage (if running directly for testing, not through uvicorn)
if __name__ == "__main__":
    import uvicorn
    # To run: uvicorn smart_chatbot:app --reload
    print("Starting Smart E-commerce Chatbot API. Access at http://127.0.0.1:8000")
    print("To interact, send POST requests to /chat with session_id and user_message.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
