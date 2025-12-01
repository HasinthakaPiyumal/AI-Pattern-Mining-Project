from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List

app = FastAPI()

# --- Simulated KV Cache and Conversation History ---
# In a real scenario, this would store actual KV tensors (e.g., torch.Tensor or numpy.ndarray)
# For this simulation, we store the prefix text that the KV cache 'represents'.
# A more complex system would manage cache eviction, memory allocation, etc.
kv_cache: Dict[str, str] = {}

# Stores the full conversation history for each session
conversation_history: Dict[str, List[str]] = {}

# --- RAG System (Simulated) ---
# In a real system, this would involve vector databases, embedding models, etc.
def simulate_rag_retrieval(query: str) -> str:
    if "product features" in query.lower() or "specifications" in query.lower():
        return "\nRelevant Product Info: Our latest XYZ smartphone features a 6.7-inch OLED display, A17 Bionic chip, 128GB storage, and a 48MP camera system. It supports 5G connectivity and has a 4500mAh battery."
    elif "shipping status" in query.lower() or "delivery time" in query.lower():
        return "\nRelevant Shipping Info: Standard shipping usually takes 3-5 business days. You can track your order using the tracking number provided in your email confirmation. Express shipping is available for an additional fee."
    elif "return policy" in query.lower():
        return "\nRelevant Return Policy: Items can be returned within 30 days of purchase, provided they are in their original condition and packaging. Refunds are processed within 7-10 business days after the returned item is received."
    return "\nRelevant Info: We aim to provide excellent customer service. How can I further assist you today?"

# --- LLM Simulation (Simplified) ---
def simulate_llm_response(full_prompt: str, processed_segment: str = None) -> str:
    print(f"[LLM_SIMULATION] Full prompt to consider: '{full_prompt}'")
    if processed_segment:
        print(f"[LLM_SIMULATION] KV Cache reused. Only processing new segment: '{processed_segment}'")
    else:
        print(f"[LLM_SIMULATION] No KV Cache reuse. Processing entire prompt.")

    # Simple rule-based response for demonstration
    if "hello" in full_prompt.lower() or "hi" in full_prompt.lower():
        return "Hello! How can I assist you with our products or services today?"
    elif "product features" in full_prompt.lower():
        return "The XYZ smartphone offers advanced features like its powerful A17 Bionic chip and stunning OLED display. Is there a specific feature you'd like to know more about?"
    elif "shipping status" in full_prompt.lower():
        return "To check your shipping status, please provide your order number. Standard delivery typically takes 3-5 days."
    elif "return policy" in full_prompt.lower():
        return "Our return policy allows returns within 30 days. The item must be unused and in its original packaging."
    elif "thank you" in full_prompt.lower() or "thanks" in full_prompt.lower():
        return "You're welcome! Is there anything else I can help you with?"
    else:
        return "I'm sorry, I need more information to provide a helpful response. Could you please elaborate?"

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    user_message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    kv_cache_status: str

# --- FastAPI Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_message = request.user_message
    
    # Initialize conversation history for new session
    if session_id not in conversation_history:
        conversation_history[session_id] = []
        
    current_conversation_context = " ".join(conversation_history[session_id])
    
    # Simulate RAG for initial context or when new information is needed
    # In a real system, RAG might be more dynamically triggered.
    rag_context = simulate_rag_retrieval(user_message)
    
    # Construct the full prompt for the LLM
    # This includes historical turns and any RAG-retrieved context
    full_prompt = f"Customer: {current_conversation_context} {user_message}{rag_context}".strip()
    
    llm_processed_segment = None
    kv_status = "Not Used - Initial or New Context"
    
    # --- KV Cache Reuse Logic ---
    if session_id in kv_cache:
        cached_prefix = kv_cache[session_id]
        
        # Check if the current full prompt starts with the cached prefix
        if full_prompt.startswith(cached_prefix):
            # If it does, we can 'reuse' the KV cache for the prefix
            llm_processed_segment = full_prompt[len(cached_prefix):].strip()
            kv_status = f"Used - Reused '{cached_prefix[:50]}...'"
            print(f"[KV_CACHE] Reusing KV cache for session '{session_id}'. Processing only new segment.")
        else:
            # If the current prompt doesn't start with the cached prefix, 
            # it means the context has changed significantly or a new branch started.
            # In a real system, we might need to invalidate or update the cache more intelligently.
            print(f"[KV_CACHE] Cached prefix mismatch for session '{session_id}'. Invalidating/Updating cache.")
    
    # Simulate LLM inference
    if llm_processed_segment and llm_processed_segment != "":
        chatbot_response = simulate_llm_response(full_prompt, processed_segment=llm_processed_segment)
    else:
        chatbot_response = simulate_llm_response(full_prompt)
        
    # Update KV cache with the current full prompt (representing the KV tensors generated for it)
    # This assumes the full prompt represents the 