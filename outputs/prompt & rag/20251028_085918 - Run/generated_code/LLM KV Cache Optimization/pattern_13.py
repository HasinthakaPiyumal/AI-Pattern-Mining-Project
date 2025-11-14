"""Python code for an Intelligent Customer Support Chatbot using FastAPI and vLLM (mocked for demonstration).

This application demonstrates the architectural concepts of:
- KV Cache Reuse (inherent in vLLM)
- PagedAttention (inherent in vLLM)
- Replication of Critical KV Cache Nodes (simulated)
- Swap-Out-Only-Once Cache Strategy (inherent in vLLM's memory management)
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str  # e.g., "user", "assistant"
    content: str

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    user_id: str
    response: str
    context_replicated: bool = False

# --- Global State / Simulated Stores ---
app = FastAPI(
    title="Intelligent Customer Support Chatbot",
    description="Chatbot leveraging advanced LLM inference optimizations."
)

# Stores full conversation history for each user
conversation_history: Dict[str, List[ChatMessage]] = {}

# Simulates a persistent store for critical KV cache nodes/prefixes
# In a real system, this would be a robust database (e.g., Redis, Cassandra)
# that survives application restarts.
critical_kv_cache_replication_store: Dict[str, str] = {}

# --- vLLM Mock (for demonstration without a GPU and vLLM setup) ---
class MockAsyncLLM:
    def __init__(self, model: str = "mock-llama-model"):
        self.model = model
        print(f"Mock LLM initialized with model: {self.model}")

    async def generate(self, prompts: List[str], 
                       sampling_params: Optional[Dict] = None, 
                       request_id: Optional[str] = None) -> List[Dict]:
        """Mock vLLM's generate method."""
        print(f"Mock LLM received prompts: {prompts}")
        print(f"Mock LLM sampling_params: {sampling_params}")
        
        # Simulate a delay for inference
        await asyncio.sleep(0.5)
        
        responses = []
        for i, prompt in enumerate(prompts):
            # Simple rule-based mock response
            if "hello" in prompt.lower() or "hi" in prompt.lower():
                mock_output = "Hello! How can I assist you with your shopping today?"
            elif "order" in prompt.lower() and "status" in prompt.lower():
                mock_output = "Please provide your order number and I can check its status for you."
            elif "product" in prompt.lower() and "return" in prompt.lower():
                mock_output = "Our return policy allows returns within 30 days of purchase. Would you like to initiate a return?"
            else:
                mock_output = f"I received your message about '{prompt[-50:]}...'. How else can I help?"
                
            responses.append({
                "request_id": request_id if request_id else f"mock_req_{i}",
                "prompt": prompt,
                "outputs": [{
                    "text": mock_output,
                    "token_ids": [], # Empty for mock
                    "logprob": 0.0, # Empty for mock
                    "finish_reason": "stop" # Empty for mock
                }],
                "finished": True, 
                "prompt_token_ids": [] # Empty for mock
            })
        return responses


# Initialize the (mocked) vLLM instance globally
# In a real scenario, this would be: llm = vllm.AsyncLLM(model="meta-llama/Llama-2-7b-chat-hf", 
#                                              trust_remote_code=True, 
#                                              enforce_eager=True)
llm = MockAsyncLLM()

# --- Helper Functions ---
def build_prompt_from_history(history: List[ChatMessage], current_message: str) -> str:
    """Constructs a conversational prompt from history for the LLM."""
    # A simple way to format conversational history for a chat LLM.
    # For more complex models, specific tokenizers and prompt templates would be used.
    prompt_parts = []
    for msg in history:
        prompt_parts.append(f"{msg.role.capitalize()}: {msg.content}")
    prompt_parts.append(f"User: {current_message}")
    prompt_parts.append("Assistant:") # Prompt the model to generate assistant's response
    return "\n".join(prompt_parts)


# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message
    
    # Initialize conversation history if it's a new user
    if user_id not in conversation_history:
        conversation_history[user_id] = []
        
    current_user_history = conversation_history[user_id]
    
    # --- Simulation of Replication of Critical KV Cache Nodes ---
    # Identify critical prefix (e.g., the very first user message or a key context setter)
    context_replicated_flag = False
    if not current_user_history: # First message of a new conversation
        critical_prefix = f"Initial Query from User {user_id}: {user_message}"
        critical_kv_cache_replication_store[user_id] = critical_prefix
        print(f"[CRITICAL KV CACHE] Stored critical prefix for user {user_id}: '{critical_prefix}'")
        context_replicated_flag = True
    elif len(current_user_history) == 2 and "order" in user_message.lower(): # Example: after a product query, if user asks about an order
         critical_prefix = f"User {user_id} now discussing order: {user_message}"
         critical_kv_cache_replication_store[user_id] = critical_prefix
         print(f"[CRITICAL KV CACHE] Updated critical prefix for user {user_id}: '{critical_prefix}'")
         context_replicated_flag = True
    
    # In a fault recovery scenario, if the LLM restarted, we would re-feed
    # `critical_kv_cache_replication_store[user_id]` as a prompt prefix
    # to vLLM to warm up its cache with this critical context.
    
    # Add user's message to history
    current_user_history.append(ChatMessage(role="user", content=user_message))
    
    # Construct the full prompt for the LLM
    full_prompt = build_prompt_from_history(current_user_history, user_message)
    
    # --- vLLM Inference (Mocked) ---
    # vLLM inherently handles KV Cache Reuse, PagedAttention, and 
    # contributes to Swap-Out-Only-Once via its efficient memory management.
    
    # Define sampling parameters (can be customized per request)
    sampling_params = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 150}
    
    try:
        # In a real vLLM setup, you might pass a list of prompts if batching.
        # For a single chat turn, we send one prompt.
        llm_output = await llm.generate(
            prompts=[full_prompt],
            sampling_params=sampling_params,
            request_id=user_id # Use user_id as request_id for potential tracing/identification
        )
        
        # Extract the generated text from the mock output
        if llm_output and llm_output[0] and llm_output[0].get("outputs"): 
            assistant_response_content = llm_output[0]["outputs"][0]["text"].strip()
        else:
            assistant_response_content = "I'm sorry, I couldn't generate a response at this moment."
            
    except Exception as e:
        print(f"Error during LLM inference: {e}")
        assistant_response_content = "I apologize, but I'm experiencing technical difficulties. Please try again later."
    
    # Add assistant's response to history
    current_user_history.append(ChatMessage(role="assistant", content=assistant_response_content))
    
    return ChatResponse(
        user_id=user_id,
        response=assistant_response_content,
        context_replicated=context_replicated_flag
    )

# --- Run the FastAPI application ---
if __name__ == "__main__":
    import uvicorn
    # To run: uvicorn customer_chatbot_app:app --reload --port 8000
    # Access at: http://127.0.0.1:8000/docs
    uvicorn.run(app, host="0.0.0.0", port=8000)
