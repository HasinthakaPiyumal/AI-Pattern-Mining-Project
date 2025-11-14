import uuid
import hashlib
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

# --- 1. Session Manager (Conceptual: session_manager.py) ---
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {'history': [], 'kv_cache_id': None} # kv_cache_id is for the *current* longest cached prefix
        print(f"SessionManager: Created new session {session_id}")
        return session_id

    def get_session_data(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get(session_id)

    def update_session_history(self, session_id: str, user_message: str, bot_response: str):
        if session_id in self.sessions:
            self.sessions[session_id]['history'].append({"user": user_message, "bot": bot_response})
            print(f"SessionManager: Updated history for session {session_id}")
        else:
            print(f"SessionManager: Session {session_id} not found for history update.")

    def update_kv_cache_id(self, session_id: str, kv_cache_id: str):
        if session_id in self.sessions:
            self.sessions[session_id]['kv_cache_id'] = kv_cache_id
            print(f"SessionManager: Stored/Updated KV cache ID for session {session_id} to {kv_cache_id}")
        else:
            print(f"SessionManager: Session {session_id} not found for KV cache ID update.")

# --- 2. RAG Service (Conceptual: rag_service.py) ---
class RAGService:
    def __init__(self):
        self.knowledge_base = {
            "order status": "To check your order status, please visit our 'Order Tracking' page and enter your order number.",
            "return policy": "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition. Please see our website for full details.",
            "shipping times": "Standard shipping usually takes 5-7 business days. Expedited shipping options are available at checkout.",
            "contact support": "You can contact our support team via email at support@example.com or by calling 1-800-555-0123 during business hours.",
            "payment methods": "We accept all major credit cards, PayPal, and Apple Pay.",
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! What can I do for you?"
        }
        print("RAGService: Knowledge base initialized.")

    def retrieve_info(self, query: str) -> Optional[str]:
        query_lower = query.lower()
        for keyword, info in self.knowledge_base.items():
            if keyword in query_lower:
                print(f"RAGService: Retrieved info for keyword '{keyword}' related to query: '{query}'")
                return info
        print(f"RAGService: No relevant info found for query: '{query}'")
        return None

# --- 3. LLM Service with KV Cache (Conceptual: llm_service.py) ---
class LLMService:
    def __init__(self):
        self.kv_cache: Dict[str, str] = {} # Key: hash of prefix string, Value: simulated KV tensors
        print("LLMService: KV cache initialized.")

    def _get_conversation_prefix_string(self, history_segment: List[Dict]) -> str:
        # Concatenate user and bot messages to form a unique string representing the conversation prefix
        return " ".join([f"User: {entry['user']} Bot: {entry['bot']}" for entry in history_segment])

    def generate_response(self, session_id: str, prompt: str, conversation_history: List[Dict]) -> (str, str):
        
        longest_matching_prefix_hash = None
        
        # Iterate from the longest possible prefix (current full conversation) down to empty
        for i in range(len(conversation_history), -1, -1):
            prefix_segment_for_hashing = conversation_history[:i]
            
            history_prefix_string = self._get_conversation_prefix_string(prefix_segment_for_hashing)
            
            if history_prefix_string:
                prefix_hash = hashlib.sha256(history_prefix_string.encode('utf-8')).hexdigest()
                if prefix_hash in self.kv_cache:
                    longest_matching_prefix_hash = prefix_hash
                    print(f"LLMService: Found longest matching prefix in cache with hash: {longest_matching_prefix_hash}, length: {i} turns.")
                    break # Found the longest one, so break
        
        response_prefix = ""
        final_kv_cache_id_for_this_turn = None # This will be the hash of the *new* full context if computed, or the reused prefix hash.

        # Decide if we reuse or compute new KV tensors
        if longest_matching_prefix_hash:
            print(f"LLMService: Reusing KV cache for session {session_id} with prefix hash {longest_matching_prefix_hash}")
            time.sleep(0.1) # Simulate faster inference due to cache reuse
            response_prefix = " (Cached Prefix) "
            final_kv_cache_id_for_this_turn = longest_matching_prefix_hash
        else:
            # No existing prefix found, so compute new KV tensors for the entire current context
            # This includes the entire conversation history + the current user prompt.
            # We'll create a hash for this entire context and cache it.
            context_for_new_cache_string = self._get_conversation_prefix_string(conversation_history) + f" User: {prompt}"
            new_cache_hash = hashlib.sha256(context_for_new_cache_string.encode('utf-8')).hexdigest()

            print(f"LLMService: Computing new KV tensors for session {session_id} for new full context hash {new_cache_hash}")
            time.sleep(0.5) # Simulate longer inference for new computation
            self.kv_cache[new_cache_hash] = f"Simulated_KV_Tensors_for_{new_cache_hash}"
            response_prefix = " (New Computation) "
            final_kv_cache_id_for_this_turn = new_cache_hash

        # Simulate LLM response generation based on the prompt
        bot_response_content = "I'm not sure how to respond to that. Can you please rephrase or ask something else?"
        prompt_lower = prompt.lower()

        if "hello" in prompt_lower or "hi" in prompt_lower:
            bot_response_content = "Hello! How can I assist you today?"
        elif "thank you" in prompt_lower:
            bot_response_content = "You're welcome! Is there anything else I can help you with?"
        elif "order" in prompt_lower:
            bot_response_content = "I can help with order related queries. Please provide your order number."
        elif "return" in prompt_lower:
            bot_response_content = "I can provide details about our return policy."
        elif "shipping" in prompt_lower:
            bot_response_content = "I can give you information on shipping times and options."
        elif "support" in prompt_lower or "contact" in prompt_lower:
            bot_response_content = "Our support team is ready to help! How would you like to reach them?"
        elif "information:" in prompt_lower: # If RAG context was added
            # This is a very simplified way to simulate using RAG context.
            # A real LLM would integrate the context into its generation process.
            # Here, we just acknowledge the RAG context.
            rag_info_start = prompt_lower.find("information:") + len("information:")
            rag_info_end = prompt_lower.find(", answer this:")
            if rag_info_start != -1 and rag_info_end != -1:
                retrieved_info = prompt[rag_info_start:rag_info_end].strip()
                bot_response_content = f"I have the information regarding {retrieved_info}. What specifically would you like to know or do next?"
            else:
                bot_response_content = "I've processed the information provided. How can I elaborate on it?"
        
        return response_prefix + bot_response_content, final_kv_cache_id_for_this_turn

# --- 4. FastAPI Backend API (Conceptual: main.py) ---
app = FastAPI(title="AI Customer Support Chatbot with KV Cache Reuse")

session_manager = SessionManager()
rag_service = RAGService()
llm_service = LLMService()

class ChatRequest(BaseModel):
    user_message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    bot_response: str
    history: List[Dict]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    user_message = request.user_message

    if not session_id:
        session_id = session_manager.create_session()
        print(f"FastAPI: No session_id provided, created new session: {session_id}")
    
    session_data = session_manager.get_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    conversation_history = session_data['history']
    
    # 5. FastAPI checks if RAG is needed. If so, calls RAGService.retrieve_info.
    rag_context = rag_service.retrieve_info(user_message)
    
    # 6. FastAPI constructs the full prompt (user message + history + RAG context).
    current_llm_prompt = user_message
    if rag_context:
        current_llm_prompt = f"Based on the following information: '{rag_context}', answer this: {user_message}"
    
    print(f"FastAPI: Calling LLMService with prompt: '{current_llm_prompt}' and history length: {len(conversation_history)}")
    
    # 7. FastAPI calls LLMService.generate_response, passing the session ID, prompt, and history.
    bot_response, final_kv_cache_id_for_this_turn = llm_service.generate_response(session_id, current_llm_prompt, conversation_history)
    
    # 10. FastAPI updates the session history and kv_cache_id in SessionManager.
    session_manager.update_session_history(session_id, user_message, bot_response)
    session_manager.update_kv_cache_id(session_id, final_kv_cache_id_for_this_turn)
    
    updated_history = session_manager.get_session_data(session_id)['history']
    
    # 11. FastAPI sends the bot's response back to the CLI.
    return ChatResponse(session_id=session_id, bot_response=bot_response, history=updated_history)