from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
import redis
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str

class MockLLM:
    def __init__(self):
        self.model_name = "MockLLM"

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def detokenize(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def generate_response(self, prompt: str, cached_kv_tensors: str = None) -> (str, str):
        logging.info(f"LLM received prompt: '{prompt}'")
        if cached_kv_tensors:
            logging.info(f"LLM reusing KV cache: '{cached_kv_tensors}'")
            base_response = f"(Cached response based on {cached_kv_tensors}) "
        else:
            base_response = ""
        
        # Simulate LLM generation
        response_text = f"{base_response}Bot's reply to '{prompt}'."
        
        # Simulate new KV tensors for the entire prompt
        new_kv_tensors = f"KV_TENSORS({prompt})"
        logging.info(f"LLM generated new KV tensors: '{new_kv_tensors}'")
        return response_text, new_kv_tensors

class KVCacheManager:
    def __init__(self, host='localhost', port=6379, db=0):
        self.cache = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        logging.info(f"Connected to Redis at {host}:{port}/{db}")

    def store_kv_tensors(self, prompt_prefix: str, kv_tensors: str):
        try:
            self.cache.set(f"kv_cache:{prompt_prefix}", kv_tensors)
            logging.info(f"Stored KV tensors for prefix: '{prompt_prefix}'")
        except redis.exceptions.ConnectionError as e:
            logging.error(f"Redis connection error while storing: {e}")

    def retrieve_kv_tensors(self, prompt_prefix: str) -> str:
        try:
            kv_tensors = self.cache.get(f"kv_cache:{prompt_prefix}")
            if kv_tensors:
                logging.info(f"Retrieved KV tensors for prefix: '{prompt_prefix}'")
            else:
                logging.info(f"No KV tensors found for prefix: '{prompt_prefix}'")
            return kv_tensors
        except redis.exceptions.ConnectionError as e:
            logging.error(f"Redis connection error while retrieving: {e}")
            return None

    def find_longest_prefix(self, full_prompt: str) -> str:
        tokens = full_prompt.split()
        longest_match = ""
        for i in range(len(tokens), 0, -1):
            current_prefix_tokens = tokens[:i]
            current_prefix_str = " ".join(current_prefix_tokens)
            if self.cache.exists(f"kv_cache:{current_prefix_str}"):
                longest_match = current_prefix_str
                break
        if longest_match:
            logging.info(f"Found longest cached prefix: '{longest_match}' for prompt '{full_prompt}'")
        else:
            logging.info(f"No cached prefix found for prompt '{full_prompt}'")
        return longest_match

class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, List[str]] = {}

    def get_history(self, session_id: str) -> List[str]:
        return self.conversations.get(session_id, [])

    def add_message(self, session_id: str, sender: str, message: str):
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        self.conversations[session_id].append(f"{sender}: {message}")

    def get_full_prompt(self, session_id: str, new_message: str) -> str:
        history = self.get_history(session_id)
        if history:
            return "\n".join(history) + f"\nUser: {new_message}"
        return f"User: {new_message}"

app = FastAPI()
kv_cache_manager = KVCacheManager()
conversation_manager = ConversationManager()
llm = MockLLM()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message

    conversation_manager.add_message(session_id, "User", user_message)
    full_prompt = conversation_manager.get_full_prompt(session_id, user_message)
    logging.info(f"Full prompt for session {session_id}: '{full_prompt}'")

    # KV Cache Reuse Logic
    cached_prefix = kv_cache_manager.find_longest_prefix(full_prompt)
    cached_kv_tensors = None
    
    if cached_prefix:
        cached_kv_tensors = kv_cache_manager.retrieve_kv_tensors(cached_prefix)
        # For demonstration, we'll pass the whole prompt to LLM even with cache,
        # but a real LLM would only process the suffix and merge with cached KV.
        # The MockLLM's internal logic simulates this by acknowledging the cache.

    # LLM Inference
    bot_response_text, new_kv_tensors = llm.generate_response(full_prompt, cached_kv_tensors)

    # Store new KV tensors for the full prompt in the cache
    kv_cache_manager.store_kv_tensors(full_prompt, new_kv_tensors)

    conversation_manager.add_message(session_id, "Bot", bot_response_text)
    
    return ChatResponse(session_id=session_id, response=bot_response_text)

# To run this application:
# 1. Make sure Redis is running (e.g., `docker run -p 6379:6379 --name some-redis -d redis`)
# 2. Install dependencies: `pip install fastapi uvicorn redis pydantic`
# 3. Run the FastAPI app: `uvicorn main:app --reload`
# 4. Interact with the API using a tool like curl or a simple frontend.
