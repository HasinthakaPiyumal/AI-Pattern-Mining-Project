from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from fastapi import FastAPI
from pydantic import BaseModel

# 1. LLM Abstraction Layer
class AbstractLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class GPTAdapter(AbstractLLM):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        # Placeholder for OpenAI API call
        return f"GPT Response to: '{prompt}' (using key: {self.api_key[:5]}...)"

class GeminiAdapter(AbstractLLM):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        # Placeholder for Gemini API call
        return f"Gemini Response to: '{prompt}' (using key: {self.api_key[:5]}...)"

class LlamaAdapter(AbstractLLM):
    def __init__(self, model_path: str = "local_llama_model"):
        self.model_path = model_path

    def generate_response(self, prompt: str) -> str:
        # Placeholder for Llama API call or local inference
        return f"Llama Response to: '{prompt}' (using model: {self.model_path})"

# 2. LLM Management/Routing Layer
class LLMManager:
    def __init__(self):
        self._llms: Dict[str, AbstractLLM] = {}

    def register_llm(self, name: str, llm_instance: AbstractLLM):
        self._llms[name] = llm_instance

    def select_llm(self, preference: Dict[str, Any]) -> Optional[AbstractLLM]:
        if "model_name" in preference and preference["model_name"] in self._llms:
            return self._llms[preference["model_name"]]
        
        # Simple routing logic for demonstration
        # In a real scenario, this would involve more complex logic
        # based on cost, performance, language, etc.
        if "cost_preference" in preference and preference["cost_preference"] == "low":
            if "Llama" in self._llms: # Assuming Llama might be cheaper (e.g., self-hosted)
                return self._llms["Llama"]
        
        # Default to GPT if available, otherwise Gemini, then Llama
        if "GPT" in self._llms:
            return self._llms["GPT"]
        if "Gemini" in self._llms:
            return self._llms["Gemini"]
        if "Llama" in self._llms:
            return self._llms["Llama"]
            
        return None

    def get_response(self, prompt: str, preferences: Dict[str, Any] = None) -> str:
        preferences = preferences or {}
        selected_llm = self.select_llm(preferences)
        if selected_llm:
            return selected_llm.generate_response(prompt)
        return "Error: No suitable LLM found to handle the request."

# 3. Chatbot Service Layer
class ChatbotService:
    def __init__(self):
        self.llm_manager = LLMManager()
        # Register LLM providers - in a real app, API keys would come from env variables
        self.llm_manager.register_llm("GPT", GPTAdapter(api_key="sk-your-openai-key"))
        self.llm_manager.register_llm("Gemini", GeminiAdapter(api_key="your-gemini-key"))
        self.llm_manager.register_llm("Llama", LlamaAdapter(model_path="/path/to/llama/model"))

    def answer_query(self, user_query: str, preferences: Dict[str, Any] = None) -> str:
        return self.llm_manager.get_response(user_query, preferences)

# 4. API Layer (FastAPI Application)
app = FastAPI()
chatbot_service = ChatbotService()

class ChatRequest(BaseModel):
    user_query: str
    preferences: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    llm_used: Optional[str] = None # Added for clarity in response

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    response_text = chatbot_service.answer_query(request.user_query, request.preferences)
    
    # A more sophisticated approach would be to get the actual LLM name used from the manager
    # For simplicity, we can infer or return a generic message for now.
    # In a real LLMManager, select_llm could return (llm_instance, llm_name).
    
    llm_name_used = None
    if request.preferences and "model_name" in request.preferences:
        llm_name_used = request.preferences["model_name"]
    elif "GPT" in response_text: # Simple inference from response content
        llm_name_used = "GPT"
    elif "Gemini" in response_text:
        llm_name_used = "Gemini"
    elif "Llama" in response_text:
        llm_name_used = "Llama"

    return ChatResponse(response=response_text, llm_used=llm_name_used)
