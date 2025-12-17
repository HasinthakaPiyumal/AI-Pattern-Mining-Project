from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. LLM Abstraction Layer

class LLMAdapter(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class GPTAdapter(LLMAdapter):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        # Simulate API call to GPT
        # In a real application, you would use openai.ChatCompletion.create
        # For demonstration, we just return a canned response
        if "hello" in prompt.lower():
            return "Hello from GPT! How can I help you today?"
        return f"GPT processed your request: '{prompt}'."

class GeminiAdapter(LLMAdapter):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        # Simulate API call to Gemini
        # In a real application, you would use google.generativeai
        if "problem" in prompt.lower():
            return "Gemini says: Let's break down that problem together."
        return f"Gemini understood: '{prompt}'."

class LlamaAdapter(LLMAdapter):
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Simulate loading Llama model

    def generate_response(self, prompt: str) -> str:
        # Simulate local inference with Llama
        if "thank you" in prompt.lower():
            return "Llama responds: You're welcome!"
        return f"Llama is thinking about: '{prompt}'."

class LLMFactory:
    @staticmethod
    def get_llm_adapter(model_type: str, config: Dict[str, Any]) -> LLMAdapter:
        if model_type.lower() == "gpt":
            return GPTAdapter(api_key=config.get("api_key", "mock_gpt_key"))
        elif model_type.lower() == "gemini":
            return GeminiAdapter(api_key=config.get("api_key", "mock_gemini_key"))
        elif model_type.lower() == "llama":
            return LlamaAdapter(model_path=config.get("model_path", "/path/to/llama_model"))
        else:
            raise ValueError(f"Unknown LLM model type: {model_type}")

# 2. Chatbot Core Logic

class ChatbotPlatform:
    def __init__(self, default_model: str = "gpt", default_config: Dict[str, Any] = None):
        self._llm_adapter: LLMAdapter = LLMFactory.get_llm_adapter(default_model, default_config or {})
        self.current_model_type = default_model

    def set_llm_adapter(self, model_type: str, config: Dict[str, Any]):
        self._llm_adapter = LLMFactory.get_llm_adapter(model_type, config)
        self.current_model_type = model_type
        print(f"Switched LLM adapter to: {model_type}")

    def process_query(self, query: str, criteria: Optional[Dict[str, Any]] = None) -> str:
        # Example of dynamic switching based on criteria
        # In a real app, this logic would be more sophisticated
        if criteria and criteria.get("cost_preference") == "low" and self.current_model_type != "llama":
            print("Criteria suggests low cost, switching to Llama (simulated local model).")
            self.set_llm_adapter("llama", {"model_path": "/path/to/llama_model"})
        elif criteria and criteria.get("query_complexity") == "high" and self.current_model_type != "gemini":
            print("Criteria suggests high complexity, switching to Gemini.")
            self.set_llm_adapter("gemini", {"api_key": "mock_gemini_key"})
        elif criteria and criteria.get("latency_tolerance") == "low" and self.current_model_type != "gpt":
            print("Criteria suggests low latency, switching to GPT.")
            self.set_llm_adapter("gpt", {"api_key": "mock_gpt_key"})

        response = self._llm_adapter.generate_response(query)
        return f"[Using {self.current_model_type}] {response}"

# 3. API Layer (FastAPI)

app = FastAPI()
chatbot = ChatbotPlatform(default_model="gpt", default_config={
    "api_key": "your_openai_api_key"
})

class ChatRequest(BaseModel):
    query: str
    preferences: Optional[Dict[str, Any]] = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = chatbot.process_query(request.query, request.preferences)
        return {"response": response, "model_used": chatbot.current_model_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run this FastAPI application:
# 1. Save the code as 'chatbot_platform.py'
# 2. Install dependencies: pip install fastapi uvicorn pydantic
# 3. Run from your terminal: uvicorn chatbot_platform:app --reload
# 4. Access the API at http://127.0.0.1:8000/docs for Swagger UI. 