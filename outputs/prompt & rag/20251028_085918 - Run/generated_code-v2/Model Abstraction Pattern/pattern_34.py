import os
from abc import ABC, abstractmethod
from typing import List, Dict, Union
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import openai
import google.generativeai as genai


# config.py
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# llm_abstractor/llm_interface.py
class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, history: List[Dict[str, str]]) -> str:
        pass


# llm_abstractor/gpt_provider.py
class GPTProvider(LLMProvider):
    def __init__(self):
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        openai.api_key = Config.OPENAI_API_KEY

    def generate_response(self, prompt: str, history: List[Dict[str, str]]) -> str:
        messages = []
        for item in history:
            messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": prompt})

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message.content


# llm_abstractor/gemini_provider.py
class GeminiProvider(LLMProvider):
    def __init__(self):
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_response(self, prompt: str, history: List[Dict[str, str]]) -> str:
        convo_history = []
        for item in history:
            convo_history.append({'role': 'user' if item['role'] == 'user' else 'model', 'parts': [item['content']]})

        chat = self.model.start_chat(history=convo_history)
        response = chat.send_message(prompt)
        return response.text


# llm_abstractor/llama_provider.py (Mock/Placeholder)
class LlamaProvider(LLMProvider):
    def generate_response(self, prompt: str, history: List[Dict[str, str]]) -> str:
        print(f"[LlamaProvider]: Simulating response for prompt: {prompt}")
        return f"This is a simulated response from Llama for: {prompt}"


# llm_abstractor/llm_factory.py
class LLMFactory:
    @staticmethod
    def get_provider(model_type: str) -> LLMProvider:
        if model_type.lower() == "gpt":
            return GPTProvider()
        elif model_type.lower() == "gemini":
            return GeminiProvider()
        elif model_type.lower() == "llama":
            return LlamaProvider()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


# chatbot/models.py
class ChatRequest(BaseModel):
    prompt: str
    model_type: str
    history: List[Dict[str, str]] = []


class ChatResponse(BaseModel):
    response: str


# chatbot/chatbot_service.py
class ChatbotService:
    def __init__(self):
        pass

    def get_chat_response(self, request: ChatRequest) -> ChatResponse:
        provider = LLMFactory.get_provider(request.model_type)
        response_text = provider.generate_response(request.prompt, request.history)
        return ChatResponse(response=response_text)


# main.py
app = FastAPI()
chatbot_service = ChatbotService()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return chatbot_service.get_chat_response(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# requirements.txt content (not actual file, just content representation)
# fastapi
# uvicorn
# python-dotenv
# pydantic
# openai
# google-generativeai

# .env content (not actual file, just content representation)
# OPENAI_API_KEY="your_openai_api_key_here"
# GEMINI_API_KEY="your_gemini_api_key_here"
