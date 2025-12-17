from fastapi import FastAPI
from pydantic import BaseModel
from abc import ABC, abstractmethod
import os
from itertools import cycle
from dotenv import load_dotenv

# Optional: Import actual LLM client libraries if keys are available
# from openai import OpenAI
# import google.generativeai as genai

# --- Configuration (config.py logic) ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- LLM Abstraction Layer (llm_abstraction.py logic) ---

class AbstractLLMService(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class GPTService(AbstractLLMService):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        if not api_key:
            raise ValueError("GPT API key is not provided.")
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("The 'openai' library is not installed. Please install it to use GPTService.")

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error with GPT: {e}"

class GeminiService(AbstractLLMService):
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        if not api_key:
            raise ValueError("Gemini API key is not provided.")
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("The 'google-generativeai' library is not installed. Please install it to use GeminiService.")

    def generate_response(self, prompt: str) -> str:
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error with Gemini: {e}"

class LlamaService(AbstractLLMService):
    def __init__(self, model: str = "llama-mock"):
        self.model = model

    def generate_response(self, prompt: str) -> str:
        return f"Llama Mock Response for '{prompt}': This is a simulated response from {self.model}."

# --- Chatbot Orchestrator (chatbot_orchestrator.py logic) ---

class ChatbotOrchestrator:
    def __init__(self, llm_services: dict[str, AbstractLLMService]):
        self.llm_services = llm_services
        self.llm_names = list(llm_services.keys())
        if not self.llm_names:
            raise ValueError("No LLM services provided to the orchestrator.")
        self.llm_iterator = cycle(self.llm_names)

    def get_response(self, query: str) -> tuple[str, str]:
        selected_llm_name = next(self.llm_iterator)
        selected_llm = self.llm_services[selected_llm_name]
        response_text = selected_llm.generate_response(query)
        return response_text, selected_llm_name

# --- FastAPI Backend (main.py logic) ---

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    provider: str

# Initialize LLM services
llm_services = {}
if OPENAI_API_KEY:
    try:
        llm_services["gpt"] = GPTService(api_key=OPENAI_API_KEY)
    except (ImportError, ValueError) as e:
        print(f"Could not initialize GPTService: {e}")

if GEMINI_API_KEY:
    try:
        llm_services["gemini"] = GeminiService(api_key=GEMINI_API_KEY)
    except (ImportError, ValueError) as e:
        print(f"Could not initialize GeminiService: {e}")

llm_services["llama_mock"] = LlamaService() 

if not llm_services:
    raise RuntimeError("No active LLM services initialized. Ensure API keys are set or at least one mock service is active.")

orchestrator = ChatbotOrchestrator(llm_services)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    response_text, provider_name = orchestrator.get_response(request.query)
    return ChatResponse(response=response_text, provider=provider_name)