from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv

# Langchain imports for LLM services
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# --- 1. Data Models (`models.py` concepts) ---
class LLMStrategy(str, Enum):
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    DEFAULT = "default"

class CustomerQuery(BaseModel):
    query: str
    strategy: LLMStrategy = LLMStrategy.DEFAULT

class LLMResponse(BaseModel):
    response: str
    llm_used: str
    strategy_applied: str

# --- 2. LLM Abstraction Layer (`llm_abstractor.py` concepts) ---
class AbstractLLMService(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class OpenAIService(AbstractLLMService):
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = self.model.invoke(messages)
        return response.content

class GeminiService(AbstractLLMService):
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = self.model.invoke(messages)
        return response.content

class LlamaService(AbstractLLMService):
    def generate_response(self, prompt: str) -> str:
        # This is a mock service for demonstration
        return f"[Mock Llama Response for '{prompt}']: I am a cost-effective, but potentially slower LLM."

# --- 3. LLM Routing Layer (`llm_router.py` concepts) ---
class LLMRouter:
    def __init__(self, services: dict[str, AbstractLLMService]):
        self.services = services
        self.default_llm = "openai"  # Or any other preferred default

    def route_llm(self, query: str, strategy: LLMStrategy) -> tuple[str, AbstractLLMService]:
        if strategy == LLMStrategy.COST_OPTIMIZED:
            # Example: Llama is considered cost-optimized (mock here)
            if "llama" in self.services:
                return "llama", self.services["llama"]
            else:
                print("Llama service not available for COST_OPTIMIZED, falling back to default.")
                return self.default_llm, self.services[self.default_llm]
        elif strategy == LLMStrategy.LATENCY_OPTIMIZED:
            # Example: OpenAI is considered latency-optimized
            if "openai" in self.services:
                return "openai", self.services["openai"]
            else:
                print("OpenAI service not available for LATENCY_OPTIMIZED, falling back to default.")
                return self.default_llm, self.services[self.default_llm]
        else:  # LLMStrategy.DEFAULT
            # Simple round-robin or just pick a primary
            return self.default_llm, self.services[self.default_llm]

# --- 4. API and Application Logic (`main.py`) ---
app = FastAPI()

# Initialize LLM services
try:
    openai_service = OpenAIService()
except ValueError as e:
    print(f"Warning: {e}. OpenAI service will not be available.")
    openai_service = None

try:
    gemini_service = GeminiService()
except ValueError as e:
    print(f"Warning: {e}. Gemini service will not be available.")
    gemini_service = None

llama_service = LlamaService() # Mock service always available

llm_services = {}
if openai_service: llm_services["openai"] = openai_service
if gemini_service: llm_services["gemini"] = gemini_service
llm_services["llama"] = llama_service

# Ensure at least one service is available
if not llm_services:
    raise RuntimeError("No LLM services could be initialized. Please check your API keys.")

llm_router = LLMRouter(llm_services)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=LLMResponse)
async def ask_llm(customer_query: CustomerQuery):
    try:
        llm_name, selected_llm_service = llm_router.route_llm(customer_query.query, customer_query.strategy)
        response_content = selected_llm_service.generate_response(customer_query.query)
        return LLMResponse(
            response=response_content,
            llm_used=llm_name,
            strategy_applied=customer_query.strategy.value
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
