import abc
import os
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. Abstract LLM Adapter ---
class AbstractLLMAdapter(abc.ABC):
    """Abstract base class for all LLM adapters."""
    @abc.abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        pass

    @abc.abstractmethod
    def get_name(self) -> str:
        pass

# --- 2. Concrete LLM Adapters (Mock Implementations) ---
class OpenAIAdapter(AbstractLLMAdapter):
    def get_name(self) -> str:
        return "OpenAI_GPT"

    def generate_response(self, prompt: str, **kwargs) -> str:
        # Simulate API call to OpenAI
        # openai_api_key = os.getenv("OPENAI_API_KEY")
        # if not openai_api_key: raise ValueError("OPENAI_API_KEY not set")
        # print(f"Calling OpenAI with prompt: {prompt}")
        return f"[OpenAI Response] Processed query: '{prompt}'. This is a simulated response."

class GeminiAdapter(AbstractLLMAdapter):
    def get_name(self) -> str:
        return "Google_Gemini"

    def generate_response(self, prompt: str, **kwargs) -> str:
        # Simulate API call to Gemini
        # gemini_api_key = os.getenv("GEMINI_API_KEY")
        # if not gemini_api_key: raise ValueError("GEMINI_API_KEY not set")
        # print(f"Calling Gemini with prompt: {prompt}")
        return f"[Gemini Response] Your query about '{prompt}' has been handled. Simulated."

class LlamaAdapter(AbstractLLMAdapter):
    def get_name(self) -> str:
        return "Meta_Llama"

    def generate_response(self, prompt: str, **kwargs) -> str:
        # Simulate API call to Llama (e.g., via local model or API)
        # llama_model_path = os.getenv("LLAMA_MODEL_PATH")
        # if not llama_model_path: raise ValueError("LLAMA_MODEL_PATH not set")
        # print(f"Calling Llama with prompt: {prompt}")
        return f"[Llama Response] For '{prompt}', here is a generated answer. Simulation active."

# --- 3. LLM Selection Strategy ---
class LLMSelectionStrategy(abc.ABC):
    """Abstract base class for LLM selection strategies."""
    @abc.abstractmethod
    def select_llm(self, query: str, available_llms: Dict[str, AbstractLLMAdapter], **kwargs) -> AbstractLLMAdapter:
        pass

class RoundRobinSelectionStrategy(LLMSelectionStrategy):
    def __init__(self, llm_names: List[str]):
        self.llm_names = llm_names
        self._current_index = 0

    def select_llm(self, query: str, available_llms: Dict[str, AbstractLLMAdapter], **kwargs) -> AbstractLLMAdapter:
        if not self.llm_names or not available_llms:
            raise ValueError("No LLMs available for selection.")

        selected_name = self.llm_names[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.llm_names)
        
        if selected_name not in available_llms:
            # Fallback if the selected LLM is somehow not in available_llms
            print(f"Warning: Selected LLM '{selected_name}' not found. Falling back to first available.")
            return next(iter(available_llms.values()))

        return available_llms[selected_name]

class RuleBasedSelectionStrategy(LLMSelectionStrategy):
    def select_llm(self, query: str, available_llms: Dict[str, AbstractLLMAdapter], **kwargs) -> AbstractLLMAdapter:
        query_lower = query.lower()

        if "billing" in query_lower or "payment" in query_lower:
            # Example: Billing queries might need more factual, precise models
            if "OpenAI_GPT" in available_llms: return available_llms["OpenAI_GPT"]
        elif "recommendation" in query_lower or "product suggestion" in query_lower:
            # Example: Creative or more conversational models for suggestions
            if "Google_Gemini" in available_llms: return available_llms["Google_Gemini"]
        elif "technical support" in query_lower:
            # Example: Models that might be fine-tuned for specific technical docs
            if "Meta_Llama" in available_llms: return available_llms["Meta_Llama"]
        
        # Default fallback
        print("No specific rule matched. Falling back to Round Robin.")
        # As a fallback, use round robin if no specific rule matches
        # This would require an instance of RoundRobin or similar default
        if available_llms:
            # For simplicity, just return the first available if no specific rule matched and no explicit RR instance is here
            # A more robust solution would pass a default strategy or have one instantiated here.
            return next(iter(available_llms.values()))
        raise ValueError("No suitable LLM found for the query.")

# --- 4. Chatbot Core (LLM Manager/Router) ---
class ChatbotCore:
    def __init__(self, llm_adapters: List[AbstractLLMAdapter], selection_strategy: LLMSelectionStrategy):
        self.llm_adapters = {adapter.get_name(): adapter for adapter in llm_adapters}
        self.selection_strategy = selection_strategy

    def process_query(self, query: str, **kwargs) -> str:
        if not self.llm_adapters:
            raise ValueError("No LLM adapters configured.")

        selected_llm_adapter = self.selection_strategy.select_llm(query, self.llm_adapters, **kwargs)
        print(f"DEBUG: Selected LLM: {selected_llm_adapter.get_name()}")
        response = selected_llm_adapter.generate_response(query, **kwargs)
        return response

# --- 5. FastAPI Application ---
app = FastAPI(title="Dynamic Customer Support Chatbot")

class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"

class ChatResponse(BaseModel):
    response: str
    llm_used: str

# Initialize LLM adapters
openai_adapter = OpenAIAdapter()
gemini_adapter = GeminiAdapter()
llama_adapter = LlamaAdapter()

available_adapters = [
    openai_adapter,
    gemini_adapter,
    llama_adapter,
]

# Initialize Selection Strategy
# For demonstration, we can switch between strategies
# llm_names_for_rr = [ad.get_name() for ad in available_adapters]
# current_selection_strategy = RoundRobinSelectionStrategy(llm_names_for_rr)
current_selection_strategy = RuleBasedSelectionStrategy()

# Initialize Chatbot Core
chatbot_core = ChatbotCore(available_adapters, current_selection_strategy)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        selected_llm_name = current_selection_strategy.select_llm(request.query, chatbot_core.llm_adapters).get_name()
        response_text = chatbot_core.process_query(request.query)
        return ChatResponse(response=response_text, llm_used=selected_llm_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Chatbot service is running."}
