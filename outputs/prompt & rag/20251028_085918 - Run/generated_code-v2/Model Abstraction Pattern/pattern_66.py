import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import openai
import google.generativeai as genai
import time
import streamlit as st

# --- Config (config.py equivalent) ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_GPT_MODEL = os.getenv("DEFAULT_GPT_MODEL", "gpt-3.5-turbo")
DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-pro")

# --- LLM Adapters (llm_adapters.py equivalent) ---
class AbstractLLMAdapter(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, history: list) -> str:
        pass

class GPTAdapter(AbstractLLMAdapter):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model_name = model_name

    def generate_response(self, prompt: str, history: list) -> str:
        messages = [{"role": "system", "content": "You are a helpful customer support assistant."}]
        for turn in history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content

class GeminiAdapter(AbstractLLMAdapter):
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_response(self, prompt: str, history: list) -> str:
        gemini_history = []
        for turn in history:
            gemini_history.append({"role": "user", "parts": [{"text": turn["user"]}]})
            gemini_history.append({"role": "model", "parts": [{"text": turn["assistant"]}]})

        convo = self.model.start_chat(history=gemini_history)
        response = convo.send_message(prompt)
        return response.text

class LlamaAdapter(AbstractLLMAdapter):
    def __init__(self):
        pass

    def generate_response(self, prompt: str, history: list) -> str:
        time.sleep(1) 
        if "product specific" in prompt.lower() or "features" in prompt.lower():
            return "This is a simulated response from Llama for product-specific queries: The product XYZ has features A, B, and C, and costs $99."
        return "This is a simulated general response from Llama."

# --- LLM Manager (llm_manager.py equivalent) ---
class LLMManager:
    def __init__(self, openai_api_key: str, google_api_key: str, default_gpt_model: str, default_gemini_model: str):
        self._adapters = {
            "gpt": GPTAdapter(api_key=openai_api_key, model_name=default_gpt_model),
            "gemini": GeminiAdapter(api_key=google_api_key, model_name=default_gemini_model),
            "llama": LlamaAdapter()
        }

    def get_llm_adapter(self, provider_name: str) -> AbstractLLMAdapter:
        adapter = self._adapters.get(provider_name.lower())
        if not adapter:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
        return adapter

# --- Query Router (query_router.py equivalent) ---
class QueryRouter:
    def __init__(self):
        pass

    def route_query(self, query: str) -> str:
        query_lower = query.lower()
        if "product specific" in query_lower or "features" in query_lower or "specifications" in query_lower:
            return "llama"
        elif len(query.split()) > 10 and ("complex" in query_lower or "analyze" in query_lower or "reasoning" in query_lower):
            return "gemini"
        else:
            return "gpt"

# --- Chatbot Service (chatbot_service.py equivalent) ---
class ChatbotService:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.query_router = QueryRouter()
        self.conversation_history = []

    def process_query(self, user_query: str) -> str:
        selected_provider = self.query_router.route_query(user_query)
        llm_adapter = self.llm_manager.get_llm_adapter(selected_provider)

        response = llm_adapter.generate_response(user_query, self.conversation_history)

        self.conversation_history.append({"user": user_query, "assistant": response})

        return response

# --- Streamlit App (main.py equivalent) ---
st.title("Multi-Modal Customer Support Chatbot")

if "chatbot_service" not in st.session_state:
    llm_manager = LLMManager(
        openai_api_key=OPENAI_API_KEY,
        google_api_key=GOOGLE_API_KEY,
        default_gpt_model=DEFAULT_GPT_MODEL,
        default_gemini_model=DEFAULT_GEMINI_MODEL
    )
    st.session_state.chatbot_service = ChatbotService(llm_manager)
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        response = st.session_state.chatbot_service.process_query(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})