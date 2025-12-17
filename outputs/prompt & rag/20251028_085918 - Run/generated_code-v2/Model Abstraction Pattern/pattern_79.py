import streamlit as st
import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import openai
import google.generativeai as genai

# --- config.py ---

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PROVIDER_OPENAI = "OpenAI"
PROVIDER_GEMINI = "Gemini"
PROVIDER_DUMMY = "Dummy"


# --- llm_abstraction/providers/base_llm.py ---

class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass


# --- llm_abstraction/providers/openai_llm.py ---

class OpenAILLM(BaseLLM):
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        openai.api_key = OPENAI_API_KEY

    def generate_response(self, prompt: str) -> str:
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error from OpenAI: {e}"


# --- llm_abstraction/providers/gemini_llm.py ---

class GeminiLLM(BaseLLM):
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=GEMINI_API_KEY)

    def generate_response(self, prompt: str) -> str:
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error from Gemini: {e}"


# --- llm_abstraction/providers/dummy_llm.py ---

class DummyLLM(BaseLLM):
    def generate_response(self, prompt: str) -> str:
        return f"Dummy LLM received: '{prompt}' and responds with a predefined answer."


# --- llm_abstraction/llm_service.py ---

class LLMService:
    def __init__(self):
        self._providers = {}
        self.register_llm(PROVIDER_OPENAI, OpenAILLM())
        self.register_llm(PROVIDER_GEMINI, GeminiLLM())
        self.register_llm(PROVIDER_DUMMY, DummyLLM())
        self._current_llm = None

    def register_llm(self, name: str, llm_instance: BaseLLM):
        self._providers[name] = llm_instance

    def get_llm(self, provider_name: str) -> BaseLLM:
        llm = self._providers.get(provider_name)
        if not llm:
            raise ValueError(f"LLM provider '{provider_name}' not registered.")
        self._current_llm = llm
        return llm

    def generate(self, prompt: str) -> str:
        if not self._current_llm:
            return "No LLM provider selected. Please select one from the dropdown."
        try:
            return self._current_llm.generate_response(prompt)
        except Exception as e:
            return f"An error occurred during LLM generation: {e}"


# --- app.py ---

st.set_page_config(page_title="Intelligent Customer Support Chatbot")
st.title("🤖 Intelligent Customer Support Chatbot")

if "llm_service" not in st.session_state:
    st.session_state.llm_service = LLMService()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for LLM selection
st.sidebar.header("LLM Configuration")
selected_provider = st.sidebar.selectbox(
    "Choose LLM Provider:", 
    [PROVIDER_OPENAI, PROVIDER_GEMINI, PROVIDER_DUMMY]
)

try:
    st.session_state.llm_service.get_llm(selected_provider)
    st.sidebar.success(f"Currently using: {selected_provider}")
except ValueError as e:
    st.sidebar.error(str(e) + " Please ensure API keys are set.")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner(f"Generating response using {selected_provider}..."):
        response = st.session_state.llm_service.generate(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
