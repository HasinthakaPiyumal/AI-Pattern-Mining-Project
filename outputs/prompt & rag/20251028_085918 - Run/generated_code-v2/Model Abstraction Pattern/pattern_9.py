import streamlit as st
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import openai
import google.generativeai as genai

# --- .env content (for demonstration, in a real project, this would be a separate file) ---
# OPENAI_API_KEY="your_openai_api_key_here"
# GOOGLE_API_KEY="your_google_api_key_here"

# --- requirements.txt content (for demonstration, in a real project, this would be a separate file) ---
# streamlit
# openai
# google-generativeai
# python-dotenv

# --- llm_abstraction/base_llm.py ---
class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

# --- llm_abstraction/openai_llm.py ---
class OpenAILLM(BaseLLM):
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            st.error("OPENAI_API_KEY not found in environment variables.")
            raise ValueError("OPENAI_API_KEY not set.")

    def generate_response(self, prompt: str) -> str:
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error from OpenAI: {e}"

# --- llm_abstraction/gemini_llm.py ---
class GeminiLLM(BaseLLM):
    def __init__(self):
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("GOOGLE_API_KEY not found in environment variables.")
            raise ValueError("GOOGLE_API_KEY not set.")
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error from Gemini: {e}"

# --- llm_abstraction/llm_manager.py ---
class LLMManager:
    def __init__(self):
        self._llms = {
            "OpenAI": OpenAILLM(),
            "Gemini": GeminiLLM()
        }

    def get_llm(self, provider_name: str) -> BaseLLM:
        llm_instance = self._llms.get(provider_name)
        if not llm_instance:
            raise ValueError(f"LLM provider '{provider_name}' not supported.")
        return llm_instance

# --- main.py ---
load_dotenv() # Load environment variables from .env

st.set_page_config(page_title="Multi-LLM Customer Support Chatbot", layout="centered")
st.title("🤖 Multi-LLM Customer Support Chatbot")

@st.cache_resource
def get_llm_manager():
    return LLMManager()

llm_manager = get_llm_manager()

# Sidebar for LLM selection
st.sidebar.header("LLM Provider Configuration")
selected_llm_provider = st.sidebar.selectbox(
    "Select LLM Provider:",
    list(llm_manager._llms.keys())
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            current_llm = llm_manager.get_llm(selected_llm_provider)
            with st.spinner(f"Generating response with {selected_llm_provider}..."):
                response = current_llm.generate_response(prompt)
            st.markdown(response)
        except ValueError as e:
            st.error(f"Configuration Error: {e}. Please check your .env file and selected provider.")
            response = f"Error: {e}"
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            response = f"Error: {e}"
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
