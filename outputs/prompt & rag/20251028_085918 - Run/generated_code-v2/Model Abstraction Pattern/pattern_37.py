import streamlit as st
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os

# Try importing specific LLM integrations, handle gracefully if not available
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # st.warning("Warning: `langchain_openai` not found. OpenAI models will not be available.")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    # st.warning("Warning: `langchain_google_genai` not found. Google Gemini models will not be available.")

load_dotenv()

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    @abstractmethod
    def generate_response(self, prompt: str, chat_history: list) -> str:
        pass

class OpenAIAdapter(LLMProvider):
    """Adapter for OpenAI GPT models."""
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAIAdapter requires `langchain_openai` to be installed.")
        self.llm = ChatOpenAI(model=model_name, api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(self, prompt: str, chat_history: list) -> str:
        messages = []
        for message in chat_history:
            if message["role"] == "user":
                messages.append(("human", message["content"]))
            elif message["role"] == "assistant":
                messages.append(("ai", message["content"]))
        messages.append(("human", prompt))
        
        response = self.llm.invoke(messages)
        return response.content

class GeminiAdapter(LLMProvider):
    """Adapter for Google Gemini models."""
    def __init__(self, model_name: str = "gemini-pro"):
        if not GEMINI_AVAILABLE:
            raise ImportError("GeminiAdapter requires `langchain_google_genai` to be installed.")
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=os.getenv("GOOGLE_API_KEY"))

    def generate_response(self, prompt: str, chat_history: list) -> str:
        messages = []
        for message in chat_history:
            if message["role"] == "user":
                messages.append(("human", message["content"]))
            elif message["role"] == "assistant":
                messages.append(("ai", message["content"]))
        messages.append(("human", prompt))
        
        response = self.llm.invoke(messages)
        return response.content

class LLMFactory:
    """Factory to create LLM provider instances."""
    @staticmethod
    def get_llm_provider(provider_name: str, model_name: str = None) -> LLMProvider:
        if provider_name.lower() == "openai":
            if not OPENAI_AVAILABLE:
                st.error("OpenAI models requested but `langchain_openai` is not installed.")
                st.stop()
            return OpenAIAdapter(model_name or "gpt-3.5-turbo")
        elif provider_name.lower() == "gemini":
            if not GEMINI_AVAILABLE:
                st.error("Gemini models requested but `langchain_google_genai` is not installed.")
                st.stop()
            return GeminiAdapter(model_name or "gemini-pro")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

class Chatbot:
    """Core chatbot logic using the LLM abstraction."""
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def get_response(self, user_query: str, chat_history: list) -> str:
        return self.llm_provider.generate_response(user_query, chat_history)


# --- Streamlit UI --- 

st.set_page_config(page_title="LLM-Agnostic Chatbot")
st.title("Customer Support Chatbot (LLM-Agnostic)")
st.write("Switch between different LLM providers seamlessly.")

# Sidebar for configuration
st.sidebar.header("Configuration")

# LLM Provider selection
llm_options = []
if OPENAI_AVAILABLE: 
    llm_options.append("OpenAI")
if GEMINI_AVAILABLE:
    llm_options.append("Gemini")

if not llm_options:
    st.error("No LLM providers available. Please install `langchain_openai` or `langchain_google_genai` and set API keys in `.env`.")
    st.stop()

selected_provider = st.sidebar.selectbox(
    "Choose LLM Provider",
    llm_options,
    key="llm_provider_selection"
)

# Model selection (basic, could be dynamic based on provider)
model_name = ""
if selected_provider == "OpenAI":
    model_name = st.sidebar.text_input("OpenAI Model Name", "gpt-3.5-turbo", key="openai_model")
elif selected_provider == "Gemini":
    model_name = st.sidebar.text_input("Gemini Model Name", "gemini-pro", key="gemini_model")

# Initialize LLM provider and chatbot
@st.cache_resource
def get_chatbot(provider: str, model: str):
    try:
        llm_provider = LLMFactory.get_llm_provider(provider, model)
        return Chatbot(llm_provider)
    except Exception as e:
        st.error(f"Error initializing LLM provider: {e}")
        st.stop()

chatbot = get_chatbot(selected_provider, model_name)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get response from chatbot
        response = chatbot.get_response(prompt, st.session_state.messages[:-1]) # Pass history excluding current user prompt
        full_response += response
        message_placeholder.markdown(full_response + " ")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
