import os
from abc import ABC, abstractmethod

import openai
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# 1. LLM Abstraction Layer
class LLMService(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, model_name: str, **kwargs) -> str:
        pass

class OpenAIService(LLMService):
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(self, prompt: str, model_name: str = "gpt-3.5-turbo", **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."}, 
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from OpenAI: {e}"

class GeminiService(LLMService):
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def generate_response(self, prompt: str, model_name: str = "gemini-pro", **kwargs) -> str:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, **kwargs)
            return response.text
        except Exception as e:
            return f"Error from Gemini: {e}"

class LlamaService(LLMService):
    def generate_response(self, prompt: str, model_name: str = "llama-placeholder", **kwargs) -> str:
        # This is a placeholder for a real Llama integration.
        # In a real scenario, this would interact with a Llama API or a local model via transformers/ollama.
        return f"Llama (placeholder) response to: '{prompt}'. (Model: {model_name})"

# 2. LLM Manager/Router
class LLMRouter:
    def __init__(self):
        self.services = {
            "openai": OpenAIService(),
            "gemini": GeminiService(),
            "llama": LlamaService(), # Placeholder
        }

    def get_response(self, prompt: str, provider: str, model_name: str = None, **kwargs) -> str:
        service = self.services.get(provider.lower())
        if not service:
            return f"Error: LLM provider '{provider}' not supported."
        
        # Use default model name if not provided
        if model_name is None:
            if provider.lower() == "openai":
                model_name = "gpt-3.5-turbo"
            elif provider.lower() == "gemini":
                model_name = "gemini-pro"
            elif provider.lower() == "llama":
                model_name = "llama-placeholder"
        
        return service.generate_response(prompt, model_name, **kwargs)

# 3. Chatbot Core Logic
class Chatbot:
    def __init__(self, llm_router: LLMRouter):
        self.llm_router = llm_router

    def get_chatbot_response(self, user_query: str, provider: str, model_name: str = None) -> str:
        response = self.llm_router.get_response(user_query, provider, model_name)
        return response

# 4. Streamlit App
st.title("Intelligent Customer Support Chatbot")
st.write("Demonstrating Model Abstraction Pattern with dynamic LLM switching.")

llm_router = LLMRouter()
chatbot_instance = Chatbot(llm_router)

# Sidebar for configuration
st.sidebar.header("Configuration")
selected_provider = st.sidebar.selectbox(
    "Choose LLM Provider",
    ("OpenAI", "Gemini", "Llama")
)

# Model selection based on provider
model_options = {
    "OpenAI": ["gpt-3.5-turbo", "gpt-4o", "gpt-4"], # Add more as needed
    "Gemini": ["gemini-pro", "gemini-1.5-pro-latest"], # Add more as needed
    "Llama": ["llama-placeholder"]
}

selected_model = st.sidebar.selectbox(
    f"Choose {selected_provider} Model",
    model_options.get(selected_provider, [""])
)

user_input = st.text_area("Enter your query:", "How can I reset my password?")

if st.button("Get Response"):
    if not user_input:
        st.warning("Please enter a query.")
    else:
        with st.spinner(f"Getting response from {selected_provider} ({selected_model})..."):
            response = chatbot_instance.get_chatbot_response(
                user_query=user_input,
                provider=selected_provider.lower(),
                model_name=selected_model
            )
            st.subheader(f"Response from {selected_provider} ({selected_model}):")
            st.write(response)

st.sidebar.markdown("---\n**Note:** Ensure your `.env` file contains `OPENAI_API_KEY` and `GEMINI_API_KEY`.")
