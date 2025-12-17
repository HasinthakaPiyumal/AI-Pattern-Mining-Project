import os
from abc import ABC, abstractmethod
import openai
import google.generativeai as genai

class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class GPTAdapter(BaseLLM):
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"Error with GPT: {e}"

class GeminiAdapter(BaseLLM):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error with Gemini: {e}"

class LLMManager:
    def __init__(self):
        self.adapters = {}

    def register_llm(self, provider_name: str, llm_adapter: BaseLLM):
        self.adapters[provider_name] = llm_adapter

    def get_llm(self, provider_name: str) -> BaseLLM:
        if provider_name not in self.adapters:
            raise ValueError(f"LLM provider '{provider_name}' not registered.")
        return self.adapters[provider_name]

    def initialize_adapters(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if openai_api_key:
            self.register_llm("gpt", GPTAdapter(openai_api_key))
        if google_api_key:
            self.register_llm("gemini", GeminiAdapter(google_api_key))
