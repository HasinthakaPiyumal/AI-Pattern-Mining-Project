import os
from pydantic import BaseSettings
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import openai
import google.generativeai as genai
import time
import random

load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    class Config:
        env_file = ".env"


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, history: list) -> str:
        pass

    def get_cost_per_token_input(self) -> float:
        return 0.0

    def get_cost_per_token_output(self) -> float:
        return 0.0

    def get_latency_estimate(self) -> float:
        return 0.0

    def is_configured(self) -> bool:
        return False


class OpenAIGPTProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self._api_key = api_key
        self.model_name = model_name
        self._cost_per_token_input = 0.0015 / 1000
        self._cost_per_token_output = 0.002 / 1000
        self._latency_estimate = 0.5
        if not self._api_key:
            print("Warning: OpenAI API key is not set. OpenAI provider will not function.")

    def generate_response(self, prompt: str, history: list) -> str:
        if not self._api_key:
            return "OpenAI provider not configured."

        messages = [{"role": "system", "content": "You are a helpful assistant for e-commerce customer support."}]
        for item in history:
            messages.append({"role": "user", "content": item["user"]})
            messages.append({"role": "assistant", "content": item["assistant"]})
        messages.append({"role": "user", "content": prompt})

        try:
            openai.api_key = self._api_key
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"Error with OpenAI: {e}"

    def get_cost_per_token_input(self) -> float:
        return self._cost_per_token_input

    def get_cost_per_token_output(self) -> float:
        return self._cost_per_token_output

    def get_latency_estimate(self) -> float:
        return self._latency_estimate
    
    def is_configured(self) -> bool:
        return bool(self._api_key)


class GoogleGeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        self._api_key = api_key
        self.model_name = model_name
        self._cost_per_token_input = 0.000125 / 1000
        self._cost_per_token_output = 0.000375 / 1000
        self._latency_estimate = 0.4
        self.model = None
        if not self._api_key:
            print("Warning: Google API key is not set. Gemini provider will not function.")
        else:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)
            except Exception as e:
                print(f"Error configuring Gemini: {e}")

    def generate_response(self, prompt: str, history: list) -> str:
        if not self.model:
            return "Gemini provider not configured."
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error with Gemini: {e}"

    def get_cost_per_token_input(self) -> float:
        return self._cost_per_token_input

    def get_cost_per_token_output(self) -> float:
        return self._cost_per_token_output

    def get_latency_estimate(self) -> float:
        return self._latency_estimate

    def is_configured(self) -> bool:
        return bool(self.model)


class LlamaLocalProvider(BaseLLMProvider):
    def __init__(self, model_path: str = "simulated_llama_model"):
        self.model_path = model_path
        self._cost_per_token_input = 0.0
        self._cost_per_token_output = 0.0
        self._latency_estimate = 1.0

    def generate_response(self, prompt: str, history: list) -> str:
        time.sleep(self._latency_estimate)
        return f"Llama Local (simulated) responded to your query: '{prompt}'"

    def get_cost_per_token_input(self) -> float:
        return self._cost_per_token_input

    def get_cost_per_token_output(self) -> float:
        return self._cost_per_token_output

    def get_latency_estimate(self) -> float:
        return self._latency_estimate

    def is_configured(self) -> bool:
        return True


class LLMRouter:
    def __init__(self, providers: dict[str, BaseLLMProvider]):
        self.providers = providers
        self.provider_names = list(providers.keys())
        self._current_provider_index = 0

    def choose_provider(self, query: str, strategy: str = "round_robin") -> BaseLLMProvider:
        available_providers = [p for p_name, p in self.providers.items() if p.is_configured()]
        if not available_providers:
            raise RuntimeError("No LLM providers are configured or available.")

        if strategy == "round_robin":
            chosen_provider = available_providers[self._current_provider_index]
            self._current_provider_index = (self._current_provider_index + 1) % len(available_providers)
            return chosen_provider
        elif strategy == "cost_effective":
            return min(available_providers, key=lambda p: p.get_cost_per_token_output())
        elif strategy == "low_latency":
            return min(available_providers, key=lambda p: p.get_latency_estimate())
        else:
            chosen_provider = available_providers[self._current_provider_index]
            self._current_provider_index = (self._current_provider_index + 1) % len(available_providers)
            return chosen_provider

    def route_request(self, prompt: str, history: list, strategy: str = "round_robin") -> str:
        provider = self.choose_provider(prompt, strategy)
        return provider.generate_response(prompt, history)


class CustomerSupportChatbot:
    def __init__(self, llm_router: LLMRouter):
        self.llm_router = llm_router
        self.conversation_history = []

    def get_response(self, user_query: str, routing_strategy: str = "round_robin") -> str:
        response = self.llm_router.route_request(user_query, self.conversation_history, routing_strategy)
        self.conversation_history.append({"user": user_query, "assistant": response})
        return response

    def reset_history(self):
        self.conversation_history = []


if __name__ == "__main__":
    settings = Settings()

    openai_provider = OpenAIGPTProvider(api_key=settings.OPENAI_API_KEY)
    gemini_provider = GoogleGeminiProvider(api_key=settings.GOOGLE_API_KEY)
    llama_provider = LlamaLocalProvider()

    llm_providers = {
        "openai": openai_provider,
        "gemini": gemini_provider,
        "llama": llama_provider
    }

    llm_router = LLMRouter(providers=llm_providers)
    chatbot = CustomerSupportChatbot(llm_router=llm_router)

    print("Welcome to the Smart Customer Support Chatbot!")
    print("Type 'exit' to end the conversation.")
    print("You can also specify a routing strategy: 'cost', 'latency', or 'round_robin'.")
    print("Example: cost: What is your return policy?")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break

            routing_strategy = "round_robin"
            query_parts = user_input.split(":", 1)
            if len(query_parts) == 2:
                strategy_prefix = query_parts[0].strip().lower()
                if strategy_prefix in ["cost", "latency", "round_robin"]:
                    routing_strategy = strategy_prefix
                    user_input = query_parts[1].strip()

            response = chatbot.get_response(user_input, routing_strategy=routing_strategy)
            print(f"Chatbot ({routing_strategy}): {response}")
        except RuntimeError as e:
            print(f"Error: {e}. Please ensure at least one LLM provider is configured and accessible.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break