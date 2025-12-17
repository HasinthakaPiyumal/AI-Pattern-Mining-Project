import os
from abc import ABC, abstractmethod
from typing import Dict, Any


class AbstractLLMService(ABC):
    """Abstract base class for LLM services, defining a unified interface."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config

    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        """Generates content based on the given prompt using the LLM."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Returns information about the underlying LLM."""
        pass


class LLMProviderFactory:
    """A factory to create and manage LLM service instances based on provider names."""

    _registry = {}

    @staticmethod
    def register_provider(name: str, llm_class: type):
        """Registers an LLM provider class with a given name."""
        if not issubclass(llm_class, AbstractLLMService):
            raise ValueError("Registered class must inherit from AbstractLLMService")
        LLMProviderFactory._registry[name.lower()] = llm_class

    @staticmethod
    def get_provider(name: str, config: Dict[str, Any] = None) -> AbstractLLMService:
        """Retrieves an instance of the specified LLM service."""
        llm_class = LLMProviderFactory._registry.get(name.lower())
        if not llm_class:
            raise ValueError(f"LLM provider \'{name}\' not registered.")
        # Pass the model_name from config to the constructor of the LLM service
        model_name = config.get("model_name") if config else None
        if model_name:
            return llm_class(model_name=model_name, config=config or {})
        else:
            return llm_class(config=config or {})


# Configuration management (simple example)
class AppConfig:
    """Manages application-wide configurations, especially API keys."""
    _instance = None
    _config_data = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # In a real application, load from .env, config files, etc.
        self._config_data["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "sk-your-openai-key")
        self._config_data["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "your-google-api-key")
        # Add other API keys as needed

    def get_setting(self, key: str, default: Any = None) -> Any:
        return self._config_data.get(key, default)

    def set_setting(self, key: str, value: Any):
        self._config_data[key] = value
