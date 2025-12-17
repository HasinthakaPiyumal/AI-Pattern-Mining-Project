import abc
import os
import logging
import click

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Management (Simplified for single file) ---
# In a real application, these would come from .env or a config file
MOCK_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "mock-openai-key")
MOCK_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "mock-gemini-key")
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")

API_KEYS = {
    "openai": MOCK_OPENAI_API_KEY,
    "gemini": MOCK_GEMINI_API_KEY,
}

# --- 1. LLM Interface (Abstract Base Class) ---
class LLM(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, prompt: str) -> str:
        raise NotImplementedError

# --- 2. Concrete LLM Implementations (Mocked) ---
class OpenAIGPTLLM(LLM):
    def __init__(self, api_key: str):
        self.api_key = api_key
        logger.info("Initialized OpenAI GPT LLM with mock API key.")

    def generate_response(self, prompt: str) -> str:
        logger.info(f"Mocking OpenAI GPT response for prompt: '{prompt[:50]}...' ")
        # Simulate API call
        return f"[OpenAI GPT Response] Your query was: '{prompt}'. This is a simulated response."

class GoogleGeminiLLM(LLM):
    def __init__(self, api_key: str):
        self.api_key = api_key
        logger.info("Initialized Google Gemini LLM with mock API key.")

    def generate_response(self, prompt: str) -> str:
        logger.info(f"Mocking Google Gemini response for prompt: '{prompt[:50]}...' ")
        # Simulate API call
        return f"[Google Gemini Response] Your query was: '{prompt}'. This is a simulated response."

# --- 3. LLM Factory ---
class LLMFactory:
    @staticmethod
    def get_llm(provider_name: str, api_keys: dict) -> LLM:
        if provider_name == "openai":
            api_key = api_keys.get("openai")
            if not api_key:
                raise ValueError("OpenAI API key not provided.")
            return OpenAIGPTLLM(api_key)
        elif provider_name == "gemini":
            api_key = api_keys.get("gemini")
            if not api_key:
                raise ValueError("Google Gemini API key not provided.")
            return GoogleGeminiLLM(api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

# --- 4. LLM Abstraction Layer (Unified Interface) ---
class LLMAbstractionLayer:
    def __init__(self, initial_provider: str, api_keys: dict):
        self._api_keys = api_keys
        self._active_provider = None
        self.set_active_provider(initial_provider)

    def set_active_provider(self, provider_name: str):
        try:
            # Test if the provider can be instantiated before setting it active
            LLMFactory.get_llm(provider_name, self._api_keys)
            self._active_provider = provider_name
            logger.info(f"Active LLM provider set to: {self._active_provider}")
        except ValueError as e:
            logger.error(f"Failed to set active provider to {provider_name}: {e}")
            raise

    def get_active_provider(self) -> str:
        return self._active_provider

    def generate_response(self, prompt: str) -> str:
        try:
            llm_instance = LLMFactory.get_llm(self._active_provider, self._api_keys)
            return llm_instance.generate_response(prompt)
        except Exception as e:
            logger.error(f"Error generating response with {self._active_provider}: {e}")
            # Implement fallback logic here if needed
            return f"Error: Could not generate response using {self._active_provider}. Please try again or contact support. Details: {e}"

# --- 5. Application Layer: Chatbot Service ---
class ChatbotService:
    def __init__(self, llm_abstraction_layer: LLMAbstractionLayer):
        self.llm_abstraction_layer = llm_abstraction_layer
        self.conversation_history = []

    def _format_prompt(self, user_message: str) -> str:
        history_str = "\n".join(self.conversation_history)
        if history_str:
            return f"Conversation History:\n{history_str}\nUser: {user_message}\nAssistant:"
        return f"User: {user_message}\nAssistant:"

    def converse(self, user_message: str) -> str:
        self.conversation_history.append(f"User: {user_message}")
        prompt = self._format_prompt(user_message)
        response = self.llm_abstraction_layer.generate_response(prompt)
        self.conversation_history.append(f"Assistant: {response}")
        return response

    def reset_conversation(self):
        self.conversation_history = []
        logger.info("Conversation history reset.")


# --- 6. Admin Interface (CLI) ---
@click.group()
def cli():
    pass

# Initialize global instances for CLI to interact with
llm_abstraction_layer = LLMAbstractionLayer(DEFAULT_LLM_PROVIDER, API_KEYS)
chatbot_service = ChatbotService(llm_abstraction_layer)

@cli.command("chat")
def chat_command():
    """Start an interactive chatbot session."""
    click.echo(f"\n--- Starting Chatbot Session with {llm_abstraction_layer.get_active_provider().upper()} --- ")
    click.echo("Type 'exit' to end the chat or 'reset' to clear history.")
    chatbot_service.reset_conversation()

    while True:
        user_input = click.prompt("You")
        if user_input.lower() == 'exit':
            click.echo("Ending chat session. Goodbye!")
            break
        elif user_input.lower() == 'reset':
            chatbot_service.reset_conversation()
            click.echo("Conversation history has been reset.")
            continue

        response = chatbot_service.converse(user_input)
        click.echo(f"Bot: {response}")

@cli.command("set-provider")
@click.argument("provider", type=click.Choice(["openai", "gemini"]))
def set_provider_command(provider):
    """Set the active LLM provider (openai or gemini)."""
    try:
        llm_abstraction_layer.set_active_provider(provider)
        click.echo(f"Active LLM provider successfully set to: {llm_abstraction_layer.get_active_provider().upper()}")
    except ValueError as e:
        click.echo(f"Error: {e}")
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}")

@cli.command("current-provider")
def current_provider_command():
    """Show the currently active LLM provider."""
    click.echo(f"Currently active LLM provider: {llm_abstraction_layer.get_active_provider().upper()}")


if __name__ == "__main__":
    cli()