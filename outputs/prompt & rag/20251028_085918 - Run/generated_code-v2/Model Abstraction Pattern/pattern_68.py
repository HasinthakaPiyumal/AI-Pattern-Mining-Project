import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Union
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

# --- Configuration and Environment Variables ---
load_dotenv()

# --- Logging Setup ---
logger.add("file.log", rotation="500 MB", level="INFO")
logger.info("Chatbot application started.")

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    message: str
    preferred_llm: str = "default" # Optional: hint for LLM selection

class ChatResponse(BaseModel):
    session_id: str
    response: str
    llm_provider: str

class Message(BaseModel):
    role: str
    content: str

# --- LLM Abstraction Layer ---
class BaseLLMProvider(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def generate_response(self, messages: List[Message]) -> str:
        pass

class PromptFormatter:
    @staticmethod
    def format_for_openai(messages: List[Message]) -> List[Dict]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    @staticmethod
    def format_for_gemini(messages: List[Message]) -> List[Dict]:
        # Gemini expects roles 'user' and 'model'
        formatted_messages = []
        for msg in messages:
            if msg.role == "system":
                # Gemini often doesn't have a direct 'system' role in chat history
                # For simplicity, we'll prepend system messages to the first user message
                if formatted_messages and formatted_messages[0]["role"] == "user":
                    formatted_messages[0]["parts"] = [{"text": msg.content + "\n" + formatted_messages[0]["parts"][0]["text"]}]
                else:
                    formatted_messages.insert(0, {"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                formatted_messages.append({"role": "model", "parts": [{"text": msg.content}]})
            else: # user role
                formatted_messages.append({"role": "user", "parts": [{"text": msg.content}]})
        return formatted_messages

    @staticmethod
    def format_for_llama(messages: List[Message]) -> str:
        # Llama models often prefer a single string prompt or specific chat templates
        # For simplicity, a concatenated string is used here.
        formatted_prompt = ""
        for msg in messages:
            if msg.role == "system":
                formatted_prompt += f"<<SYS>>\n{msg.content}\n<</SYS>>\n"
            elif msg.role == "user":
                formatted_prompt += f"[INST] {msg.content} [/INST]"
            elif msg.role == "assistant":
                formatted_prompt += f" {msg.content}\n"
        return formatted_prompt.strip()

class ErrorHandler:
    @staticmethod
    def handle_llm_error(provider_name: str, error: Exception):
        logger.error(f"Error from {provider_name} LLM: {error}")
        # In a real app, this could trigger alerts, fallbacks, etc.
        raise HTTPException(status_code=500, detail=f"LLM error from {provider_name}: {str(error)}")


# Concrete LLM Providers
class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        super().__init__("OpenAI")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables.")
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        except ImportError:
            logger.error("openai library not installed. Please run 'pip install openai'")
            self.client = None

    async def generate_response(self, messages: List[Message]) -> str:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized.")
        try:
            formatted_messages = PromptFormatter.format_for_openai(messages)
            chat_completion = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            ErrorHandler.handle_llm_error(self.name, e)

class GeminiProvider(BaseLLMProvider):
    def __init__(self):
        super().__init__("Gemini")
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables.")
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
            self.model = genai.GenerativeModel(self.model_name)
        except ImportError:
            logger.error("google-generativeai library not installed. Please run 'pip install google-generativeai'")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}")
            self.model = None

    async def generate_response(self, messages: List[Message]) -> str:
        if not self.model:
            raise RuntimeError("Gemini client not initialized.")
        try:
            # Gemini's `start_chat` expects history, then `send_message` expects new message
            # For simplicity, we'll just send the last user message and provide context if available.
            # A more robust solution would manage the chat session directly.
            formatted_messages = PromptFormatter.format_for_gemini(messages)
            
            # Separate history and current message for `start_chat` and `send_message`
            history = formatted_messages[:-1] if len(formatted_messages) > 1 else []
            current_message = formatted_messages[-1] if formatted_messages else None

            if not current_message or current_message['role'] != 'user':
                raise ValueError("No user message to send to Gemini.")

            chat_session = self.model.start_chat(history=history)
            response = await chat_session.send_message_async(current_message['parts'])
            return response.text
        except Exception as e:
            ErrorHandler.handle_llm_error(self.name, e)


class LlamaProvider(BaseLLMProvider):
    def __init__(self):
        super().__init__("Llama (Ollama)")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "llama2")
        try:
            import ollama
            # Check if ollama server is running
            ollama.list() # This will raise an error if server is not reachable
            self.client = ollama
        except ImportError:
            logger.error("ollama library not installed. Please run 'pip install ollama'")
            self.client = None
        except Exception as e:
            logger.error(f"Ollama server not reachable at {self.ollama_base_url} or other error: {e}")
            self.client = None

    async def generate_response(self, messages: List[Message]) -> str:
        if not self.client:
            raise RuntimeError("Llama (Ollama) client not initialized or server not reachable.")
        try:
            # Ollama's chat expects a list of messages similar to OpenAI
            # For simplicity, if we have a simple 'prompt string' requirement, we could use that.
            # Let's adapt to ollama's chat format.
            formatted_messages = []
            for msg in messages:
                if msg.role == "assistant": # Ollama expects 'assistant' for model responses
                    formatted_messages.append({"role": "assistant", "content": msg.content})
                elif msg.role == "user":
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif msg.role == "system":
                    # Ollama handles system messages by prepending to the first user message or within 'messages'
                    # Let's add it as a user message for simplicity if no explicit system role is expected for chat API.
                    formatted_messages.append({"role": "system", "content": msg.content})

            response = await self.client.chat(model=self.model_name, messages=formatted_messages)
            return response['message']['content']
        except Exception as e:
            ErrorHandler.handle_llm_error(self.name, e)


class LLMManager:
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {
            "openai": OpenAIProvider(),
            "gemini": GeminiProvider(),
            "llama": LlamaProvider(), # Assumes Ollama is running
        }
        self.default_provider_name = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()
        if self.default_provider_name not in self.providers:
            logger.warning(f"Default LLM provider '{self.default_provider_name}' not found. Falling back to OpenAI.")
            self.default_provider_name = "openai"

    def get_provider(self, preferred_llm: str = None) -> BaseLLMProvider:
        provider_name = (preferred_llm or self.default_provider_name).lower()
        provider = self.providers.get(provider_name)
        if not provider:
            logger.warning(f"Requested LLM provider '{provider_name}' not found. Using default '{self.default_provider_name}'.")
            provider = self.providers[self.default_provider_name]
        return provider

# --- Chatbot Backend API ---
app = FastAPI(
    title="Intelligent Customer Support Chatbot",
    description="Chatbot with an abstract LLM layer for flexible model switching."
)

# In-memory session store (replace with Redis/DB for production)
session_store: Dict[str, List[Message]] = {}
llm_manager = LLMManager()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Chatbot API is up and running."}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    user_message_content = request.message

    if session_id not in session_store:
        session_store[session_id] = []
        # Add a system message for initial context if desired
        # session_store[session_id].append(Message(role="system", content="You are a helpful customer support assistant."))

    # Add user message to history
    session_store[session_id].append(Message(role="user", content=user_message_content))

    try:
        provider = llm_manager.get_provider(request.preferred_llm)
        logger.info(f"Session {session_id}: Using LLM provider: {provider.name}")

        # Get current conversation history for the LLM
        conversation_history = session_store[session_id]

        llm_response_content = await provider.generate_response(conversation_history)

        # Add AI response to history
        session_store[session_id].append(Message(role="assistant", content=llm_response_content))

        return ChatResponse(session_id=session_id, response=llm_response_content, llm_provider=provider.name)
    except HTTPException as e:
        logger.error(f"Session {session_id}: Chat error: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Session {session_id}: Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# To run this application:
# 1. Save the code as main.py
# 2. Create a .env file in the same directory with your API keys:
#    OPENAI_API_KEY="your_openai_api_key"
#    GEMINI_API_KEY="your_gemini_api_key"
#    OLLAMA_BASE_URL="http://localhost:11434" # If using Ollama
#    OLLAMA_MODEL="llama2" # If using Ollama
#    DEFAULT_LLM_PROVIDER="openai" # or gemini, or llama
# 3. Install necessary libraries:
#    pip install fastapi uvicorn python-dotenv loguru openai google-generativeai ollama
# 4. Run the FastAPI application:
#    uvicorn main:app --reload
# 5. For Ollama, ensure the Ollama server is running and the specified model is pulled (e.g., 'ollama pull llama2').
