import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# 1. LLM Abstraction Layer
class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt_template: ChatPromptTemplate, user_query: str, model_config: dict) -> str:
        pass

class GeminiLLM(LLMProvider):
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

    def generate_response(self, prompt_template: ChatPromptTemplate, user_query: str, model_config: dict = None) -> str:
        if model_config and "model_name" in model_config:
            self.model = ChatGoogleGenerativeAI(model=model_config["model_name"], google_api_key=os.getenv("GOOGLE_API_KEY"))
        chain = prompt_template | self.model
        response = chain.invoke({"query": user_query})
        return response.content

class GPTLLM(LLMProvider):
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(self, prompt_template: ChatPromptTemplate, user_query: str, model_config: dict = None) -> str:
        if model_config and "model_name" in model_config:
            self.model = ChatOpenAI(model=model_config["model_name"], openai_api_key=os.getenv("OPENAI_API_KEY"))
        chain = prompt_template | self.model
        response = chain.invoke({"query": user_query})
        return response.content

class LlamaLLM(LLMProvider):
    def __init__(self):
        # Placeholder for Llama integration. This could be a local model, Replicate, or Hugging Face Inference API.
        # For simplicity, we'll return a static response or raise an error for now.
        pass

    def generate_response(self, prompt_template: ChatPromptTemplate, user_query: str, model_config: dict = None) -> str:
        # In a real scenario, integrate with Llama here.
        # Example: using a local model with a simple API or a cloud service.
        # For this example, we'll simulate a response.
        return f"[Llama Model Response] I understand you're asking about '{user_query}'. I am currently under development for this type of query."


class LLMAbstractionLayer:
    def __init__(self):
        self.providers = {
            "gemini": GeminiLLM(),
            "gpt": GPTLLM(),
            "llama": LlamaLLM(),
        }

    def get_llm(self, provider_name: str) -> LLMProvider:
        provider = self.providers.get(provider_name.lower())
        if not provider:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
        return provider

# 2. Chatbot Service
class ChatbotService:
    def __init__(self, llm_abstraction_layer: LLMAbstractionLayer):
        self.llm_abstraction_layer = llm_abstraction_layer
        self.general_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful e-commerce customer support assistant. Answer questions concisely and professionally."),
            ("user", "{query}")
        ])
        self.complex_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an advanced e-commerce technical support specialist. Provide detailed and accurate solutions, asking clarifying questions if necessary."),
            ("user", "{query}")
        ])
        self.product_specific_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an e-commerce product expert. Provide information specifically related to product features, comparisons, and availability. If you don't know, state that you're looking it up."),
            ("user", "{query}")
        ])

    def _determine_llm_and_prompt(self, query: str, customer_context: dict) -> tuple[str, ChatPromptTemplate, dict]:
        query_lower = query.lower()
        model_config = {}

        if "technical issue" in query_lower or "troubleshoot" in query_lower or "repair" in query_lower:
            # Route complex queries to Gemini or a more capable model
            return "gemini", self.complex_prompt, {"model_name": "gemini-pro"}
        elif "product details" in query_lower or "features" in query_lower or "specifications" in query_lower:
            # Route product-specific queries, potentially to a fine-tuned Llama or a specialized GPT instance
            # For this example, let's use GPT and pretend it's specialized.
            return "gpt", self.product_specific_prompt, {"model_name": "gpt-4"}
        elif "order status" in query_lower or "shipping" in query_lower or "delivery" in query_lower:
            # These are often simpler, but can become complex. Start with GPT.
            return "gpt", self.general_prompt, {"model_name": "gpt-3.5-turbo"}
        elif "return policy" in query_lower or "refund" in query_lower:
            return "gemini", self.general_prompt, {"model_name": "gemini-pro"}
        elif "hello" in query_lower or "hi" in query_lower or "support" in query_lower:
            return "gpt", self.general_prompt, {"model_name": "gpt-3.5-turbo"}
        else:
            # Default to GPT for general queries
            return "gpt", self.general_prompt, {"model_name": "gpt-3.5-turbo"}

    def process_query(self, query: str, customer_context: dict = None) -> str:
        if customer_context is None:
            customer_context = {}

        provider_name, prompt_template, model_config = self._determine_llm_and_prompt(query, customer_context)

        try:
            llm_provider = self.llm_abstraction_layer.get_llm(provider_name)
            response = llm_provider.generate_response(prompt_template, query, model_config)
            return response
        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"Chatbot configuration error: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing query with LLM provider: {e}")

# FastAPI Application
app = FastAPI(title="E-commerce Chatbot API", description="Intelligent customer support chatbot with LLM abstraction.")

llm_abs_layer = LLMAbstractionLayer()
chatbot_service = ChatbotService(llm_abs_layer)

class ChatbotQuery(BaseModel):
    query: str
    customer_context: dict = {}

@app.post("/chat")
async def chat_with_bot(chatbot_query: ChatbotQuery):
    try:
        response = chatbot_service.process_query(chatbot_query.query, chatbot_query.customer_context)
        return {"response": response}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
