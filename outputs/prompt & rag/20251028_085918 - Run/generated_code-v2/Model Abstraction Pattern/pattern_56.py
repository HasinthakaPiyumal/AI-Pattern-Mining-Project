import os
from abc import ABC, abstractmethod
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- LLM Service Abstraction ---

class LLMService(ABC):
    @abstractmethod
    def generate_response(self, query: str, context: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def analyze_sentiment(self, text: str) -> str:
        pass

# --- Concrete LLM Implementations ---

class GPTService(LLMService):
    def __init__(self):
        # In a real application, you would initialize the OpenAI client here
        # For this example, we'll just use a mock response.
        self.api_key = os.getenv("OPENAI_API_KEY", "mock_openai_key")

    def generate_response(self, query: str, context: Optional[str] = None) -> str:
        # Mocking GPT response
        if "order status" in query.lower():
            return "Your order #12345 is currently in transit and expected to arrive by Friday."
        elif "return policy" in query.lower():
            return "Our return policy allows returns within 30 days of purchase with a valid receipt."
        return f"(GPT) I understand you're asking about: {query}. If context is provided: {context or 'No context provided.'}"

    def analyze_sentiment(self, text: str) -> str:
        # Mocking GPT sentiment analysis
        if "happy" in text.lower() or "good" in text.lower():
            return "Positive"
        elif "unhappy" in text.lower() or "bad" in text.lower():
            return "Negative"
        return "Neutral"

class GeminiService(LLMService):
    def __init__(self):
        # In a real application, you would initialize the Gemini client here
        # For this example, we'll just use a mock response.
        self.api_key = os.getenv("GEMINI_API_KEY", "mock_gemini_key")

    def generate_response(self, query: str, context: Optional[str] = None) -> str:
        # Mocking Gemini response
        if "shipping cost" in query.lower():
            return "Standard shipping within the US costs $5.99. Expedited options are available."
        elif "payment methods" in query.lower():
            return "We accept Visa, Mastercard, American Express, and PayPal."
        return f"(Gemini) I can help with: {query}. More details if context is provided: {context or 'No context provided.'}"

    def analyze_sentiment(self, text: str) -> str:
        # Mocking Gemini sentiment analysis
        if "excellent" in text.lower() or "satisfied" in text.lower():
            return "Strongly Positive"
        elif "terrible" in text.lower() or "frustrated" in text.lower():
            return "Strongly Negative"
        return "Mixed"

class LlamaService(LLMService):
    def __init__(self):
        # Placeholder for Llama. In a real scenario, this might interact with
        # a local Llama instance, an API, or a Hugging Face model.
        self.model_path = os.getenv("LLAMA_MODEL_PATH", "mock_llama_model")

    def generate_response(self, query: str, context: Optional[str] = None) -> str:
        # Mocking Llama response
        return f"(Llama) Processing your query: {query}. Context: {context or 'N/A'}. This is a placeholder response."

    def analyze_sentiment(self, text: str) -> str:
        # Mocking Llama sentiment analysis
        return f"(Llama) Sentiment analysis for '{text}': Undetermined (placeholder)"

# --- LLM Factory ---

class LLMFactory:
    @staticmethod
    def get_llm_service(provider: str) -> LLMService:
        if provider.lower() == "gpt":
            return GPTService()
        elif provider.lower() == "gemini":
            return GeminiService()
        elif provider.lower() == "llama":
            return LlamaService()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

# --- Query Classifier ---

class QueryClassifier:
    def classify_query(self, query: str) -> dict:
        query_lower = query.lower()
        if "order status" in query_lower or "delivery" in query_lower or "tracking" in query_lower:
            return {"type": "Order Related", "complexity": "low", "provider_preference": "gpt"}
        elif "return policy" in query_lower or "refund" in query_lower or "exchange" in query_lower:
            return {"type": "Policy Related", "complexity": "medium", "provider_preference": "gpt"}
        elif "shipping cost" in query_lower or "payment methods" in query_lower or "discount" in query_lower:
            return {"type": "Billing/Shipping", "complexity": "low", "provider_preference": "gemini"}
        elif "product" in query_lower or "features" in query_lower or "recommendation" in query_lower:
            return {"type": "Product Inquiry", "complexity": "medium", "provider_preference": "gemini"}
        elif "technical issue" in query_lower or "troubleshoot" in query_lower or "bug" in query_lower:
            return {"type": "Technical Support", "complexity": "high", "provider_preference": "llama"}
        elif "sentiment" in query_lower or "feedback" in query_lower or "experience" in query_lower:
            return {"type": "Sentiment Analysis", "complexity": "medium", "provider_preference": "gpt"}
        else:
            return {"type": "General Inquiry", "complexity": "low", "provider_preference": "gemini"}

# --- Customer Support Assistant ---

class CustomerSupportAssistant:
    def __init__(self):
        self.query_classifier = QueryClassifier()

    def handle_query(self, query: str, context: Optional[str] = None) -> str:
        classification = self.query_classifier.classify_query(query)
        provider_preference = classification.get("provider_preference", "gemini") # Default to Gemini

        print(f"Classified query: {classification}")
        print(f"Selected provider based on preference: {provider_preference}")

        try:
            llm_service = LLMFactory.get_llm_service(provider_preference)
            response = llm_service.generate_response(query, context)
            return response
        except ValueError as e:
            return f"Error: {e}. Could not find a suitable LLM provider."

    def get_sentiment(self, text: str) -> str:
        # For sentiment, let's explicitly use GPT for this example's routing
        try:
            llm_service = LLMFactory.get_llm_service("gpt")
            sentiment = llm_service.analyze_sentiment(text)
            return sentiment
        except ValueError as e:
            return f"Error: {e}. Could not perform sentiment analysis."

# --- API Layer (FastAPI) ---

app = FastAPI()
customer_assistant = CustomerSupportAssistant()

class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None

class SentimentRequest(BaseModel):
    text: str

@app.post("/ask")
async def ask_assistant(request: QueryRequest):
    try:
        response = customer_assistant.handle_query(request.query, request.context)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment")
async def get_text_sentiment(request: SentimentRequest):
    try:
        sentiment = customer_assistant.get_sentiment(request.text)
        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run this application:
# 1. Save the code as customer_support_assistant.py
# 2. Install necessary libraries: pip install fastapi uvicorn openai google-generativeai
#    (Note: `openai` and `google-generativeai` are not strictly needed for the mock, but for real integration)
# 3. Set environment variables if you want to use real LLMs (e.g., OPENAI_API_KEY, GEMINI_API_KEY)
# 4. Run: uvicorn customer_support_assistant:app --reload
# 5. Access the API at http://127.0.0.1:8000/docs