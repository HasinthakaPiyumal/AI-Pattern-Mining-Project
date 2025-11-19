from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger
import os
from typing import List, Dict, Any

load_dotenv()

# --- LLM Integration Layer (Placeholder) ---
class LLMService:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        # In a real application, you'd initialize actual LLM clients here
        if not self.openai_api_key and not self.google_api_key:
            logger.warning("No LLM API keys found. LLM services will return mock data.")

    def generate_text(self, prompt: str) -> str:
        # This is a mock LLM call. In reality, you'd use openai.ChatCompletion.create or similar.
        logger.info(f"Simulating LLM generation for prompt: {prompt[:50]}...")
        if self.openai_api_key or self.google_api_key:
            # Simulate a more intelligent response if keys are present (still mock)
            return f"LLM response to '{prompt[:30]}...': This is a generated text based on your input. [Powered by LLM]"
        return f"Mock LLM response to '{prompt[:30]}...': Generated text. [No LLM API key]"

# --- Recommendation Engine (Placeholder) ---
class RecommendationEngine:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        # Mock item knowledge base
        self.products = {
            "P001": {"name": "Laptop Pro", "category": "Electronics", "price": 1200, "features": "High performance, sleek design"},
            "P002": {"name": "Bluetooth Headphones", "category": "Electronics", "price": 150, "features": "Noise cancelling, long battery life"},
            "P003": {"name": "Ergonomic Chair", "category": "Office", "price": 400, "features": "Lumbar support, adjustable height"},
            "P004": {"name": "Smartwatch", "category": "Wearables", "price": 300, "features": "Heart rate monitor, GPS"},
        }

    def get_recommendations(self, user_id: str, preferences: List[str]) -> List[Dict[str, Any]]:
        logger.info(f"Generating recommendations for user {user_id} with preferences: {preferences}")
        # In a real system, LangChain/LlamaIndex would process preferences and item knowledge
        # For now, a simple mock or LLM call to suggest products
        if "electronics" in [p.lower() for p in preferences]:
            return [self.products["P001"], self.products["P002"]]
        elif "office" in [p.lower() for p in preferences]:
            return [self.products["P003"]]
        else:
            # Use LLM to generalize if preferences are complex or abstract
            llm_prompt = f"Given user preferences: {', '.join(preferences)}. Suggest some product categories or specific products relevant to an e-commerce store."
            llm_response = self.llm_service.generate_text(llm_prompt)
            logger.info(f"LLM-based recommendation hint: {llm_response}")
            return [self.products["P004"], self.products["P001"]]

    def get_recommendation_explanation(self, user_id: str, product_id: str) -> str:
        logger.info(f"Generating explanation for user {user_id} and product {product_id}")
        product = self.products.get(product_id)
        if not product:
            return "Product not found."

        llm_prompt = f"Explain why a user might like {product['name']} ({product['category']}) with features '{product['features']}' based on general e-commerce purchase patterns."
        return self.llm_service.generate_text(llm_prompt)

# --- Customer Support Chatbot (Placeholder) ---
class CustomerSupportChatbot:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        # In a real system, ChromaDB would be initialized here with loaded documents.
        self.faq_knowledge = [
            "How to return a product: You can return products within 30 days.",
            "Shipping costs: Standard shipping is free on orders over $50.",
            "Payment methods: We accept Visa, Mastercard, and PayPal."
        ]

    def chat(self, user_id: str, message: str, chat_history: List[str]) -> str:
        logger.info(f"User {user_id} message: {message}. History length: {len(chat_history)}")

        # Simulate RAG: simple keyword match for now
        relevant_docs = []
        for doc in self.faq_knowledge:
            if any(keyword in doc.lower() for keyword in message.lower().split()):
                relevant_docs.append(doc)

        context = "\n".join(relevant_docs) if relevant_docs else "No specific knowledge found."

        llm_prompt = f"You are an e-commerce customer support agent. Answer the user's question concisely. Use the following context if relevant: {context}\n\nUser: {message}\nChat History: {'\n'.join(chat_history)}\nAgent:"

        response = self.llm_service.generate_text(llm_prompt)
        return response

# --- Content Generator (Placeholder) ---
class ContentGenerator:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_product_description(self, product_name: str, features: List[str], category: str) -> str:
        logger.info(f"Generating description for {product_name}")
        llm_prompt = f"Generate an engaging and SEO-friendly product description for a product named '{product_name}' in the '{category}' category, with key features: {', '.join(features)}. Make it concise and persuasive."
        return self.llm_service.generate_text(llm_prompt)

    def generate_marketing_content(self, product_name: str, target_audience: str, tone: str) -> str:
        logger.info(f"Generating marketing content for {product_name}")
        llm_prompt = f"Create a short, catchy marketing slogan or ad copy for '{product_name}'. Target audience: {target_audience}. Tone: {tone}."
        return self.llm_service.generate_text(llm_prompt)


# --- FastAPI Application --- 
app = FastAPI(
    title="LLM-Powered E-commerce System",
    description="Intelligent recommendation, customer support, and content generation for e-commerce.",
    version="0.1.0",
)

# Initialize services
llm_service = LLMService()
recommendation_engine = RecommendationEngine(llm_service)
customer_support_chatbot = CustomerSupportChatbot(llm_service)
content_generator = ContentGenerator(llm_service)

# --- Pydantic Models for API Requests/Responses ---
class RecommendationRequest(BaseModel):
    user_id: str
    preferences: List[str]

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    explanation: str

class ChatRequest(BaseModel):
    user_id: str
    message: str
    chat_history: List[str] = []

class ChatResponse(BaseModel):
    user_id: str
    response: str
    updated_chat_history: List[str]

class ProductDescriptionRequest(BaseModel):
    product_name: str
    features: List[str]
    category: str

class ProductDescriptionResponse(BaseModel):
    product_name: str
    description: str

class MarketingContentRequest(BaseModel):
    product_name: str
    target_audience: str
    tone: str

class MarketingContentResponse(BaseModel):
    product_name: str
    content: str

# --- API Endpoints ---

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_product_recommendations(request: RecommendationRequest):
    recommendations = recommendation_engine.get_recommendations(request.user_id, request.preferences)
    # For simplicity, generating explanation for the first recommended product or a generic one
    explanation = "No specific explanation generated" # Default
    if recommendations:
        explanation = recommendation_engine.get_recommendation_explanation(request.user_id, recommendations[0].get('id', 'N/A'))
    else:
        explanation = llm_service.generate_text(f"Explain why no recommendations were found for user with preferences: {request.preferences}")

    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        explanation=explanation
    )

@app.post("/chat", response_model=ChatResponse)
async def customer_support_chat(request: ChatRequest):
    response = customer_support_chatbot.chat(request.user_id, request.message, request.chat_history)
    updated_history = request.chat_history + [f"User: {request.message}", f"Agent: {response}"]
    return ChatResponse(
        user_id=request.user_id,
        response=response,
        updated_chat_history=updated_history
    )

@app.post("/generate-description", response_model=ProductDescriptionResponse)
async def generate_product_desc(request: ProductDescriptionRequest):
    description = content_generator.generate_product_description(request.product_name, request.features, request.category)
    return ProductDescriptionResponse(
        product_name=request.product_name,
        description=description
    )

@app.post("/generate-marketing-content", response_model=MarketingContentResponse)
async def generate_marketing(request: MarketingContentRequest):
    content = content_generator.generate_marketing_content(request.product_name, request.target_audience, request.tone)
    return MarketingContentResponse(
        product_name=request.product_name,
        content=content
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application...")
    # To run: uvicorn main:app --reload --port 8000
    # Ensure you have a .env file with OPENAI_API_KEY or GOOGLE_API_KEY for 'smarter' mock LLM responses.
    uvicorn.run(app, host="0.0.0.0", port=8000)
