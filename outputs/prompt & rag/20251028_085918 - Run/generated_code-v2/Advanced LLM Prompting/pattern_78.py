import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

PRODUCTS_DB = [
    {"name": "Budget Smartphone X", "description": "An affordable smartphone with essential features."},
    {"name": "Luxury Silk Scarf", "description": "Hand-woven silk scarf, perfect for an elegant touch."},
    {"name": "Eco-Friendly Reusable Water Bottle", "description": "Sustainable water bottle made from recycled materials."},
    {"name": "High-Performance Gaming PC", "description": "Top-tier gaming rig for immersive experiences."},
    {"name": "Organic Cotton T-shirt", "description": "Soft and breathable t-shirt made from 100% organic cotton."},
    {"name": "Smart Home Hub Pro", "description": "Centralize your smart home devices with this advanced hub."},
    {"name": "Vintage Leather Wallet", "description": "Handcrafted leather wallet with a classic design."},
    {"name": "Noise-Cancelling Headphones", "description": "Immerse yourself in sound with superior noise cancellation."},
    {"name": "Travel Backpack 50L", "description": "Spacious and durable backpack for extended trips."},
    {"name": "Gourmet Coffee Beans - Ethiopian Yirgacheffe", "description": "Premium single-origin coffee with floral notes."}
]

class RecommendationService:
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for RecommendationService.")
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

    def get_recommendations(self, product_query: str, persona: str) -> list[str]:
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a {persona}. Based on the user's query, recommend 3-5 relevant e-commerce products. Provide only the product names, one per line."),
                ("human", "User query: {product_query}"),
            ]
        )
        chain = prompt_template | self.llm
        response = chain.invoke({"persona": persona, "product_query": product_query})
        
        product_names = [line.strip() for line in response.content.split('\n') if line.strip()]
        return product_names

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    recommendation_service = RecommendationService(openai_api_key=OPENAI_API_KEY)
except ValueError:
    class DummyRecommendationService:
        def get_recommendations(self, product_query: str, persona: str) -> list[str]:
            return ["Error: OpenAI API key missing. Cannot provide recommendations. Please set OPENAI_API_KEY environment variable."]
    recommendation_service = DummyRecommendationService()

class RecommendationRequest(BaseModel):
    product_query: str
    persona: str

@app.post("/recommend")
async def get_product_recommendations(request: RecommendationRequest):
    recommendations = recommendation_service.get_recommendations(
        product_query=request.product_query,
        persona=request.persona
    )
    return {"recommendations": recommendations}
