
import os
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables for API keys
# from dotenv import load_dotenv
# load_dotenv()

# --- 1. Initialize LLM and Embedding Model ---
# Ensure OPENAI_API_KEY is set in your environment variables
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI(
    title="LLM-Enhanced E-commerce Recommender",
    description="An intelligent e-commerce recommender system leveraging LLMs for explanations, personalization, categorization, and marketing copy generation."
)

# --- 2. Data Models ---
class Product(BaseModel):
    id: str
    name: str
    description: str
    category: str = "" # Will be filled by LLM or pre-defined

class RecommendationRequest(BaseModel):
    user_query: str
    num_recommendations: int = 5

class RecommendedProduct(BaseModel):
    product: Product
    score: float
    explanation: str

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendedProduct]

class CategorizationRequest(BaseModel):
    product_description: str

class CategorizationResponse(BaseModel):
    product_description: str
    category: str

class MarketingCopyRequest(BaseModel):
    product_description: str
    product_name: str

class MarketingCopyResponse(BaseModel):
    product_name: str
    marketing_copy: str

# --- 3. Simulated Product Database (In-memory for demonstration) ---
products_data = [
    {"id": "P001", "name": "Wireless Bluetooth Headphones", "description": "High-quality over-ear headphones with noise cancellation and 20-hour battery life. Perfect for travel and daily commute.", "category": "Electronics"},
    {"id": "P002", "name": "Ergonomic Office Chair", "description": "Adjustable lumbar support, breathable mesh, and 360-degree swivel. Designed for long hours of comfortable work.", "category": "Furniture"},
    {"id": "P003", "name": "Smart Home Security Camera", "description": "1080p HD video, motion detection, two-way audio, and cloud storage. Monitor your home from anywhere.", "category": "Electronics"},
    {"id": "P004", "name": "Organic Green Tea Sampler", "description": "A selection of 10 organic green tea blends from around the world. Rich in antioxidants and natural flavor.", "category": "Food & Beverages"},
    {"id": "P005", "name": "Portable External SSD 1TB", "description": "Ultra-fast read/write speeds, compact design, and shock-resistant. Ideal for backing up large files on the go.", "category": "Electronics"},
    {"id": "P006", "name": "Yoga Mat with Carrying Strap", "description": "Non-slip surface, 6mm thick for comfort, and eco-friendly TPE material. Perfect for yoga, Pilates, and floor exercises.", "category": "Sports & Outdoors"},
    {"id": "P007", "name": "Espresso Coffee Machine", "description": "Semi-automatic espresso maker with milk frother. Brew barista-quality coffee at home.", "category": "Home Appliances"}
]

product_db: Dict[str, Product] = {p["id"]: Product(**p) for p in products_data}

# Pre-compute embeddings for products
product_descriptions = [p["description"] for p in products_data]
product_embeddings = embedding_model.encode(product_descriptions, convert_to_tensor=True)

# --- 4. Core Recommendation Logic ---

def get_embedding(text: str):
    return embedding_model.encode(text, convert_to_tensor=True)

def get_recommendations(
    user_query_embedding: np.ndarray, 
    top_n: int = 5
) -> List[Dict]:
    
    similarities = cosine_similarity(user_query_embedding.cpu().numpy().reshape(1, -1), product_embeddings.cpu().numpy())
    
    # Get top_n product indices based on similarity
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    
    recommended_products = []
    for i in top_indices:
        product = product_db[products_data[i]["id"]]
        score = similarities[0][i]
        recommended_products.append({"product": product, "score": float(score)})
    
    return recommended_products

def generate_llm_explanation(user_query: str, product: Product) -> str:
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful e-commerce assistant. Explain why a product is suitable for the user's query."),
        ("user", f"User query: '{user_query}'\nProduct Name: '{product.name}'\nProduct Description: '{product.description}'\n\nExplain in one concise sentence why this product is a good recommendation for the user, focusing on the query and product features.")
    ])
    chain = template | llm
    response = chain.invoke({"user_query": user_query, "product_name": product.name, "product_description": product.description})
    return response.content

def categorize_product_with_llm(description: str) -> str:
    template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert product categorizer. Assign a single, most appropriate e-commerce category to the given product description."),
        ("user", f"Product Description: '{description}'\n\nCategorize this product into a single, general e-commerce category (e.g., 'Electronics', 'Home & Kitchen', 'Apparel', 'Books', 'Sports & Outdoors', 'Food & Beverages'). Provide only the category name.")
    ])
    chain = template | llm
    response = chain.invoke({"description": description})
    return response.content.strip()

def generate_marketing_copy_with_llm(name: str, description: str) -> str:
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a creative marketing copywriter. Generate a compelling short marketing blurb for the given product."),
        ("user", f"Product Name: '{name}'\nProduct Description: '{description}'\n\nGenerate a captivating 2-3 sentence marketing copy for this product, highlighting its key benefits and appealing to potential customers.")
    ])
    chain = template | llm
    response = chain.invoke({"name": name, "description": description})
    return response.content

# --- 5. FastAPI Endpoints ---

@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend(request: RecommendationRequest):
    """Generates personalized product recommendations with LLM-enhanced explanations."""
    user_query_embedding = get_embedding(request.user_query)
    raw_recommendations = get_recommendations(user_query_embedding, request.num_recommendations)
    
    enhanced_recommendations = []
    for rec in raw_recommendations:
        product = rec["product"]
        explanation = generate_llm_explanation(request.user_query, product)
        enhanced_recommendations.append(RecommendedProduct(
            product=product,
            score=rec["score"],
            explanation=explanation
        ))
    
    return RecommendationResponse(recommendations=enhanced_recommendations)

@app.post("/categorize", response_model=CategorizationResponse, tags=["Product Management"])
async def categorize_product(request: CategorizationRequest):
    """Categorizes a product using an LLM based on its description."""
    category = categorize_product_with_llm(request.product_description)
    return CategorizationResponse(
        product_description=request.product_description,
        category=category
    )

@app.post("/marketing-copy", response_model=MarketingCopyResponse, tags=["Product Management"])
async def generate_marketing_copy(request: MarketingCopyRequest):
    """Generates marketing copy for a product using an LLM."""
    copy = generate_marketing_copy_with_llm(request.product_name, request.product_description)
    return MarketingCopyResponse(
        product_name=request.product_name,
        marketing_copy=copy
    )


# To run this application:
# 1. pip install fastapi uvicorn "langchain-openai" sentence-transformers scikit-learn numpy
# 2. Set your OpenAI API key as an environment variable: export OPENAI_API_KEY="your_api_key"
# 3. Run: uvicorn ecommerce_llm_recommender:app --reload
# 4. Access the API documentation at http://127.0.0.1:8000/docs

