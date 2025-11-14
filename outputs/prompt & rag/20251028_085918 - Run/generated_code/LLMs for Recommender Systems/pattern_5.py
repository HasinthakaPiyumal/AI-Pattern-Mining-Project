from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

# Initialize FastAPI app
app = FastAPI()

# --- Data Models ---
class Product(BaseModel):
    id: int
    name: str
    description: str
    category: str
    price: float

class RecommendationRequest(BaseModel):
    user_query: str
    user_preferences: dict = {}

class RecommendedProduct(BaseModel):
    id: int
    name: str
    description: str
    explanation: str

class RecommendationResponse(BaseModel):
    recommendations: list[RecommendedProduct]

# --- Global Variables / Initialization ---

# Sample Product Data (in a real app, this would come from a database)
products_data = [
    {"id": 1, "name": "Smartwatch Pro", "description": "Advanced smartwatch with health tracking and GPS.", "category": "Electronics", "price": 299.99},
    {"id": 2, "name": "Ergonomic Office Chair", "description": "Comfortable chair for long working hours, adjustable.", "category": "Home & Office", "price": 189.50},
    {"id": 3, "name": "Noise-Cancelling Headphones", "description": "Premium over-ear headphones with superior sound and active noise cancellation.", "category": "Electronics", "price": 349.00},
    {"id": 4, "name": "Yoga Mat Deluxe", "description": "High-quality, non-slip yoga mat for all types of practice.", "category": "Sports & Outdoors", "price": 45.00},
    {"id": 5, "name": "Espresso Machine", "description": "Automatic espresso maker with milk frother, perfect for coffee lovers.", "category": "Home & Kitchen", "price": 499.00},
    {"id": 6, "name": "Gaming Laptop X", "description": "High-performance gaming laptop with RTX graphics and fast processor.", "category": "Electronics", "price": 1500.00},
    {"id": 7, "name": "Wireless Earbuds", "description": "Compact wireless earbuds with great sound and long battery life.", "category": "Electronics", "price": 120.00},
    {"id": 8, "name": "Desk Lamp with Wireless Charger", "description": "Modern desk lamp with adjustable brightness and integrated phone charger.", "category": "Home & Office", "price": 75.00},
    {"id": 9, "name": "Resistance Bands Set", "description": "Complete set of resistance bands for full-body workouts.", "category": "Sports & Outdoors", "price": 30.00},
    {"id": 10, "name": "Cookbook: Italian Classics", "description": "A collection of authentic Italian recipes for home cooking.", "category": "Books", "price": 25.00},
]
products = [Product(**data) for data in products_data]

# Load Sentence Transformer Model for embeddings
# Using 'all-MiniLM-L6-v2' for a balance of speed and performance
# In a production environment, consider larger models or fine-tuning
print("Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# Generate embeddings for all products
product_descriptions = [p.name + " " + p.description for p in products]
product_embeddings = model.encode(product_descriptions, convert_to_tensor=True)
product_embeddings_np = product_embeddings.cpu().numpy()

# Initialize FAISS index for efficient similarity search
dimension = product_embeddings_np.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
faiss_index.add(product_embeddings_np)

print(f"FAISS index created with {faiss_index.ntotal} items.")

# --- LLM Simulation for Explanation Generation ---
# In a real system, this would be an actual LLM call (e.g., OpenAI API, Llama, Gemini)
# For demonstration, we use a simple rule-based explanation.
def generate_llm_explanation(product: Product, query: str, user_preferences: dict = None) -> str:
    explanation = f"Based on your query '{query}', we recommend {product.name} ({product.category}). "
    
    # Simulate incorporating product attributes and query relevance
    if "health" in query.lower() or "fitness" in query.lower() and "Smartwatch" in product.name:
        explanation += f"Its advanced health tracking features align with your interest in health."
    elif "office" in query.lower() or "work" in query.lower() and "Office Chair" in product.name:
        explanation += f"Its ergonomic design is perfect for long working hours."
    elif "music" in query.lower() or "audio" in query.lower() and ("Headphones" in product.name or "Earbuds" in product.name):
        explanation += f"You'll appreciate its superior sound quality and noise cancellation."
    elif "coffee" in query.lower() and "Espresso Machine" in product.name:
        explanation += f"As a coffee lover, you'll enjoy freshly brewed espresso at home."
    elif "gaming" in query.lower() and "Gaming Laptop" in product.name:
        explanation += f"Its high-performance specifications are ideal for an immersive gaming experience."
    else:
        explanation += f"We think you'll find its features like '{product.description.split('.')[0].lower()}' very useful."
    
    # Simulate incorporating user preferences
    if user_preferences and user_preferences.get("budget"):
        budget = user_preferences["budget"]
        if product.price <= budget:
            explanation += f" It also fits within your budget of ${budget:.2f}."
        else:
            explanation += f" Please note its price is ${product.price:.2f}."

    return explanation


# --- API Endpoints ---

@app.post("/recommend", response_model=RecommendationResponse)
async def get_product_recommendations(request: RecommendationRequest, k: int = 3):
    """
    Generates product recommendations based on a user query using LLM embeddings
    and FAISS for similarity search, and provides LLM-enhanced explanations.
    """
    print(f"Received recommendation request: {request.user_query}")
    
    # 1. Embed the user query
    query_embedding = model.encode([request.user_query], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().numpy()

    # 2. Perform similarity search using FAISS
    # D: Distances, I: Indices of the nearest neighbors
    distances, indices = faiss_index.search(query_embedding_np, k) 
    
    recommended_products_list = []
    for i in indices[0]: # indices[0] because we have only one query
        product = products[i]
        # 3. Generate human-centric explanation using a simulated LLM
        explanation = generate_llm_explanation(product, request.user_query, request.user_preferences)
        recommended_products_list.append(
            RecommendedProduct(
                id=product.id,
                name=product.name,
                description=product.description,
                explanation=explanation
            )
        )

    return RecommendationResponse(recommendations=recommended_products_list)

@app.get("/products", response_model=list[Product])
async def get_all_products():
    """
    Returns a list of all available products.
    """
    return products

# To run this application:
# 1. Save the code as main.py
# 2. Install necessary libraries: pip install fastapi uvicorn sentence-transformers faiss-cpu pydantic numpy torch
# 3. Run from your terminal: uvicorn main:app --reload
# 4. Access the API at http://127.0.0.1:8000/docs for interactive documentation.
