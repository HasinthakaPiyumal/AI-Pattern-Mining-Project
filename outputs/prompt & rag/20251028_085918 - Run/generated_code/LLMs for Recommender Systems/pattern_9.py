
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import asyncio
import re

# --- 1. Data Ingestion & Management Layer (Simulated In-Memory) ---

# Simulated Product Database
# In a real system, this would be a persistent database (PostgreSQL, MongoDB)
PRODUCTS_DB = [
    {
        "id": "prod_001",
        "name": "XYZ Ultrabook",
        "description": "Powerful and lightweight laptop with Intel i7 processor, 16GB RAM, 512GB SSD. Ideal for professionals and light video editing. Long battery life.",
        "category": "Laptops",
        "brand": "XYZ",
        "price": 1150.00,
        "rating": 4.7,
        "features": ["Intel i7", "16GB RAM", "512GB SSD", "Lightweight", "Long Battery Life"]
    },
    {
        "id": "prod_002",
        "name": "ABC Probook",
        "description": "High-performance laptop with AMD Ryzen 7, 32GB RAM, 1TB SSD, dedicated NVIDIA RTX 3050 GPU. Slightly heavier but excellent for heavy video editing and gaming.",
        "category": "Laptops",
        "brand": "ABC",
        "price": 1499.00,
        "rating": 4.5,
        "features": ["AMD Ryzen 7", "32GB RAM", "1TB SSD", "NVIDIA RTX 3050", "Gaming Ready"]
    },
    {
        "id": "prod_003",
        "name": "LMN Slimline",
        "description": "Budget-friendly slim laptop with Intel i5, 8GB RAM, 256GB SSD. Good for everyday tasks, browsing, and light office work. Very portable.",
        "category": "Laptops",
        "brand": "LMN",
        "price": 850.00,
        "rating": 4.2,
        "features": ["Intel i5", "8GB RAM", "256GB SSD", "Slim", "Portable"]
    },
    {
        "id": "prod_004",
        "name": "PQR Essentials",
        "description": "Value-for-money laptop with large 15.6-inch display, AMD Ryzen 5, 8GB RAM, 512GB SSD. Great for students and home use.",
        "category": "Laptops",
        "brand": "PQR",
        "price": 780.00,
        "rating": 4.0,
        "features": ["AMD Ryzen 5", "8GB RAM", "512GB SSD", "Large Screen", "Good Value"]
    },
    {
        "id": "prod_005",
        "name": "AquaGuard Waterproof Jacket",
        "description": "Durable and breathable waterproof jacket, ideal for outdoor adventures in rainy weather. Features sealed seams and adjustable hood.",
        "category": "Outdoor Gear",
        "brand": "TrailBlazer",
        "price": 120.00,
        "rating": 4.8,
        "features": ["Waterproof", "Breathable", "Sealed Seams", "Adjustable Hood", "Outdoor"]
    },
    {
        "id": "prod_006",
        "name": "CloudWalk Running Shoes",
        "description": "Lightweight running shoes with superior cushioning for long-distance comfort. Designed for road running.",
        "category": "Footwear",
        "brand": "Stride",
        "price": 95.00,
        "rating": 4.6,
        "features": ["Lightweight", "Cushioned", "Running", "Road Running"]
    },
    {
        "id": "prod_007",
        "name": "ErgoFit Office Chair",
        "description": "Ergonomic office chair with lumbar support, adjustable armrests, and headrest. Perfect for long working hours.",
        "category": "Office Furniture",
        "brand": "ComfortZone",
        "price": 299.00,
        "rating": 4.3,
        "features": ["Ergonomic", "Lumbar Support", "Adjustable", "Comfortable"]
    },
    {
        "id": "prod_008",
        "name": "ZenFlow Yoga Mat",
        "description": "Eco-friendly yoga mat made from natural rubber, non-slip surface for superior grip during practice. Comes with carrying strap.",
        "category": "Fitness",
        "brand": "Harmony",
        "price": 50.00,
        "rating": 4.9,
        "features": ["Eco-friendly", "Natural Rubber", "Non-slip", "Yoga", "Fitness"]
    },
    {
        "id": "prod_009",
        "name": "GamerX Headset Pro",
        "description": "High-fidelity gaming headset with 7.1 surround sound, noise-cancelling mic, and RGB lighting. Compatible with PC and consoles.",
        "category": "Gaming Peripherals",
        "brand": "GamerX",
        "price": 180.00,
        "rating": 4.7,
        "features": ["7.1 Surround Sound", "Noise-Cancelling Mic", "RGB Lighting", "Gaming", "PC/Console Compatible"]
    },
    {
        "id": "prod_010",
        "name": "SmartHome Hub Max",
        "description": "Central smart home hub with voice assistant, touch screen, and integrated security camera. Controls all smart devices.",
        "category": "Smart Home",
        "brand": "ConnectIQ",
        "price": 220.00,
        "rating": 4.5,
        "features": ["Voice Assistant", "Touch Screen", "Security Camera", "Smart Home Control"]
    },
]

# Convert to pandas DataFrame for easier manipulation
products_df = pd.DataFrame(PRODUCTS_DB)
products_df["full_text"] = products_df.apply(
    lambda row: f"{row['name']}. {row['description']}. Category: {row['category']}. Brand: {row['brand']}. Features: {', '.join(row['features'])}",
    axis=1
)

# Simulated User Profile Database (in-memory session storage)
USER_SESSIONS: Dict[str, Dict[str, Any]] = {}

# --- 2. Embedding & Semantic Layer ---

# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for all products
product_embeddings = embedding_model.encode(products_df["full_text"].tolist())
embedding_dim = product_embeddings.shape[1]

# Initialize Faiss index
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(product_embeddings)

# --- Pydantic Models for API ---

class ProductRecommendation(BaseModel):
    id: str
    name: str
    description: str
    price: float
    rating: float
    brand: str
    category: str
    explanation: str

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    recommendations: Optional[List[ProductRecommendation]] = None
    user_filters: Optional[Dict[str, Any]] = None
    session_id: str

# --- Helper Functions ---

def get_user_session(user_id: str) -> Dict[str, Any]:
    """Retrieves or initializes a user session."""
    if user_id not in USER_SESSIONS:
        USER_SESSIONS[user_id] = {
            "chat_history": [],
            "inferred_preferences": {},
            "current_filters": {},
            "last_recommendations": []
        }
    return USER_SESSIONS[user_id]

def update_user_session(user_id: str, updates: Dict[str, Any]):
    """Updates a user session with new information."""
    session = get_user_session(user_id)
    session.update(updates)

def generate_embedding(text: str) -> np.ndarray:
    """Generates an embedding for a given text."""
    return embedding_model.encode([text])[0]

def search_vector_db(query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
    """Searches the Faiss index for top_k similar products."""
    D, I = faiss_index.search(np.array([query_embedding]), top_k)
    product_ids = [products_df.iloc[i]["id"] for i in I[0] if i != -1]
    return products_df[products_df["id"].isin(product_ids)].to_dict(orient="records")

def get_product_details(product_ids: List[str]) -> List[Dict[str, Any]]:
    """Retrieves full product details for a list of product IDs."""
    return products_df[products_df["id"].isin(product_ids)].to_dict(orient="records")

# --- 3. LLM Orchestration & Reasoning Layer (Simulated) ---

async def simulate_llm_response(
    prompt: str,
    context: Dict[str, Any],
    chat_history: List[str]
) -> Dict[str, Any]:
    """Simulates LLM behavior for intent recognition, entity extraction, and explanation.
    In a real application, this would call an actual LLM API (e.g., OpenAI, Cohere).
    """
    # Simple intent recognition based on keywords
    user_query = chat_history[-1] if chat_history else ""
    intent = "recommendation" # Default
    entities = {}
    response_text = "I'm processing your request."
    explanation_text = "Based on your preferences and our product catalog."

    lower_query = user_query.lower()

    if "laptop" in lower_query or "computer" in lower_query:
        entities["category"] = "Laptops"
        intent = "recommendation"

    if "waterproof jacket" in lower_query or "outdoor gear" in lower_query:
        entities["category"] = "Outdoor Gear"
        intent = "recommendation"

    if "cheaper" in lower_query or "budget" in lower_query or "price range" in lower_query:
        intent = "refine"
        price_match = re.search(r"around \$(\d+)-?(\d*)", lower_query)
        if price_match:
            min_price = float(price_match.group(1))
            max_price = float(price_match.group(2)) if price_match.group(2) else min_price + 200 # Arbitrary range
            entities["min_price"] = min_price
            entities["max_price"] = max_price
        else:
            # Try to extract a single price point
            single_price_match = re.search(r"under \$(\d+)|less than \$(\d+)|up to \$(\d+)", lower_query)
            if single_price_match:
                max_price = float(single_price_match.group(1) or single_price_match.group(2) or single_price_match.group(3))
                entities["max_price"] = max_price
                entities["min_price"] = 0.0 # From zero

    if "filter by brand" in lower_query:
        brand_match = re.search(r"filter by brand (\w+)", lower_query)
        if brand_match:
            entities["brand"] = brand_match.group(1).upper()
            intent = "refine"

    if "star rating" in lower_query or "rated" in lower_query:
        rating_match = re.search(r"(\d+)\+ star ratings?", lower_query)
        if rating_match:
            entities["min_rating"] = float(rating_match.group(1))
            intent = "refine"

    if "why did you recommend" in lower_query or "explain" in lower_query:
        intent = "explain"
        prod_name_match = re.search(r"why did you recommend the (.*?)\?", lower_query)
        if prod_name_match:
            entities["product_name"] = prod_name_match.group(1)

    # Simulate explanation generation
    if intent == "explain" and "product_name" in entities:
        # Find the product based on name (simple match)
        product = next((p for p in PRODUCTS_DB if entities["product_name"].lower() in p["name"].lower()), None)
        if product:
            explanation_text = f"The {product['name']} was recommended because you were looking for items like '{context.get('last_query', user_query)}' and this product features {', '.join(product['features'])}. It's highly rated at {product['rating']} stars and costs ${product['price']:.2f}."
            response_text = f"Here's why I recommended the {product['name']}:"
        else:
            explanation_text = f"I couldn't find details for '{entities['product_name']}' in the last recommendations."
            response_text = explanation_text

    if intent == "recommendation":
        response_text = f"Okay, I'll look for {entities.get('category', 'products')} based on your request."
    elif intent == "refine":
        response_text = "Sure, let me refine the recommendations for you."


    await asyncio.sleep(0.1)  # Simulate LLM processing time
    return {
        "intent": intent,
        "entities": entities,
        "llm_response_text": response_text,
        "explanation": explanation_text
    }

# --- 4. Recommendation Engine Layer ---

def apply_filters(products: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Applies dynamic filters to a list of products."""
    filtered_products = products

    if "category" in filters:
        filtered_products = [p for p in filtered_products if p["category"].lower() == filters["category"].lower()]
    if "brand" in filters:
        filtered_products = [p for p in filtered_products if p["brand"].lower() == filters["brand"].lower()]
    if "min_price" in filters:
        filtered_products = [p for p in filtered_products if p["price"] >= filters["min_price"]]
    if "max_price" in filters:
        filtered_products = [p for p in filtered_products if p["price"] <= filters["max_price"]]
    if "min_rating" in filters:
        filtered_products = [p for p in filtered_products if p["rating"] >= filters["min_rating"]]

    return filtered_products

def generate_recommendations(
    user_query: str,
    current_filters: Dict[str, Any],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Generates recommendations based on user query and filters.
    Combines embedding search with filtering.
    """
    query_embedding = generate_embedding(user_query)
    candidate_products = search_vector_db(query_embedding, top_k=20) # Search a wider net initially

    # Apply current filters
    filtered_products = apply_filters(candidate_products, current_filters)

    # Sort by relevance (embedding similarity is implicit from search, but can re-sort or use rating)
    # For this demo, we'll sort by rating for a bit of diversity after filtering
    filtered_products.sort(key=lambda x: x.get("rating", 0), reverse=True)

    return filtered_products[:top_k]

# --- 5. API & User Interface Layer (FastAPI Backend) ---

app = FastAPI(
    title="Intelligent Conversational E-commerce Recommender",
    description="LLM-enhanced system for personalized product recommendations via chat."
)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_recommender(request: ChatRequest):
    session = get_user_session(request.user_id)
    session["chat_history"].append(request.message)

    # --- LLM Interpretation ---
    llm_output = await simulate_llm_response(
        prompt=request.message, # Simplified prompt
        context=session,
        chat_history=session["chat_history"]
    )

    intent = llm_output["intent"]
    extracted_entities = llm_output["entities"]
    llm_response_text = llm_output["llm_response_text"]
    generated_explanation = llm_output["explanation"]

    recommendations_list: List[ProductRecommendation] = []
    current_filters = session["current_filters"].copy()

    # Update filters based on LLM's entity extraction if intent is 'refine' or initial recommendation
    if intent == "refine" or intent == "recommendation":
        current_filters.update(extracted_entities)
        update_user_session(request.user_id, {"current_filters": current_filters})

    if intent == "recommendation" or intent == "refine":
        # --- Recommendation Generation ---
        # Use current user query for embedding search or last known relevant query
        effective_query = request.message
        if not extracted_entities and session["chat_history"] and len(session["chat_history"]) > 1:
            # If no new entities, try to use context from previous turns for embedding search
            effective_query = session["chat_history"][-2] # Look at previous user message

        raw_recommendations = generate_recommendations(effective_query, current_filters)

        for prod in raw_recommendations:
            # For each recommended product, generate a specific explanation
            # This is a simplified call; in a real app, the LLM would dynamically explain based on context.
            explanation_for_prod = await simulate_llm_response(
                prompt=f"Explain why {prod['name']} is a good recommendation for a user looking for '{effective_query}' with filters {current_filters}",
                context=session,
                chat_history=[] # New context for explanation
            )
            recommendations_list.append(ProductRecommendation(
                id=prod["id"],
                name=prod["name"],
                description=prod["description"],
                price=prod["price"],
                rating=prod["rating"],
                brand=prod["brand"],
                category=prod["category"],
                explanation=explanation_for_prod["explanation"]
            ))
        update_user_session(request.user_id, {"last_recommendations": recommendations_list})


    elif intent == "explain":
        # If the user specifically asked for an explanation of a *previous* recommendation
        target_product_name = extracted_entities.get("product_name")
        if target_product_name:
            found_prod = next((p for p in session["last_recommendations"] if target_product_name.lower() in p.name.lower()), None)
            if found_prod:
                llm_response_text = f"Here's the explanation for {found_prod.name}: {found_prod.explanation}"
            else:
                llm_response_text = f"I can't find '{target_product_name}' in the recent recommendations to explain."
        else:
             llm_response_text = generated_explanation # Use general explanation from LLM


    # If LLM response already provides a good summary, use that, otherwise generate a generic one
    final_response_text = llm_response_text
    if not recommendations_list and not llm_response_text.strip():
        final_response_text = "I'm sorry, I couldn't find any recommendations matching your request or I didn't understand it fully."
    elif recommendations_list and not llm_response_text.strip():
         final_response_text = f"Here are some recommendations based on your request:"


    return ChatResponse(
        response=final_response_text,
        recommendations=recommendations_list if recommendations_list else None,
        user_filters=current_filters if current_filters else None,
        session_id=request.user_id
    )

# To run the FastAPI application:
# uvicorn ecommerce_recommender:app --reload

# Example usage with curl:
# curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"user_id": "user123", "message": "I am looking for a new laptop for work, something fast and lightweight."}'
# curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"user_id": "user123", "message": "Windows, and ideally under $1200. I also do some light video editing, so good performance is key."}'
# curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"user_id": "user123", "message": "Why did you recommend the XYZ Ultrabook?"}'
# curl -X POST "http://127.00.0.1:8000/chat" -H "Content-Type: application/json" -d '{"user_id": "user123", "message": "Show me cheaper options, around $800-900."}'

