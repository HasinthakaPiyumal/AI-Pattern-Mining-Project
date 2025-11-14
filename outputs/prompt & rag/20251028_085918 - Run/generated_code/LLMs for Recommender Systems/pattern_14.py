import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os

# --- 1. Data Layer --- 
# Simulated Product Data
products_data = [
    {"product_id": "P001", "name": "Smartwatch X", "description": "A sleek smartwatch with health tracking, notifications, and long battery life.", "category": "Electronics", "price": 199.99},
    {"product_id": "P002", "name": "Wireless Earbuds Pro", "description": "Premium wireless earbuds with noise cancellation and crystal-clear audio.", "category": "Electronics", "price": 129.99},
    {"product_id": "P003", "name": "Ergonomic Office Chair", "description": "Comfortable office chair with lumbar support and adjustable features for long working hours.", "category": "Home & Office", "price": 249.00},
    {"product_id": "P004", "name": "Gaming Keyboard RGB", "description": "Mechanical gaming keyboard with customizable RGB lighting and tactile switches.", "category": "Electronics", "price": 99.50},
    {"product_id": "P005", "name": "Yoga Mat Deluxe", "description": "Thick, non-slip yoga mat made from eco-friendly materials for all types of yoga.", "category": "Fitness", "price": 35.00},
    {"product_id": "P006", "name": "Stainless Steel Water Bottle", "description": "Insulated water bottle keeping drinks cold for 24 hours and hot for 12 hours.", "category": "Kitchen & Dining", "price": 25.00},
    {"product_id": "P007", "name": "Noise-Cancelling Headphones", "description": "Over-ear headphones with superior noise cancellation and immersive sound.", "category": "Electronics", "price": 299.99},
    {"product_id": "P008", "name": "Adventure Backpack 30L", "description": "Durable and spacious backpack for hiking and travel, with multiple compartments.", "category": "Outdoor", "price": 75.00},
    {"product_id": "P009", "name": "Espresso Machine Compact", "description": "Compact espresso maker for delicious coffee at home, easy to use and clean.", "category": "Kitchen & Dining", "price": 150.00},
    {"product_id": "P010", "name": "Smart LED Strip Lights", "description": "App-controlled LED strip lights with millions of colors for ambiance and decoration.", "category": "Smart Home", "price": 45.00},
]
products_df = pd.DataFrame(products_data)

# Simulated User Interaction Data
user_history_data = {
    "user1": ["P001", "P002", "P004"],  # Interested in electronics
    "user2": ["P003", "P006"],          # Interested in home goods and utilities
    "user3": ["P005", "P008"],          # Interested in fitness and outdoors
    "user4": ["P001", "P007", "P002"]   # Heavy electronics user
}

# --- 2. Embedding Generation ---
# Load Sentence-BERT model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for product descriptions
product_descriptions = products_df['description'].tolist()
product_embeddings = embedding_model.encode(product_descriptions, convert_to_tensor=True)

# Map product ID to its index in the embeddings array
product_id_to_idx = {pid: idx for idx, pid in enumerate(products_df['product_id'])}

# --- Mock LLM (Simulating an OpenAI LLM for demonstration) ---
# In a real application, this would involve actual API calls to an LLM provider (e.g., OpenAI, Gemini).
# We are simulating its behavior as calling external APIs is outside the scope of code generation.
class MockLLM:
    def __init__(self):
        pass

    def generate_recommendations(self, user_id: str, semantically_relevant_products: List[Dict], user_history_products: List[Dict], current_query: Optional[str] = None) -> Dict:
        print(f"MockLLM: Generating recommendations for user {user_id} based on query '{current_query}'...")
        
        # Simple mock logic: prioritize items from semantically relevant, avoid history, and add some variety
        recommended_products = []
        explanation_phrases = []

        history_pids = [p['product_id'] for p in user_history_products]
        
        # Try to recommend relevant products not in history
        for prod in semantically_relevant_products:
            if prod['product_id'] not in history_pids and len(recommended_products) < 3:
                recommended_products.append(prod)
                explanation_phrases.append(f"'{prod['name']}' because it's similar to items you've shown interest in and fits your current search.")
        
        # If not enough, just pick some top relevant ones regardless of history (for variety in mock)
        if len(recommended_products) < 3:
            for prod in semantically_relevant_products:
                if len(recommended_products) < 3 and prod not in recommended_products:
                    recommended_products.append(prod)
                    explanation_phrases.append(f"'{prod['name']}' is a top match for your preferences.")

        if not recommended_products:
            # Fallback for no relevant products
            fallback_products = products_df.sample(3).to_dict(orient='records')
            for prod in fallback_products:
                recommended_products.append(prod)
                explanation_phrases.append(f"'{prod['name']}' is a popular choice you might like.")

        recommendation_text = f"Based on your activity and interests (and your query: '{current_query}' if provided), we recommend: " + ", ".join([p['name'] for p in recommended_products]) + "."
        explanation_text = "Here's why: " + " ".join(explanation_phrases) + " This selection aims to provide you with new and relevant options that align with your taste."

        return {"recommendations": recommended_products, "explanation": explanation_text}

    def generate_chat_response(self, query: str, user_id: str) -> str:
        print(f"MockLLM: Generating chat response for user {user_id} with query: '{query}'")
        query = query.lower()
        if "hello" in query or "hi" in query:
            return "Hello! How can I assist you with your shopping today?"
        elif "recommend" in query and "product" in query:
            return "I can help with recommendations! What kind of products are you looking for, or what's your budget?"
        elif "smartwatch" in query or "electronics" in query:
            return "We have a great selection of smartwatches and other electronics. Are you looking for a specific brand or feature?"
        elif "thank you" in query or "thanks" in query:
            return "You're welcome! Is there anything else?"
        else:
            return "I'm not sure I understand. Can you please rephrase or ask about products or categories?"

mock_llm = MockLLM()

# --- Recommendation Core Functions ---
def get_product_details_by_ids(product_ids: List[str]) -> List[Dict]:
    return products_df[products_df['product_id'].isin(product_ids)].to_dict(orient='records')

def get_semantic_similar_products(query_embedding, top_n: int = 5) -> List[Dict]:
    similarities = cosine_similarity(query_embedding.reshape(1, -1), product_embeddings.cpu().numpy())[0]
    top_indices = similarities.argsort()[-top_n:][::-1] # Get indices of top N similar products
    top_product_ids = [products_df.iloc[idx]['product_id'] for idx in top_indices]
    return get_product_details_by_ids(top_product_ids)

def get_user_profile_embedding(user_id: str):
    if user_id in user_history_data and user_history_data[user_id]:
        history_pids = user_history_data[user_id]
        history_indices = [product_id_to_idx[pid] for pid in history_pids if pid in product_id_to_idx]
        if history_indices:
            return np.mean([product_embeddings[idx].cpu().numpy() for idx in history_indices], axis=0)
    return None


def generate_personalized_recommendations(
    user_id: str, 
    query: Optional[str] = None, 
    top_n_semantic: int = 10,
    top_n_final: int = 5
) -> Dict:
    user_history_products_details = get_product_details_by_ids(user_history_data.get(user_id, []))
    semantically_relevant_products = []
    query_embedding = None

    if query:
        query_embedding = embedding_model.encode(query, convert_to_tensor=True)
        semantically_relevant_products = get_semantic_similar_products(query_embedding.cpu().numpy(), top_n=top_n_semantic)
    else:
        # If no specific query, try to use user's historical interest
        user_profile_emb = get_user_profile_embedding(user_id)
        if user_profile_emb is not None:
            semantically_relevant_products = get_semantic_similar_products(user_profile_emb, top_n=top_n_semantic)
        else:
            # Fallback: if no query and no history, just get some popular items or random ones
            print(f"No specific query or history for user {user_id}. Recommending popular items.")
            semantically_relevant_products = products_df.sample(top_n_semantic).to_dict(orient='records')

    # LLM for Recommendation Refinement and Explanation
    llm_output = mock_llm.generate_recommendations(
        user_id=user_id,
        semantically_relevant_products=semantically_relevant_products,
        user_history_products=user_history_products_details,
        current_query=query
    )
    
    # The mock LLM already refines and explains, so we just return its output
    return llm_output


# --- 5. API Layer with FastAPI ---
app = FastAPI(title="LLM Enhanced E-commerce Recommender")

class RecommendationRequest(BaseModel):
    user_id: str
    query: Optional[str] = None

class ProductRecommendation(BaseModel):
    product_id: str
    name: str
    description: str
    category: str
    price: float

class RecommendationResponse(BaseModel):
    recommendations: List[ProductRecommendation]
    explanation: str

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str


@app.post("/recommendations", response_model=RecommendationResponse, summary="Get personalized product recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Provides personalized product recommendations with explanations."""
    try:
        result = generate_personalized_recommendations(
            user_id=request.user_id,
            query=request.query
        )
        # Ensure the recommended products match the ProductRecommendation Pydantic model structure
        formatted_recommendations = [
            ProductRecommendation(**prod)
            for prod in result['recommendations']
        ]
        return RecommendationResponse(
            recommendations=formatted_recommendations,
            explanation=result['explanation']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, summary="Interact with the conversational shopping assistant")
async def chat(request: ChatRequest):
    """Engages with an AI-powered conversational shopping assistant."""
    try:
        response_text = mock_llm.generate_chat_response(request.message, request.user_id)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # To run this application, save it as `main.py` and then run:
    # pip install fastapi uvicorn pandas numpy scikit-learn sentence-transformers
    # uvicorn main:app --reload
    print("\n--- LLM Enhanced E-commerce Recommender System --- ")
    print("To start the server, run: uvicorn main:app --reload")
    print("Access the API documentation at: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
