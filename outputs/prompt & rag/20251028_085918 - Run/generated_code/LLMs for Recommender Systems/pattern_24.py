from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import uvicorn


class Product(BaseModel):
    product_id: str
    name: str
    description: str
    category: str
    price: float
    enriched_description: str = ""


class UserPreference(BaseModel):
    user_id: str
    browsing_history: List[str]
    purchase_history: List[str]
    reviews: List[str]
    preference_embedding: List[float] = []


class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 5


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    explanation: Dict[str, str] = {}


class ChatRequest(BaseModel):
    user_id: str
    query: str


class ChatResponse(BaseModel):
    user_id: str
    response: str
    recommended_products: List[Dict[str, Any]] = []


class MockLLM:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)

    def enrich_product_description(self, description: str) -> str:
        return f"[ENRICHED] {description}. Key features: AI-generated insights."

    def generate_explanation(self, product_name: str, reason_context: str) -> str:
        return f"Based on your recent activity and preferences, we recommend {product_name} because {reason_context}. It aligns with products you've shown interest in, offering advanced features and excellent value."

    def understand_user_intent(self, query: str) -> str:
        return f"[INTENT] User is looking for products related to '{query}'."

    def generate_chat_response(self, user_query: str, retrieved_info: List[str]) -> str:
        info_str = "; ".join(retrieved_info) if retrieved_info else "no specific information found"
        return f"You asked about: '{user_query}'. Based on our knowledge, here's what I found related to {info_str}. How else can I assist you?"

    def get_embedding(self, text: str) -> List[float]:
        return self.embedding_model.encode(text).tolist()


app = FastAPI()
llm = MockLLM()

# Mock Data Store
mock_products_data = [
    {"product_id": "P001", "name": "Smartwatch Pro", "description": "Advanced smartwatch with health tracking.", "category": "Electronics", "price": 299.99},
    {"product_id": "P002", "name": "Wireless Earbuds X", "description": "Noise-cancelling wireless earbuds.", "category": "Electronics", "price": 149.99},
    {"product_id": "P003", "name": "Organic Coffee Beans", "description": "Premium single-origin organic coffee.", "category": "Grocery", "price": 19.99},
    {"product_id": "P004", "name": "Ergonomic Office Chair", "description": "Comfortable chair for long working hours.", "category": "Home & Office", "price": 349.99},
    {"product_id": "P005", "name": "4K Smart TV 55 inch", "description": "Ultra HD smart TV with streaming apps.", "category": "Electronics", "price": 799.99},
]

mock_users_data = [
    {"user_id": "U001", "browsing_history": ["P001", "P002"], "purchase_history": ["P001"], "reviews": ["Great smartwatch!", "Looking for more tech gadgets"]},
    {"user_id": "U002", "browsing_history": ["P003", "P004"], "purchase_history": ["P003"], "reviews": ["Love the coffee.", "Need a good chair"]},
]

products: Dict[str, Product] = {p["product_id"]: Product(**p) for p in mock_products_data}
users: Dict[str, UserPreference] = {u["user_id"]: UserPreference(**u) for u in mock_users_data}

# Product Knowledge Base Enrichment
enriched_products: Dict[str, Product] = {}
product_embeddings = []
for product_id, product in products.items():
    product.enriched_description = llm.enrich_product_description(product.description)
    products[product_id] = product
    embedding = llm.get_embedding(product.enriched_description + " " + product.name + " " + product.category)
    product_embeddings.append({"product_id": product_id, "embedding": embedding})

# User Preference Interpretation
for user_id, user_pref in users.items():
    user_text = " ".join(user_pref.reviews + [products[pid].name for pid in user_pref.browsing_history + user_pref.purchase_history if pid in products])
    user_pref.preference_embedding = llm.get_embedding(user_text)
    users[user_id] = user_pref


def get_recommendations_for_user(user_id: str, num_recommendations: int = 5) -> List[Dict[str, Any]]:
    user_pref = users.get(user_id)
    if not user_pref or not user_pref.preference_embedding:
        return []

    user_embedding = np.array(user_pref.preference_embedding)
    similarities = []

    for prod_embed_data in product_embeddings:
        prod_id = prod_embed_data["product_id"]
        prod_embedding = np.array(prod_embed_data["embedding"])
        similarity = np.dot(user_embedding, prod_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(prod_embedding))
        similarities.append((prod_id, similarity))

    # Sort by similarity and get top N
    sorted_products = sorted(similarities, key=lambda x: x[1], reverse=True)
    recommended_product_ids = [pid for pid, _ in sorted_products[:num_recommendations]]

    recommendations = []
    for pid in recommended_product_ids:
        product = products[pid].dict()
        product.pop("enriched_description") # Remove enriched_description for cleaner output
        recommendations.append(product)
    return recommendations


@app.post("/recommend", response_model=RecommendationResponse)
async def get_product_recommendations(request: RecommendationRequest):
    if request.user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")

    recs = get_recommendations_for_user(request.user_id, request.num_recommendations)
    
    explanation_map = {}
    if recs:
        # Generate explanations for the top recommendation as an example
        top_rec_id = recs[0]["product_id"]
        top_rec_name = recs[0]["name"]
        explanation = llm.generate_explanation(top_rec_name, f"you previously bought {users[request.user_id].purchase_history[0]} and browsed {', '.join(users[request.user_id].browsing_history)}")
        explanation_map[top_rec_id] = explanation

    return RecommendationResponse(user_id=request.user_id, recommendations=recs, explanation=explanation_map)


@app.post("/chat", response_model=ChatResponse)
async def conversational_search(request: ChatRequest):
    user_intent = llm.understand_user_intent(request.query)
    
    # Simple RAG: use query embedding to find similar products
    query_embedding = np.array(llm.get_embedding(request.query))
    retrieved_product_info = []
    product_similarity_scores = []

    for prod_embed_data in product_embeddings:
        prod_id = prod_embed_data["product_id"]
        prod_embedding = np.array(prod_embed_data["embedding"])
        similarity = np.dot(query_embedding, prod_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(prod_embedding))
        product_similarity_scores.append((prod_id, similarity))
    
    # Get top 3 most similar products based on the query
    sorted_products = sorted(product_similarity_scores, key=lambda x: x[1], reverse=True)
    top_chat_product_ids = [pid for pid, sim in sorted_products[:3] if sim > 0.5] # Only if similarity is above a threshold

    recommended_for_chat = []
    for pid in top_chat_product_ids:
        product = products[pid]
        retrieved_product_info.append(f"{product.name} ({product.category})")
        recommended_for_chat.append(product.dict())

    llm_response = llm.generate_chat_response(request.query, retrieved_product_info)
    
    return ChatResponse(user_id=request.user_id, response=llm_response, recommended_products=recommended_for_chat)


if __name__ == "__main__":
    # To run: uvicorn ecommerce_recommender:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
