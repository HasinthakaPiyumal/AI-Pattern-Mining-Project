
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import random

# Pydantic Models
class UserProfile(BaseModel):
    user_id: str
    location: str # e.g., "Japan", "USA", "India"
    language: str # e.g., "Japanese", "English", "Hindi"
    gender: Optional[str] = None
    age: Optional[int] = None
    preferences: List[str] = [] # e.g., ["tea", "electronics", "traditional crafts"]
    cultural_tags: List[str] = [] # e.g., ["anime_lover", "yoga_enthusiast"]

class Product(BaseModel):
    product_id: str
    name: str
    description: str
    category: str
    price: float
    image_url: Optional[str] = None
    cultural_tags: List[str] = [] # e.g., ["traditional_japanese", "sustainable_indian"]
    demographic_affinity: Dict[str, float] = {} # e.g., {"gender:female": 0.8, "age:25-35": 0.7}

class Recommendation(BaseModel):
    product_id: str
    name: str
    culturally_aware_description: str
    relevance_score: float
    cultural_relevance: float

app = FastAPI()

# --- Mock Databases and Models ---
# In-memory databases for demonstration
user_profiles_db: Dict[str, UserProfile] = {}
product_catalog_db: Dict[str, Product] = {}
product_embeddings_db: Dict[str, np.ndarray] = {}

# Initialize Embedding Model
# For a real application, consider a more robust model or fine-tuning
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Placeholder for LLM for cultural content generation
# In a real system, this would involve calling a hosted LLM service
def mock_llm_generate_culturally_aware_content(prompt: str) -> str:
    """Simulates LLM generating culturally aware product descriptions."""
    # A very simplified mock: just appends cultural context to a generic description
    if "Japanese tea ceremony" in prompt:
        return "Experience the tranquility of a traditional Japanese tea ceremony with this exquisite matcha set, perfect for gifting during New Year's or family gatherings."
    elif "Indian festival" in prompt:
        return "Celebrate with vibrant colors! This festive attire is perfect for Diwali celebrations and traditional Indian weddings."
    elif "Korean skincare" in prompt:
        return "Achieve glowing skin with this innovative Korean skincare routine, a staple for beauty enthusiasts worldwide."
    else:
        return f"A high-quality product: {prompt.replace('Generate a culturally aware product description for ', '').replace(' considering cultural context: ', '')}"


def _generate_embedding(text: str) -> np.ndarray:
    """Generates a vector embedding for a given text."""
    return embedding_model.encode(text, convert_to_tensor=False)


def _get_cultural_context_from_user_profile(user_profile: UserProfile) -> str:
    """Extracts a string representing cultural context from user profile."""
    context_parts = []
    if user_profile.location: context_parts.append(user_profile.location)
    if user_profile.language: context_parts.append(user_profile.language)
    if user_profile.cultural_tags: context_parts.extend(user_profile.cultural_tags)
    if user_profile.preferences: context_parts.extend(user_profile.preferences)
    return ", ".join(context_parts)


def _calculate_relevance(query_embedding: np.ndarray, product_embedding: np.ndarray) -> float:
    """Calculates cosine similarity as a relevance score."""
    return np.dot(query_embedding, product_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(product_embedding))


def apply_fairness_re_ranking(
    recommendations: List[Dict],
    user_profile: UserProfile,
    product_catalog: Dict[str, Product]
) -> List[Dict]:
    """Applies a simplified fairness re-ranking based on demographic affinity and diversity.
    
    This is a conceptual implementation. Real-world fairness re-ranking would involve
    more sophisticated algorithms, group fairness metrics, and potentially a separate
    model trained to reduce specific biases.
    """
    # Sort initially by relevance score
    ranked_recommendations = sorted(recommendations, key=lambda x: x['relevance_score'], reverse=True)

    # Simple diversity injection: re-shuffle top N to avoid monocultural/monodemographic bias
    # In a real system, this would be more strategic, e.g., ensuring representation
    # across categories, cultural origins, or target demographics if user_profile suggests it.
    top_n_to_diversify = min(len(ranked_recommendations), 5) # Diversify top 5 for example
    to_diversify = ranked_recommendations[:top_n_to_diversify]
    remaining = ranked_recommendations[top_n_to_diversify:]
    random.shuffle(to_diversify)

    final_recommendations = to_diversify + remaining

    # Further (conceptual) re-ranking based on user-specific fairness criteria
    # For instance, if user has explicitly opted for diverse recommendations
    # or if we detect a lack of diversity for a specific user segment.
    # This part would typically leverage fairness metrics (e.g., disparate impact)
    # and counterfactual reasoning to adjust rankings.
    
    # Example: If a user is from a particular demographic, ensure products with high
    # affinity for that demographic are not unfairly suppressed, or conversely,
    # ensure a broad range if bias towards that demographic is detected.
    
    return final_recommendations

# --- API Endpoints ---

@app.post("/user_profile", response_model=UserProfile)
async def create_or_update_user_profile(profile: UserProfile):
    user_profiles_db[profile.user_id] = profile
    return profile

@app.post("/products", response_model=Product)
async def add_product(product: Product):
    product_catalog_db[product.product_id] = product
    # Generate and store embedding for the product description
    product_embeddings_db[product.product_id] = _generate_embedding(product.description)
    return product

@app.get("/recommendations/{user_id}", response_model=List[Recommendation])
async def get_recommendations(user_id: str, limit: int = 10):
    if user_id not in user_profiles_db:
        raise HTTPException(status_code=404, detail="User not found")

    user_profile = user_profiles_db[user_id]
    
    # --- Cultural Awareness (Prompt Design & LLM Integration) ---
    # This part simulates how a query might be culturally adapted before similarity search
    cultural_context = _get_cultural_context_from_user_profile(user_profile)
    query_text = f"Products for a user interested in {', '.join(user_profile.preferences)} from {user_profile.location} with cultural background: {cultural_context}"
    query_embedding = _generate_embedding(query_text)

    # --- Embedding and Similarity Search (Mock ChromaDB) ---
    # Find top N similar products based on the culturally-aware query embedding
    product_scores = []
    for prod_id, prod_embedding in product_embeddings_db.items():
        if prod_id in product_catalog_db: # Ensure product still exists in catalog
            relevance = _calculate_relevance(query_embedding, prod_embedding)
            product_scores.append({
                "product_id": prod_id,
                "relevance_score": float(relevance), # Convert numpy float to Python float
                "cultural_relevance": random.uniform(0.5, 1.0) # Mock cultural relevance
            })
    
    # Sort by initial relevance
    initial_recommendations = sorted(product_scores, key=lambda x: x['relevance_score'], reverse=True)[:limit*2] # Get more than 'limit' for re-ranking

    # --- Fairness & Bias Mitigation ---
    final_ranked_recommendations = apply_fairness_re_ranking(
        initial_recommendations,
        user_profile,
        product_catalog_db
    )
    
    # Prepare the final list of recommendations with culturally-aware descriptions
    recommendations_output = []
    for rec_item in final_ranked_recommendations[:limit]:
        product = product_catalog_db[rec_item['product_id']]
        
        # Generate culturally-aware description using the mock LLM
        llm_prompt = f"Generate a culturally aware product description for {product.name} (Category: {product.category}, Cultural tags: {', '.join(product.cultural_tags)}) considering user's cultural context: {cultural_context}."
        culturally_aware_description = mock_llm_generate_culturally_aware_content(llm_prompt)
        
        recommendations_output.append(Recommendation(
            product_id=product.product_id,
            name=product.name,
            culturally_aware_description=culturally_aware_description,
            relevance_score=rec_item['relevance_score'],
            cultural_relevance=rec_item['cultural_relevance']
        ))

    return recommendations_output

# Example Usage (after running this script with `uvicorn main:app --reload`):
# 1. Add a user:
#    POST http://127.0.0.1:8000/user_profile
#    {
#        "user_id": "user123",
#        "location": "Japan",
#        "language": "Japanese",
#        "preferences": ["tea", "books"],
#        "cultural_tags": ["anime_fan"]
#    }
#
# 2. Add products:
#    POST http://127.0.0.1:8000/products
#    {
#        "product_id": "prod001",
#        "name": "Matcha Green Tea Set",
#        "description": "Traditional Japanese matcha tea set with whisk and bowl.",
#        "category": "Beverages",
#        "price": 45.0,
#        "cultural_tags": ["traditional_japanese", "tea_ceremony"]
#    }
#
#    POST http://127.0.0.1:8000/products
#    {
#        "product_id": "prod002",
#        "name": "Modern Art Print",
#        "description": "Contemporary abstract art for home decor.",
#        "category": "Home Decor",
#        "price": 120.0
#    }
#
#    POST http://127.0.0.1:8000/products
#    {
#        "product_id": "prod003",
#        "name": "Indian Silk Saree",
#        "description": "Handwoven silk saree, perfect for festivals and weddings.",
#        "category": "Apparel",
#        "price": 200.0,
#        "cultural_tags": ["traditional_indian", "festival_wear"],
#        "demographic_affinity": {"gender:female": 0.9, "age:25-50": 0.8}
#    }
#
#    POST http://127.0.0.1:8000/products
#    {
#        "product_id": "prod004",
#        "name": "Korean Facial Sheet Masks",
#        "description": "A pack of 10 hydrating and brightening Korean facial sheet masks.",
#        "category": "Beauty",
#        "price": 25.0,
#        "cultural_tags": ["k_beauty", "skincare"]
#    }
#
# 3. Get recommendations for the user:
#    GET http://127.0.0.1:8000/recommendations/user123


