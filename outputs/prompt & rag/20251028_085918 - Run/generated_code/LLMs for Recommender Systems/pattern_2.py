from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import random

# Initialize FastAPI app
app = FastAPI(title="LLM-Enhanced E-commerce Recommender")

# --- 1. Data Layer ---
# Simulate product data
products_data = {
    "product_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
    "name": ["Laptop Pro", "Gaming Mouse X", "Wireless Keyboard Z", "4K Monitor M", "Webcam HD", "Noise-Cancelling Headphones"],
    "description": [
        "High-performance laptop with 16GB RAM and 1TB SSD. Ideal for professionals and gamers.",
        "Ergonomic gaming mouse with customizable RGB lighting and high precision sensor.",
        "Sleek wireless keyboard with mechanical keys and long battery life.",
        "Stunning 27-inch 4K UHD monitor with HDR support and fast refresh rate.",
        "Full HD 1080p webcam with autofocus and built-in microphone for clear video calls.",
        "Premium over-ear headphones with active noise cancellation and superb audio quality."
    ],
    "category": ["Electronics", "Electronics", "Electronics", "Electronics", "Electronics", "Audio"],
    "price": [1500, 75, 120, 450, 60, 250]
}
products_df = pd.DataFrame(products_data)
products_df.set_index('product_id', inplace=True)

# Simulate user interaction data
user_interactions_data = {
    "user_id": ["u1", "u1", "u2", "u2", "u1", "u3"],
    "product_id": ["p1", "p3", "p2", "p4", "p5", "p1"],
    "interaction_type": ["purchase", "view", "add_to_cart", "view", "purchase", "view"],
    "timestamp": pd.to_datetime([
        "2023-01-15", "2023-01-20", "2023-02-01", "2023-02-05", "2023-03-10", "2023-03-12"
    ])
}
user_interactions_df = pd.DataFrame(user_interactions_data)

# --- 2. Embedding Generation & Vector Store ---
print("Loading Sentence Transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Generating product embeddings...")
product_descriptions = products_df['description'].tolist()
product_embeddings = embedding_model.encode(product_descriptions)

# Build FAISS index
embedding_dim = product_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(np.array(product_embeddings).astype('float32'))

# Map FAISS index to product IDs
product_id_map = {i: pid for i, pid in enumerate(products_df.index.tolist())}

# --- 4. LLM Enhancement Module ---
print("Loading LLM for explanations...")
# Using a small local model for demonstration. For production, consider larger models via API.
# pipeline("text2text-generation", model="t5-small") or pipeline("text-generation", model="gpt2")
llm_explainer = pipeline("text2generation", model="sshleifer/tiny-gpt2") # Using tiny-gpt2 as a lightweight example

# Helper function to generate explanations
def generate_explanation(product_name: str, product_description: str, user_preference_context: str = "") -> str:
    prompt = f"Explain why a user might like '{product_name}' based on its description: '{product_description}'."
    if user_preference_context:
        prompt += f" Consider the user's preference for: {user_preference_context}."
    prompt += " Be concise and persuasive."
    
    # The LLM output usually contains the prompt + generated text
    # We need to extract only the generated part.
    try:
        response = llm_explainer(prompt, max_new_tokens=50, num_return_sequences=1, do_sample=True, temperature=0.7)
        explanation = response[0]['generated_text'].replace(prompt, "").strip()
        # Simple post-processing to remove potential incomplete sentences or stray text
        explanation = explanation.split('\n')[0].split('.')[0] + '.' if '.' in explanation else explanation
        return explanation
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return f"This product, {product_name}, is highly rated and popular."

# Placeholder for dynamic adaptability based on feedback
def adapt_recommendation_strategy(user_feedback: dict):
    # In a real system, this would involve updating model parameters, fine-tuning LLM, etc.
    print(f"Received feedback for adaptation: {user_feedback}")
    # Example: LLM could suggest adjusting weighting for categories, or re-ranking logic
    feedback_prompt = f"Analyze this user feedback: '{user_feedback}'. Suggest how to adapt a recommendation system strategy to better serve this user or similar users. Focus on adjusting category preferences or recommendation diversity.\nSuggestions: "
    
    try:
        response = llm_explainer(feedback_prompt, max_new_tokens=100, num_return_sequences=1, do_sample=True, temperature=0.7)
        adaptation_suggestion = response[0]['generated_text'].replace(feedback_prompt, "").strip()
        print(f"LLM Adaptation Suggestion: {adaptation_suggestion}")
        return {"status": "Feedback processed", "suggestion": adaptation_suggestion}
    except Exception as e:
        print(f"Error generating adaptation suggestion: {e}")
        return {"status": "Feedback processed, no LLM suggestion due to error."}

# --- Pydantic Models for API Requests ---
class RecommendRequest(BaseModel):
    user_id: str
    query: str = ""
    num_recommendations: int = 3

class FeedbackRequest(BaseModel):
    user_id: str
    product_id: str
    rating: int = None  # 1-5
    feedback_text: str = ""
    interaction_type: str = "implicit" # e.g., 'click', 'skip', 'positive', 'negative'

class Recommendation(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    description: str
    explanation: str

class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]

class FeedbackResponse(BaseModel):
    status: str
    suggestion: str = None

# --- 5. API Layer ---
@app.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    user_id = request.user_id
    query = request.query
    num_recommendations = request.num_recommendations

    # Get user's past interactions to infer preferences
    user_history = user_interactions_df[user_interactions_df['user_id'] == user_id]
    user_preference_context = ""
    if not user_history.empty:
        # Simple heuristic: preferred categories are those most interacted with
        preferred_categories = user_history['product_id'].apply(lambda pid: products_df.loc[pid, 'category']).value_counts().index.tolist()
        if preferred_categories:
            user_preference_context = f"The user has previously shown interest in: {', '.join(preferred_categories)} categories."

    candidate_product_ids = []

    if query: # Use query for content-based similarity
        query_embedding = embedding_model.encode([query])
        D, I = faiss_index.search(np.array(query_embedding).astype('float32'), num_recommendations * 2) # Get more candidates
        candidate_indices = I[0]
        candidate_product_ids.extend([product_id_map[idx] for idx in candidate_indices])
    
    # Fallback/補充: If no query or not enough candidates, recommend popular or diverse items
    if not query or len(candidate_product_ids) < num_recommendations:
        # Add some random popular products or products from preferred categories
        popular_products = user_interactions_df['product_id'].value_counts().head(num_recommendations).index.tolist()
        for pid in popular_products:
            if pid not in candidate_product_ids:
                candidate_product_ids.append(pid)
        
        # Add products from preferred categories if any
        if user_preference_context:
             for cat in preferred_categories:
                 cat_products = products_df[products_df['category'] == cat].index.tolist()
                 random.shuffle(cat_products)
                 for pid in cat_products:
                     if pid not in candidate_product_ids:
                         candidate_product_ids.append(pid)
                         break # Add one from category

    # Ensure unique recommendations and limit to num_recommendations
    recommended_product_ids = list(dict.fromkeys(candidate_product_ids))[:num_recommendations]

    if not recommended_product_ids:
        # If still no recommendations, return some random products
        random_pids = random.sample(products_df.index.tolist(), min(num_recommendations, len(products_df)))
        recommended_product_ids.extend(random_pids)
        recommended_product_ids = list(dict.fromkeys(recommended_product_ids))[:num_recommendations]

    recommendations_list = []
    for pid in recommended_product_ids:
        product_info = products_df.loc[pid].to_dict()
        explanation = generate_explanation(
            product_name=product_info['name'],
            product_description=product_info['description'],
            user_preference_context=user_preference_context
        )
        recommendations_list.append(Recommendation(
            product_id=pid,
            name=product_info['name'],
            category=product_info['category'],
            price=product_info['price'],
            description=product_info['description'],
            explanation=explanation
        ))
    
    return RecommendResponse(recommendations=recommendations_list)


@app.post("/feedback", response_model=FeedbackResponse)
async def post_feedback(request: FeedbackRequest):
    # In a real system, save feedback to a database
    # For this demo, we just print and pass to LLM for adaptation suggestion
    feedback_data = request.dict()
    print(f"User {request.user_id} provided feedback for product {request.product_id}: {feedback_data}")
    
    # LLM processes feedback for adaptation suggestion
    adaptation_response = adapt_recommendation_strategy(feedback_data)
    
    return FeedbackResponse(
        status=adaptation_response['status'],
        suggestion=adaptation_response.get('suggestion')
    )


# To run the application:
# 1. Save this code as main.py
# 2. Run: pip install fastapi uvicorn pandas numpy sentence-transformers faiss-cpu transformers pydantic
# 3. Run: uvicorn main:app --reload
# 4. Access API at http://127.0.0.1:8000/docs