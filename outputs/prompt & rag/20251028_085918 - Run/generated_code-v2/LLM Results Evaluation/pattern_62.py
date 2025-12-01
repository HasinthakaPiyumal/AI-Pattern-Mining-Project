import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import random
import time # For simulating chronological order

# --- Configuration ---
# In a real application, these would come from environment variables or a config file
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY" # Placeholder
LLM_MODEL_NAME = "gpt-3.5-turbo" # Placeholder

# --- Data Models ---
class Exemplar(BaseModel):
    id: str
    customer_query: str
    chatbot_response: str
    timestamp: float = None # For chronological ordering
    embedding: Optional[List[float]] = None

class CustomerQueryRequest(BaseModel):
    query: str
    num_exemplars: int = 5
    ordering_strategy: str = "similarity" # "similarity", "diversity", "chronological", "random"

class ChatbotResponse(BaseModel):
    response: str
    prompt_used: str
    exemplar_order_strategy: str

# --- Core Application Logic ---
app = FastAPI(
    title="Intelligent Chatbot Response Optimizer",
    description="Optimizes chatbot responses for e-commerce customer support using dynamic exemplar ordering."
)

# Initialize Sentence Transformer for embedding
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}. Please ensure you have it installed or handle network issues.")
    print("Using a mock embedding function.")
    embedding_model = None

# In-memory exemplar store (simulating a vector database)
# In a real system, this would be a persistent database (e.g., Pinecone, Chroma, FAISS)
exemplar_store: List[Exemplar] = []

def _generate_mock_exemplars(num_examples: int = 20):
    global exemplar_store
    sample_queries = [
        "Where is my order?", "I want to return an item.", "How do I change my shipping address?",
        "Do you offer international shipping?", "My product arrived damaged.", "How do I track my package?",
        "What is your refund policy?", "Can I cancel my order?", "How do I apply a discount code?",
        "Do you have this item in a different size?", "My payment failed.", "When will this item be back in stock?",
        "How do I contact customer service?", "What are your business hours?", "I received the wrong item.",
        "How do I create an account?", "I forgot my password.", "What payment methods do you accept?",
        "Can I get a gift receipt?", "How do I subscribe to your newsletter?"
    ]
    sample_responses = [
        "Please provide your order number and I can check its status for you.",
        "You can initiate a return through your account page or by contacting support.",
        "To change your shipping address, please log into your account before dispatch.",
        "Yes, we offer international shipping to select countries.",
        "We apologize for the inconvenience. Please send us photos of the damaged item.",
        "You can track your package using the tracking number sent to your email.",
        "Our refund policy allows returns within 30 days of purchase.",
        "Orders can be canceled if they haven't been processed for shipping yet.",
        "Enter your discount code at checkout in the 'Promo Code' field.",
        "Please check the product page for available sizes or sign up for notifications.",
        "Please try again or use an alternative payment method.",
        "We regularly restock popular items. Sign up for email alerts on the product page.",
        "You can contact us via live chat, email, or phone during business hours.",
        "Our business hours are Monday-Friday, 9 AM - 5 PM EST.",
        "We apologize. Please provide your order details so we can correct this.",
        "You can create an account by clicking 'Sign Up' at the top right of our website.",
        "Click 'Forgot Password' on the login page to reset it.",
        "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
        "Yes, you can select the gift receipt option during checkout.",
        "Enter your email in the newsletter sign-up section at the bottom of our homepage."
    ]

    if embedding_model:
        # Combine query and response for a more comprehensive embedding context
        texts_to_embed = [f"Customer: {q}\nChatbot: {r}" for q, r in zip(sample_queries, sample_responses)]
        embeddings = embedding_model.encode(texts_to_embed, convert_to_numpy=True).tolist()
    else:
        # Mock embeddings if model failed to load
        embeddings = [np.random.rand(384).tolist() for _ in range(len(sample_queries))]

    for i, (q, r, emb) in enumerate(zip(sample_queries, sample_responses, embeddings)):
        exemplar_store.append(Exemplar(
            id=f"ex_{i+1}",
            customer_query=q,
            chatbot_response=r,
            timestamp=time.time() - (len(sample_queries) - i) * 3600, # Simulate older examples first
            embedding=emb
        ))
    print(f"Generated {len(exemplar_store)} mock exemplars.")


@app.on_event("startup")
async def startup_event():
    _generate_mock_exemplars()

def get_embedding(text: str) -> List[float]:
    if embedding_model:
        return embedding_model.encode(text, convert_to_numpy=True).tolist()
    else:
        # Mock embedding for robustness
        return np.random.rand(384).tolist()

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    dot_product = np.dot(vec1_np, vec2_np)
    norm_vec1 = np.linalg.norm(vec1_np)
    norm_vec2 = np.linalg.norm(vec2_np)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def retrieve_relevant_exemplars(query_embedding: List[float], num_exemplars: int) -> List[Exemplar]:
    # Calculate similarity to all exemplars
    similarities = []
    for ex in exemplar_store:
        if ex.embedding:
            sim = cosine_similarity(query_embedding, ex.embedding)
            similarities.append((sim, ex))
    
    # Sort by similarity and take top N
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [ex for sim, ex in similarities[:num_exemplars]]

def order_exemplars(exemplars: List[Exemplar], strategy: str, query_embedding: Optional[List[float]] = None) -> List[Exemplar]:
    if not exemplars:
        return []

    if strategy == "similarity" and query_embedding:
        # Order by similarity to the current query (most similar first)
        exemplars_with_sim = []
        for ex in exemplars:
            if ex.embedding:
                sim = cosine_similarity(query_embedding, ex.embedding)
                exemplars_with_sim.append((sim, ex))
        exemplars_with_sim.sort(key=lambda x: x[0], reverse=True)
        return [ex for sim, ex in exemplars_with_sim]
    elif strategy == "diversity":
        # A simple diversity strategy: shuffle and pick a subset, or a more advanced approach
        # For this example, we'll implement a simple greedy approach to maximize distance between selected pairs.
        # This is a simplification; true diversity often involves MMR or clustering.
        if len(exemplars) <= 1:
            return exemplars
        
        diverse_exemplars = []
        remaining_exemplars = list(exemplars)
        
        # Start with a random exemplar
        start_ex = remaining_exemplars.pop(random.randrange(len(remaining_exemplars)))
        diverse_exemplars.append(start_ex)
        
        while remaining_exemplars and len(diverse_exemplars) < len(exemplars):
            best_ex_to_add = None
            max_min_dist = -1

            for current_ex in remaining_exemplars:
                if not current_ex.embedding:
                    continue
                # Find the minimum distance to any already selected diverse exemplar
                min_dist_to_selected = float('inf')
                for selected_ex in diverse_exemplars:
                    if selected_ex.embedding:
                        # Using 1 - cosine_similarity as a distance metric
                        dist = 1 - cosine_similarity(current_ex.embedding, selected_ex.embedding)
                        min_dist_to_selected = min(min_dist_to_selected, dist)
                
                if min_dist_to_selected > max_min_dist:
                    max_min_dist = min_dist_to_selected
                    best_ex_to_add = current_ex
            
            if best_ex_to_add:
                diverse_exemplars.append(best_ex_to_add)
                remaining_exemplars.remove(best_ex_to_add)
            else:
                # If no more exemplars can be added based on embeddings, just add remaining randomly
                diverse_exemplars.extend(remaining_exemplars)
                break
        return diverse_exemplars

    elif strategy == "chronological":
        # Order by timestamp, oldest first (or newest first, depending on intent)
        return sorted(exemplars, key=lambda x: x.timestamp if x.timestamp is not None else 0)
    elif strategy == "random":
        random.shuffle(exemplars)
        return exemplars
    else:
        return exemplars # Default to original retrieved order

def construct_few_shot_prompt(customer_query: str, ordered_exemplars: List[Exemplar]) -> str:
    prompt_parts = ["You are an AI assistant for an e-commerce customer support."]
    prompt_parts.append("Your task is to provide helpful and concise answers based on the customer's query.")
    prompt_parts.append("Here are some examples of past customer interactions:")

    for ex in ordered_exemplars:
        prompt_parts.append(f"---")
        prompt_parts.append(f"Customer: {ex.customer_query}")
        prompt_parts.append(f"Chatbot: {ex.chatbot_response}")
    
    prompt_parts.append(f"---")
    prompt_parts.append(f"Customer: {customer_query}")
    prompt_parts.append(f"Chatbot:")
    
    return "\n".join(prompt_parts)

def get_llm_response(prompt: str) -> str:
    # This is a mock function. In a real application, you'd integrate with an LLM API (e.g., OpenAI, Cohere)
    # using 'openai' or 'cohere' libraries.
    print(f"\n--- Mock LLM Call with Prompt (first 200 chars):\n{prompt[:200]}...\n---")
    
    # Simulate LLM processing time
    time.sleep(0.5)
    
    # Simple mock response based on prompt content or a generic one
    if "track" in prompt.lower() and "package" in prompt.lower():
        return "Please provide your tracking number so I can check the status of your package."
    elif "return" in prompt.lower() and "item" in prompt.lower():
        return "You can initiate a return through your order history in your account or by contacting our returns department."
    elif "cancel" in prompt.lower() and "order" in prompt.lower():
        return "Orders can only be canceled before they are processed for shipping. Please provide your order ID to check if it's eligible for cancellation."
    else:
        return f"I understand you have a question about your recent inquiry. How can I assist you further?"

# --- API Endpoint ---
@app.post("/optimize_response", response_model=ChatbotResponse)
async def optimize_chatbot_response(request: CustomerQueryRequest):
    query_embedding = get_embedding(request.query)
    
    # 1. Retrieve relevant exemplars
    retrieved_exemplars = retrieve_relevant_exemplars(query_embedding, request.num_exemplars * 2) # Retrieve more to allow for diversity selection
    
    # 2. Order exemplars based on strategy
    final_exemplars_to_use = order_exemplars(
        retrieved_exemplars,
        request.ordering_strategy,
        query_embedding=query_embedding
    )[:request.num_exemplars] # Take the final N after ordering
    
    # 3. Construct few-shot prompt
    prompt = construct_few_shot_prompt(request.query, final_exemplars_to_use)
    
    # 4. Get LLM response
    llm_response = get_llm_response(prompt)
    
    return ChatbotResponse(
        response=llm_response,
        prompt_used=prompt,
        exemplar_order_strategy=request.ordering_strategy
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "embedding_model_loaded": bool(embedding_model)}

# To run this application:
# 1. pip install fastapi uvicorn sentence-transformers numpy pydantic
# 2. uvicorn main:app --reload
# Then access http://127.0.0.1:8000/docs for the API documentation and to test it.
