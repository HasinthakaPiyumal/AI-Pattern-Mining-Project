from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi import FastAPI
import uvicorn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

class Product(BaseModel):
    id: str
    name: str
    description: str
    category: str

class User(BaseModel):
    id: str
    preferences_keywords: List[str]

class RecommendationRequest(BaseModel):
    user_id: str

class RecommendationResponse(BaseModel):
    recommended_product: Product
    explanation: str

class ChatRequest(BaseModel):
    user_query: str
    conversation_history: List[str] = []

class ChatResponse(BaseModel):
    chatbot_response: str

MOCK_PRODUCTS = {
    "P101": Product(id="P101", name="Wireless Earbuds Pro", description="Premium sound quality, noise cancellation, comfortable fit.", category="Electronics"),
    "P102": Product(id="P102", name="Smartwatch X", description="Fitness tracking, heart rate monitor, notifications.", category="Wearables"),
    "P103": Product(id="P103", name="Ergonomic Office Chair", description="Adjustable lumbar support, breathable mesh, durable.", category="Home Office"),
}

MOCK_USERS = {
    "U001": User(id="U001", preferences_keywords=["electronics", "gadgets", "audio"]),
    "U002": User(id="U002", preferences_keywords=["fitness", "health", "accessories"]),
}

def get_mock_recommendation(user_id: str) -> (Product, List[str]):
    if user_id == "U001":
        return MOCK_PRODUCTS["P101"], ["similar_to_past_purchase", "high_rating", "good_for_audio_lovers"]
    elif user_id == "U002":
        return MOCK_PRODUCTS["P102"], ["fitness_related", "new_arrival", "trending"]
    return MOCK_PRODUCTS["P103"], ["general_utility"]

explanation_tokenizer = None
explanation_model = None
chatbot_tokenizer = None
chatbot_model = None

try:
    explanation_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    explanation_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
except Exception:
    pass

try:
    chatbot_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-small-90M")
    chatbot_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-small-90M")
except Exception:
    pass

def generate_explanation(user: User, product: Product, reasons: List[str]) -> str:
    if explanation_model is None or explanation_tokenizer is None:
        return "LLM explanation service is unavailable. Model failed to load."

    reason_str = ", ".join(reasons).replace("_", " ")
    user_pref_str = ", ".join(user.preferences_keywords)

    prompt = (
        f"Generate a personalized explanation for why user {user.id} might like this product. "
        f"User preferences: {user_pref_str}. "
        f"Product: {product.name} ({product.category}) - {product.description}. "
        f"Key reasons: {reason_str}. "
        f"Explanation:"
    )
    
    inputs = explanation_tokenizer(prompt, return_tensors="pt")
    outputs = explanation_model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    return explanation_tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_chatbot_response(user_query: str, conversation_history: List[str]) -> str:
    if chatbot_model is None or chatbot_tokenizer is None:
        return "Chatbot service is unavailable. Model failed to load."
    
    history_context = "\n".join(conversation_history)
    if history_context:
        input_text = f"Customer: {history_context}\nCustomer: {user_query}"
    else:
        input_text = f"Customer: {user_query}"

    inputs = chatbot_tokenizer([input_text], return_tensors="pt")
    outputs = chatbot_model.generate(**inputs, max_new_tokens=100, num_beams=5, early_stopping=True)
    response = chatbot_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if response.startswith("Customer:"):
        response = response.split("Customer:", 1)[-1]
    
    return response.strip()

app = FastAPI()

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendation_with_explanation_endpoint(request: RecommendationRequest):
    user_id = request.user_id
    
    mock_product, mock_reasons = get_mock_recommendation(user_id)
    user_data = MOCK_USERS.get(user_id, User(id=user_id, preferences_keywords=[]))
    
    explanation_text = generate_explanation(user_data, mock_product, mock_reasons)
    
    return RecommendationResponse(recommended_product=mock_product, explanation=explanation_text)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot_endpoint(request: ChatRequest):
    response_text = get_chatbot_response(request.user_query, request.conversation_history)
    return ChatResponse(chatbot_response=response_text)