from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Intent Recognition Module
model = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-defined intents and example queries
intents_data = {
    "check_order_status": [
        "Where is my order?",
        "Track my package",
        "What is the status of my recent purchase?",
        "Has my item shipped?"
    ],
    "faq_lookup": [
        "How do I return an item?",
        "What is your return policy?",
        "How can I contact support?",
        "Do you offer international shipping?"
    ],
    "reset_password": [
        "I forgot my password",
        "How do I reset my account?",
        "Change my login credentials",
        "Help with password recovery"
    ],
    "greeting": [
        "Hello",
        "Hi there",
        "Good morning",
        "Hey"
    ],
    "goodbye": [
        "Goodbye",
        "See you later",
        "Bye",
        "Farewell"
    ]
}

# Generate embeddings for pre-defined intent queries
intent_embeddings = {}
for intent, queries in intents_data.items():
    intent_embeddings[intent] = model.encode(queries).mean(axis=0)


def recognize_intent(user_query: str):
    user_embedding = model.encode([user_query])
    
    max_similarity = -1
    best_intent = "unknown"
    
    for intent, embedding in intent_embeddings.items():
        similarity = cosine_similarity(user_embedding, embedding.reshape(1, -1))[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_intent = intent
            
    return best_intent, max_similarity

# 2. Knowledge Base (Mock)
knowledge_base = {
    "return_policy": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
    "contact_support": "You can contact support via email at support@example.com or call us at 1-800-123-4567.",
    "shipping_info": "We offer standard and express shipping. International shipping is available to select countries."
}

# 3. Action Execution Module (Mock)
def check_order_status(user_query: str):
    return "Please provide your order number to check the status."

def faq_lookup(user_query: str):
    if "return" in user_query.lower():
        return knowledge_base["return_policy"]
    elif "contact" in user_query.lower():
        return knowledge_base["contact_support"]
    elif "shipping" in user_query.lower():
        return knowledge_base["shipping_info"]
    else:
        return "I can help with general FAQs. Can you be more specific?"

def reset_password(user_query: str):
    return "To reset your password, please visit our website and click on 'Forgot Password' link."

def greet_user():
    return "Hello! How can I help you today?"

def say_goodbye():
    return "Goodbye! Have a great day."


# 4. Dialogue Manager
INTENT_THRESHOLD = 0.65

def dialogue_manager(intent: str, confidence: float, user_query: str):
    if confidence < INTENT_THRESHOLD:
        return "I am not sure I understood your request. Can you please rephrase or provide more details?"
    
    if intent == "check_order_status":
        return check_order_status(user_query)
    elif intent == "faq_lookup":
        return faq_lookup(user_query)
    elif intent == "reset_password":
        return reset_password(user_query)
    elif intent == "greeting":
        return greet_user()
    elif intent == "goodbye":
        return say_goodbye()
    else:
        return "I'm sorry, I can't help with that at the moment."


# 5. FastAPI Application
app = FastAPI()

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat(chat_message: ChatMessage):
    user_query = chat_message.message
    
    intent, confidence = recognize_intent(user_query)
    response = dialogue_manager(intent, confidence, user_query)
    
    return {"response": response}

# To run this application:
# 1. Save the code as chatbot_app.py
# 2. Install necessary libraries: pip install fastapi uvicorn sentence-transformers scikit-learn
# 3. Run from your terminal: uvicorn chatbot_app:app --reload
# 4. Access the API at http://127.0.0.1:8000/docs for Swagger UI. 