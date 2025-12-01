from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np
import uvicorn

# 1. Data Management
training_data = [
    {"query": "My internet is not working", "intent": "technical_support"},
    {"query": "I can't access my account", "intent": "account_management"},
    {"query": "How do I pay my bill?", "intent": "billing_issue"},
    {"query": "What are your product prices?", "intent": "product_inquiry"},
    {"query": "I have a general question", "intent": "general_inquiry"},
    {"query": "My computer is frozen", "intent": "technical_support"},
    {"query": "I forgot my password", "intent": "account_management"},
    {"query": "When is my next payment due?", "intent": "billing_issue"},
    {"query": "Tell me more about your new service", "intent": "product_inquiry"},
    {"query": "Hello there", "intent": "general_inquiry"}
]

intent_responses = {
    "technical_support": {"type": "route", "destination": "Technical Support Department"},
    "account_management": {"type": "response", "message": "Please visit our 'Forgot Password' page or contact account management for assistance."},
    "billing_issue": {"type": "route", "destination": "Billing Department"},
    "product_inquiry": {"type": "response", "message": "You can find detailed information about our products on our website under the 'Products' section."},
    "general_inquiry": {"type": "response", "message": "How can I help you further?"}
}

# 2. Intent Recognition Module Setup
model_name = 'all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(model_name)

train_queries = [d["query"] for d in training_data]
train_intents = [d["intent"] for d in training_data]

X_train = embedding_model.encode(train_queries)
y_train = train_intents

classifier = LogisticRegression(random_state=42, solver='liblinear')
classifier.fit(X_train, y_train)

# FastAPI Application
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/predict_intent")
async def predict_intent(request: QueryRequest):
    user_query = request.query
    
    # Generate embedding for the user query
    query_embedding = embedding_model.encode([user_query])
    
    # Predict intent
    predicted_intent = classifier.predict(query_embedding)[0]
    confidence_scores = classifier.predict_proba(query_embedding)[0]
    confidence = np.max(confidence_scores)
    
    # 3. Response Generation/Routing Module
    response_info = intent_responses.get(predicted_intent, {"type": "response", "message": "I'm not sure how to handle that. Please try rephrasing or contact general support."})
    
    if response_info["type"] == "route":
        action = f"Routing to: {response_info['destination']}"
        message = f"Based on your query, I'm routing you to the {response_info['destination']}."
    else:
        action = "Automated Response"
        message = response_info['message']
        
    return {"predicted_intent": predicted_intent, "confidence": float(confidence), "action": action, "message": message}

if __name__ == "__main__":
    # To run this application:
    # 1. Save the code as main.py
    # 2. Install dependencies: pip install fastapi uvicorn sentence-transformers scikit-learn
    # 3. Run from your terminal: uvicorn main:app --reload
    print("To run this application:")
    print("1. Save the code as main.py")
    print("2. Install dependencies: pip install fastapi uvicorn sentence-transformers scikit-learn")
    print("3. Run from your terminal: uvicorn main:app --reload")

