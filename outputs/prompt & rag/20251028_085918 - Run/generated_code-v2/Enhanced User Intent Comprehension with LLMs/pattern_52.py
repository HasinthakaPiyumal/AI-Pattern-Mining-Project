from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List
import random

# --- 1. Mock Intent Recognition Module (Core LLM) ---
# In a real scenario, this would load a fine-tuned transformers model.
class MockIntentRecognizer:
    def __init__(self):
        self.intents = {
            "check_order_status": ["where is my order", "track my package", "order status", "delivery update"],
            "return_item": ["how to return", "return policy", "send back an item", "refund request"],
            "product_inquiry": ["tell me about product X", "specifications", "is this compatible", "product details"],
            "technical_support": ["troubleshooting", "my device is not working", "technical issue", "help with installation"],
            "greeting": ["hello", "hi", "hey", "good morning"],
            "unknown": [] # For unhandled queries
        }

    def predict_intent(self, query: str) -> Dict[str, float]:
        query_lower = query.lower()
        predicted_scores = {}
        matched_intents = []

        # Simple keyword matching for demonstration
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in query_lower:
                    matched_intents.append(intent)
                    break

        if not matched_intents:
            return {"unknown": 1.0}

        # Simulate confidence - higher for more direct matches, lower for ambiguous
        if len(matched_intents) == 1:
            predicted_scores[matched_intents[0]] = random.uniform(0.85, 0.99)
        else:
            # Simulate ambiguity: assign lower, close scores to multiple intents
            for intent in matched_intents:
                predicted_scores[intent] = random.uniform(0.4, 0.6)
            # Add 'unknown' if highly ambiguous
            if sum(predicted_scores.values()) < 1.0:
                predicted_scores["unknown"] = 1.0 - sum(predicted_scores.values())
        
        return {k: v for k, v in sorted(predicted_scores.items(), key=lambda item: item[1], reverse=True)}

# --- 2. Mock Knowledge Base (Chroma & Sentence-transformers replacement) ---
# In a real scenario, this would involve a vector DB and embedding model.
class MockKnowledgeBase:
    def __init__(self):
        self.documents = {
            "return_policy": "Our return policy allows returns within 30 days of purchase for a full refund. Items must be in original condition.",
            "shipping_times": "Standard shipping usually takes 5-7 business days. Expedited shipping options are available at checkout.",
            "product_warranty": "Most electronics come with a 1-year manufacturer's warranty. Please check the product page for specific details.",
            "account_creation": "To create an account, click 'Sign Up' in the top right corner and follow the prompts."
        }

    def search_knowledge_base(self, query: str) -> str:
        query_lower = query.lower()
        for key, doc in self.documents.items():
            if any(keyword in query_lower for keyword in key.split('_')) or any(keyword in query_lower for keyword in ["return", "shipping", "warranty", "account"]):
                return doc
        return "I couldn't find specific information about that in our knowledge base. Can you please rephrase or provide more details?"

# --- 3. Mock External Tools ---
class MockExternalTools:
    def check_order_status(self, query: str) -> str:
        # Placeholder for actual API call
        order_id = "#12345" # Simulate extracting an order ID or using user context
        return f"Your order {order_id} is currently being processed and is expected to arrive within 2-3 business days."

    def process_return(self, query: str) -> str:
        # Placeholder for actual API call
        return "I can help you initiate a return. Please provide your order number and the reason for the return."
    
    def get_product_details(self, query: str) -> str:
        # Placeholder for actual API call
        product_name = "" # Simulate extracting product name
        if "laptop" in query.lower():
            product_name = "XPS 15 Laptop"
            return f"The {product_name} features an Intel i7 processor, 16GB RAM, and a 15.6-inch 4K display."
        return "Could you please specify which product you are interested in?"

# --- 4. Customer Support Assistant (Langchain-like Orchestration) ---
class CustomerSupportAssistant:
    def __init__(self):
        self.intent_recognizer = MockIntentRecognizer()
        self.knowledge_base = MockKnowledgeBase()
        self.external_tools = MockExternalTools()
        self.chat_history: List[Dict[str, str]] = []

    def _get_tool_function(self, intent: str):
        if intent == "check_order_status":
            return self.external_tools.check_order_status
        elif intent == "return_item":
            return self.external_tools.process_return
        elif intent == "product_inquiry":
            return self.external_tools.get_product_details
        return None

    def process_query(self, query: str) -> str:
        self.chat_history.append({"role": "user", "content": query})
        
        intent_scores = self.intent_recognizer.predict_intent(query)
        top_intent = next(iter(intent_scores))
        top_score = intent_scores[top_intent]

        response = ""

        if top_intent == "greeting":
            response = "Hello! How can I assist you today?"
        elif top_score < 0.7: # Threshold for ambiguity or low confidence
            ambiguous_intents = [intent for intent, score in intent_scores.items() if score > 0.3 and intent != "unknown"]
            if len(ambiguous_intents) > 1:
                response = f"I'm not entirely sure if you want to {ambiguous_intents[0].replace('_', ' ')} or {ambiguous_intents[1].replace('_', ' ')}. Could you clarify?"
            else:
                response = "I'm having a little trouble understanding your request. Could you please rephrase it or provide more details?"
        elif top_intent == "unknown":
            response = self.knowledge_base.search_knowledge_base(query)
            if "couldn't find specific information" in response:
                 response = "I couldn't quite understand that. Please try asking in a different way, or I can connect you to a human agent if needed."
        else:
            tool_func = self._get_tool_function(top_intent)
            if tool_func:
                response = tool_func(query)
            else:
                response = self.knowledge_base.search_knowledge_base(query)

        self.chat_history.append({"role": "assistant", "content": response})
        return response

# --- FastAPI Application --- 
app = FastAPI()
assistant = CustomerSupportAssistant()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_assistant(request: ChatRequest) -> Dict[str, str]:
    response_message = assistant.process_query(request.message)
    return {"response": response_message}

# To run this FastAPI application:
# 1. Save the code as a Python file (e.g., customer_support_assistant.py)
# 2. Install uvicorn: pip install uvicorn
# 3. Run from your terminal: uvicorn customer_support_assistant:app --reload
# 4. Access the API at http://127.0.0.1:8000/docs for interactive testing.