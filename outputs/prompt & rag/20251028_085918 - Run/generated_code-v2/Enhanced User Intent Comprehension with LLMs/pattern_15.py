from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import re

app = FastAPI()

# Simulated NLU Module
def recognize_intent(text: str) -> dict:
    text_lower = text.lower()
    if "order status" in text_lower or "where is my order" in text_lower:
        return {"intent": "order_status", "confidence": 0.9}
    elif "product" in text_lower and ("info" in text_lower or "details" in text_lower):
        return {"intent": "product_inquiry", "confidence": 0.85}
    elif "return" in text_lower or "refund" in text_lower:
        return {"intent": "returns", "confidence": 0.8}
    elif "payment" in text_lower or "bill" in text_lower:
        return {"intent": "payment_issue", "confidence": 0.75}
    elif "hello" in text_lower or "hi" in text_lower:
        return {"intent": "greeting", "confidence": 0.95}
    elif "thank you" in text_lower or "thanks" in text_lower:
        return {"intent": "gratitude", "confidence": 0.95}
    else:
        return {"intent": "unknown", "confidence": 0.4}

def extract_entities(text: str) -> dict:
    entities = {}
    order_id_match = re.search(r'order\s*id\s*(\w+)', text, re.IGNORECASE)
    if order_id_match:
        entities["order_id"] = order_id_match.group(1)
    product_name_match = re.search(r'product\s*"([^"]+)"', text, re.IGNORECASE)
    if product_name_match:
        entities["product_name"] = product_name_match.group(1)
    return entities

# Simulated Knowledge Base / Tool Integration
def get_product_details(product_name: str) -> str:
    products = {"laptop": "The latest XYZ laptop features a 15-inch display, 16GB RAM, and 512GB SSD.",
                "mouse": "This ergonomic wireless mouse offers precise tracking and long battery life.",
                "keyboard": "Mechanical keyboard with customizable RGB lighting and tactile keys."
               }
    return products.get(product_name.lower(), f"Sorry, I can't find details for product '{product_name}'.")

def get_order_status(order_id: str) -> str:
    # Simulate order lookup
    if order_id == "12345":
        return "Your order 12345 is currently out for delivery and expected today."
    elif order_id == "67890":
        return "Your order 67890 was delivered on October 26, 2023."
    else:
        return f"I cannot find any information for order ID '{order_id}'. Please double-check."

def search_faq(query: str) -> str:
    faqs = {
        "shipping": "Standard shipping takes 3-5 business days. Expedited options are available at checkout.",
        "returns": "You can return most items within 30 days of purchase for a full refund. Please visit our returns page for more details.",
        "warranty": "All electronics come with a 1-year manufacturer's warranty.",
        "account": "To reset your password, click 'Forgot Password' on the login page."
    }
    for keyword, answer in faqs.items():
        if keyword in query.lower():
            return answer
    return "I'm sorry, I couldn't find an answer in our FAQs for that query."

# Dialogue Management Module (simple in-memory context)
SESSION_CONTEXT = {}

class ChatRequest(BaseModel):
    session_id: str = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    clarification_needed: bool = False
    suggested_actions: list = []

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    user_message = request.message

    # Get or initialize session context
    context = SESSION_CONTEXT.get(session_id, {"history": [], "last_intent": None, "entities": {}})
    context["history"].append({"user": user_message})

    intent_result = recognize_intent(user_message)
    intent = intent_result["intent"]
    confidence = intent_result["confidence"]
    entities = extract_entities(user_message)

    response = ""
    clarification_needed = False
    suggested_actions = []

    if confidence < 0.6 or intent == "unknown":
        clarification_needed = True
        response = "I'm not sure I fully understand. Could you please rephrase or provide more details?"
        suggested_actions = ["Check order status", "Ask about a product", "Initiate a return"]
    else:
        if intent == "greeting":
            response = "Hello! How can I assist you today?"
        elif intent == "gratitude":
            response = "You're welcome! Is there anything else I can help with?"
        elif intent == "order_status":
            order_id = entities.get("order_id") or context["entities"].get("order_id")
            if order_id:
                response = get_order_status(order_id)
            else:
                clarification_needed = True
                response = "To check your order status, please provide your order ID."
                context["last_intent"] = "order_status"
        elif intent == "product_inquiry":
            product_name = entities.get("product_name")
            if product_name:
                response = get_product_details(product_name)
            else:
                clarification_needed = True
                response = "What product are you interested in? Please provide the product name."
                context["last_intent"] = "product_inquiry"
        elif intent == "returns":
            response = search_faq("returns") # Direct to FAQ for returns
        elif intent == "payment_issue":
            response = "For payment issues, please contact our billing department directly at billing@example.com or call us at 1-800-PAYMENT."
        else:
            response = "I'm still learning, but I'll do my best to help. Can you tell me more?"

    # Update context with current entities and intent if applicable
    if entities: # Only update if new entities were extracted
        context["entities"].update(entities)

    # Handle clarification follow-up
    if not clarification_needed and context["last_intent"]:
        if context["last_intent"] == "order_status" and not entities.get("order_id"):
            # If user provided order ID in follow-up
            order_id = entities.get("order_id")
            if order_id:
                response = get_order_status(order_id)
                context["last_intent"] = None # Clear intent after fulfilling
            else:
                # Still no order ID, keep asking
                clarification_needed = True
                response = "I still need your order ID to check the status."
        elif context["last_intent"] == "product_inquiry" and not entities.get("product_name"):
            product_name = entities.get("product_name")
            if product_name:
                response = get_product_details(product_name)
                context["last_intent"] = None
            else:
                clarification_needed = True
                response = "Please tell me the name of the product you're asking about."

    context["history"].append({"bot": response})
    SESSION_CONTEXT[session_id] = context

    return ChatResponse(session_id=session_id, response=response, clarification_needed=clarification_needed, suggested_actions=suggested_actions)

@app.get("/")
async def root():
    return {"message": "Smart Customer Support Chatbot API. Use /chat to interact."}

# To run this application:
# 1. Save the code as main.py
# 2. Install uvicorn: pip install uvicorn fastapi pydantic
# 3. Run from your terminal: uvicorn main:app --reload
# 4. Access it at http://127.0.0.1:8000 (or the assigned port)
#    You can interact with the /chat endpoint using a tool like Postman or curl, or a simple UI.