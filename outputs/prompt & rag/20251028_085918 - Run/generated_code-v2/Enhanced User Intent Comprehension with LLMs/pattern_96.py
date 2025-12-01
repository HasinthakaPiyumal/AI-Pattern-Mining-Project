from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import spacy
import requests
from loguru import logger
import os

# --- Configuration and Constants ---
# Mock API Endpoints for an E-commerce Platform
MOCK_ECOMMERCE_API = {
    "order_tracking": "https://api.ecommerce.com/orders/{order_id}",
    "product_info": "https://api.ecommerce.com/products/{product_name}",
    "initiate_return": "https://api.ecommerce.com/returns",
    "human_handover": "https://api.ecommerce.com/support/handover",
}

# Predefined Intents
INTENTS = [
    "track_order",
    "product_info_query",
    "initiate_return",
    "connect_to_human_agent",
    "general_greeting",
    "unknown_intent",
]

# --- NLU Module ---
class NLUModule:
    def __init__(self):
        # Initialize a pre-trained sentiment analysis model as a placeholder for intent classification
        # In a real scenario, this would be a fine-tuned sequence classification model for specific intents.
        logger.info("Loading NLU models...")
        try:
            self.intent_classifier = pipeline("sentiment-analysis") # Placeholder
            # For a real application, you would load a fine-tuned model:
            # self.intent_classifier = pipeline("text-classification", model="your-finetuned-intent-model")

            # Load a small English model for entity extraction
            try:
                self.entity_extractor = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Downloading...")
                spacy.cli.download("en_core_web_sm")
                self.entity_extractor = spacy.load("en_core_web_sm")
            logger.info("NLU models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading NLU models: {e}")
            raise

    def classify_intent(self, text: str) -> str:
        # Placeholder logic: Map sentiment to a dummy intent
        # In a real system, this would use a proper intent classification model.
        result = self.intent_classifier(text)[0]
        label = result['label']
        score = result['score']

        if "greeting" in text.lower() or "hello" in text.lower():
            return "general_greeting"
        elif "order" in text.lower() and ("track" in text.lower() or "status" in text.lower()):
            return "track_order"
        elif "product" in text.lower() and ("info" in text.lower() or "details" in text.lower()):
            return "product_info_query"
        elif "return" in text.lower():
            return "initiate_return"
        elif "human" in text.lower() or "agent" in text.lower() or "speak to someone" in text.lower():
            return "connect_to_human_agent"
        else:
            # For a real system, you'd use the model's actual prediction.
            # This is a very simplistic fallback.
            if score > 0.8: # Example threshold
                if label == 'POSITIVE':
                    return 'product_info_query' # Arbitrary mapping
                elif label == 'NEGATIVE':
                    return 'initiate_return' # Arbitrary mapping
            return "unknown_intent"

    def extract_entities(self, text: str) -> dict:
        doc = self.entity_extractor(text)
        entities = {}
        for ent in doc.ents:
            # Simplified entity extraction; real system would map types to domain entities
            if ent.label_ == "PRODUCT" or ent.label_ == "ORG": # Example entity types
                entities["product_name"] = ent.text
            elif ent.label_ == "CARDINAL" and "order" in text.lower(): # Simple order ID heuristic
                entities["order_id"] = ent.text
        return entities

# --- Action Execution Module ---
class EcommerceAPIClient:
    def __init__(self, api_endpoints: dict):
        self.api_endpoints = api_endpoints

    def track_order(self, order_id: str) -> dict:
        logger.info(f"Attempting to track order: {order_id}")
        url = self.api_endpoints["order_tracking"].format(order_id=order_id)
        try:
            # Mocking an API call
            # response = requests.get(url, timeout=5).json()
            response = {"status": "Shipped", "estimated_delivery": "2023-10-27", "items": ["Item A", "Item B"]}
            logger.info(f"Order tracking successful for {order_id}")
            return {"success": True, "data": response}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error tracking order {order_id}: {e}")
            return {"success": False, "error": "Could not track order at this time."}

    def get_product_info(self, product_name: str) -> dict:
        logger.info(f"Attempting to get product info for: {product_name}")
        url = self.api_endpoints["product_info"].format(product_name=product_name)
        try:
            # Mocking an API call
            # response = requests.get(url, timeout=5).json()
            response = {"name": product_name, "price": "$49.99", "description": "A great product!", "availability": "In Stock"}
            logger.info(f"Product info successful for {product_name}")
            return {"success": True, "data": response}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting product info for {product_name}: {e}")
            return {"success": False, "error": "Could not retrieve product information."}

    def initiate_return(self, order_id: str, reason: str) -> dict:
        logger.info(f"Attempting to initiate return for order {order_id} with reason: {reason}")
        url = self.api_endpoints["initiate_return"]
        payload = {"order_id": order_id, "reason": reason}
        try:
            # Mocking an API call
            # response = requests.post(url, json=payload, timeout=5).json()
            response = {"return_id": "RET12345", "status": "Pending Approval"}
            logger.info(f"Return initiation successful for order {order_id}")
            return {"success": True, "data": response}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error initiating return for order {order_id}: {e}")
            return {"success": False, "error": "Could not initiate return at this time."}

    def connect_to_human_agent(self) -> dict:
        logger.info("Connecting to human agent...")
        url = self.api_endpoints["human_handover"]
        try:
            # Mocking an API call
            # response = requests.post(url, timeout=5).json()
            response = {"handover_status": "initiated", "queue_position": 3}
            logger.info("Human agent handover initiated.")
            return {"success": True, "data": response}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to human agent: {e}")
            return {"success": False, "error": "Could not connect to a human agent."}

# --- Response Generation Module ---
def generate_response(intent: str, action_result: dict, entities: dict) -> str:
    if action_result.get("success"):
        data = action_result.get("data", {})
        if intent == "track_order":
            return (f"Your order {entities.get('order_id', 'N/A')} is currently "
                    f"{data.get('status', 'unknown')}. Estimated delivery: {data.get('estimated_delivery', 'N/A')}.")
        elif intent == "product_info_query":
            return (f"Product: {data.get('name', 'N/A')}, Price: {data.get('price', 'N/A')}, "
                    f"Description: {data.get('description', 'N/A')}, Availability: {data.get('availability', 'N/A')}.")
        elif intent == "initiate_return":
            return (f"Your return (ID: {data.get('return_id', 'N/A')}) has been initiated "
                    f"and is {data.get('status', 'pending')}.")
        elif intent == "connect_to_human_agent":
            return (f"Connecting you to a human agent. Your position in queue is "
                    f"{data.get('queue_position', 'N/A')}. Please wait.")
        elif intent == "general_greeting":
            return "Hello! How can I assist you with your e-commerce needs today?"
    elif action_result.get("error"): # Handle errors from action execution
        return f"I apologize, but there was an issue: {action_result['error']}"
    elif intent == "general_greeting":
        return "Hello! How can I assist you with your e-commerce needs today?"
    elif intent == "unknown_intent":
        return "I'm sorry, I didn't understand your request. Can you please rephrase it or ask for something else?"
    return "I'm sorry, I couldn't process that request."

# --- Dialogue Management Module ---
class DialogueManager:
    def __init__(self, nlu: NLUModule, api_client: EcommerceAPIClient):
        self.nlu = nlu
        self.api_client = api_client
        self.conversation_context = {}

    def process_message(self, user_message: str) -> str:
        logger.info(f"Processing user message: {user_message}")

        intent = self.nlu.classify_intent(user_message)
        entities = self.nlu.extract_entities(user_message)

        logger.info(f"Detected intent: {intent}, Extracted entities: {entities}")

        action_result = {"success": False, "error": "No action taken.", "data": {}}
        if intent == "track_order":
            order_id = entities.get("order_id")
            if order_id:
                action_result = self.api_client.track_order(order_id)
            else:
                action_result["error"] = "Please provide an order ID to track."
        elif intent == "product_info_query":
            product_name = entities.get("product_name")
            if product_name:
                action_result = self.api_client.get_product_info(product_name)
            else:
                action_result["error"] = "Please specify the product you are interested in."
        elif intent == "initiate_return":
            order_id = entities.get("order_id") # Assuming order_id is needed for return
            # In a real scenario, we might prompt for return reason if not found
            reason = "customer_dissatisfaction" # Placeholder
            if order_id:
                action_result = self.api_client.initiate_return(order_id, reason)
            else:
                action_result["error"] = "Please provide an order ID to initiate a return."
        elif intent == "connect_to_human_agent":
            action_result = self.api_client.connect_to_human_agent()
        elif intent == "general_greeting":
            action_result["success"] = True # No API action needed for greeting
        elif intent == "unknown_intent":
            # Specific error for unknown intent, handled by response generation
            pass
        else:
            action_result["error"] = f"Unhandled intent: {intent}"

        return generate_response(intent, action_result, entities)

# --- FastAPI Application ---
app = FastAPI(
    title="E-commerce Chatbot with Intent Understanding",
    description="An intelligent chatbot leveraging NLU to interpret user queries and perform e-commerce actions."
)

# Initialize modules globally or pass as dependencies (for simplicity, global here)
try:
    nlu_module = NLUModule()
    api_client = EcommerceAPIClient(MOCK_ECOMMERCE_API)
    dialogue_manager = DialogueManager(nlu_module, api_client)
except Exception as e:
    logger.critical(f"Failed to initialize chatbot components: {e}")
    # Exit or mark service as unhealthy if critical components fail to load
    # For this example, we'll allow the app to start but log the error.


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    intent: str = None
    entities: dict = None


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    
    # Re-run NLU to get intent and entities for the response object
    # This is a bit redundant if dialogue_manager already did it, but good for explicit output.
    intent = nlu_module.classify_intent(user_message)
    entities = nlu_module.extract_entities(user_message)

    bot_response = dialogue_manager.process_message(user_message)
    
    return ChatResponse(response=bot_response, intent=intent, entities=entities)


@app.get("/health")
async def health_check():
    return {"status": "ok", "nlu_loaded": nlu_module is not None, "api_client_ready": api_client is not None}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting E-commerce Chatbot FastAPI application...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
