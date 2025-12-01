import gradio as gr
import re
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification # Uncomment for actual transformer model usage

# --- 1. Mock Intent Recognizer (simulating transformers usage) ---
class IntentRecognizer:
    def __init__(self):
        # In a real application, you would load a fine-tuned model here:
        # self.tokenizer = AutoTokenizer.from_pretrained("your-intent-model")
        # self.model = AutoModelForSequenceClassification.from_pretrained("your-intent-model")
        # self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        # For this example, we'll use a simple keyword-based approach to simulate classification
        self.intents = {
            "order_status": ["order status", "where is my order", "delivery time", "when will it arrive", "my order"],
            "track_shipment": ["track shipment", "shipping update", "package location", "my package"],
            "initiate_return": ["return item", "exchange product", "refund", "send back", "return a product"],
            "connect_to_agent": ["talk to human", "speak to representative", "customer service", "help me", "agent"],
            "product_inquiry": ["product details", "tell me about", "specifications", "about product"],
            "greeting": ["hello", "hi", "hey"]
        }

    def predict_intent(self, query: str) -> str:
        query_lower = query.lower()
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return intent
        return "unknown"

# --- 2. Mock E-commerce Tools ---
def check_order_status(order_id: str = None) -> str:
    if order_id and order_id != "":
        return f"Checking status for order {order_id}. It is currently out for delivery and expected tomorrow." # Mock data
    return "Please provide an order ID to check its status."

def track_shipment(tracking_number: str = None) -> str:
    if tracking_number and tracking_number != "":
        return f"Tracking shipment {tracking_number}. Your package is in transit, expected by Friday." # Mock data
    return "Please provide a tracking number."

def initiate_return(item_name: str = None) -> str:
    if item_name and item_name != "":
        return f"Initiating return process for '{item_name}'. Please check your email for return instructions within 24 hours." # Mock data
    return "Which item would you like to return?"

def connect_to_agent() -> str:
    return "Connecting you to a customer service representative. Please wait while we find an available agent."

def product_inquiry(product_query: str) -> str:
    if "smartphone" in product_query.lower():
        return "We have a wide range of smartphones. Are you looking for a specific brand, price range, or features?"
    elif "laptop" in product_query.lower():
        return "Our laptops come in various configurations. Do you have a particular use-case in mind, like gaming or work?"
    return f"I can help with product inquiries. What specific product or type of product are you interested in?"

# --- 3. Dialogue Manager ---
class DialogueManager:
    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.current_state = "initial"
        self.pending_intent = None
        self.pending_params = {}

    def reset_state(self):
        self.current_state = "initial"
        self.pending_intent = None
        self.pending_params = {}

    def respond(self, user_query: str) -> str:
        response = ""
        
        # First, check if we are in a pending state and the user's input can resolve it
        if self.current_state == "awaiting_order_id":
            self.pending_params["order_id"] = user_query.strip()
            response = check_order_status(self.pending_params["order_id"])
            self.reset_state()
            return response
        elif self.current_state == "awaiting_tracking_number":
            self.pending_params["tracking_number"] = user_query.strip()
            response = track_shipment(self.pending_params["tracking_number"])
            self.reset_state()
            return response
        elif self.current_state == "awaiting_return_item":
            self.pending_params["item_name"] = user_query.strip()
            response = initiate_return(self.pending_params["item_name"])
            self.reset_state()
            return response

        # If not in a pending state, predict intent
        intent = self.intent_recognizer.predict_intent(user_query)

        # Handle different intents
        if intent == "greeting":
            self.reset_state()
            response = "Hello! How can I assist you today regarding your orders or products?"
        elif intent == "order_status":
            match = re.search(r"order\s+(\w+)", user_query, re.IGNORECASE)
            order_id = match.group(1) if match else ""
            
            if order_id:
                response = check_order_status(order_id)
                self.reset_state()
            else:
                self.current_state = "awaiting_order_id"
                self.pending_intent = "order_status"
                response = "I can help with that. What is your order ID?"
        elif intent == "track_shipment":
            match = re.search(r"number\s+(\w+)", user_query, re.IGNORECASE)
            tracking_number = match.group(1) if match else ""

            if tracking_number:
                response = track_shipment(tracking_number)
                self.reset_state()
            else:
                self.current_state = "awaiting_tracking_number"
                self.pending_intent = "track_shipment"
                response = "I can help with that. What is your tracking number?"
        elif intent == "initiate_return":
            # For simplicity, always ask for the item name first for returns
            self.current_state = "awaiting_return_item"
            self.pending_intent = "initiate_return"
            response = "Sure, I can help you initiate a return. Which item would you like to return?"
        elif intent == "connect_to_agent":
            response = connect_to_agent()
            self.reset_state()
        elif intent == "product_inquiry":
            response = product_inquiry(user_query)
            self.reset_state()
        else: # "unknown" intent
            response = "I'm sorry, I didn't quite understand that. Could you please rephrase or tell me what you're trying to do? For example, 'Check my order status' or 'Track my package'."
            self.reset_state()

        return response

# --- 4. Gradio Interface ---
# Initialize the dialogue manager
dm = DialogueManager()

def chatbot_interface(user_message, history):
    # The dialogue manager manages its own state, so we just pass the user message
    response = dm.respond(user_message)
    # Gradio history is a list of lists: [[user_msg, bot_msg], ...]
    history.append([user_message, response])
    return "", history # Clear input box and return updated history

# Gradio ChatInterface
demo = gr.ChatInterface(
    chatbot_interface,
    title="E-commerce Customer Support Chatbot",
    description="I can help you with order status, tracking, returns, and product inquiries. Please try asking me something like 'Where is my order?' or 'I want to return a product.'",
    examples=[
        ["Where is my order ABC123DEF"],
        ["I want to return an item."],
        ["Track my package with number GHI456JKL."],
        ["Talk to a human."],
        ["Hello"],
        ["Tell me about a new smartphone."]
    ],
    clear_btn="Clear Chat",
    submit_btn="Send"
)

# To run the Gradio app, save this code as a .py file and execute it.
# Then open your web browser to the address provided by Gradio (usually http://127.0.0.1:7860)
# demo.launch() # Uncomment to run directly if this file is executed