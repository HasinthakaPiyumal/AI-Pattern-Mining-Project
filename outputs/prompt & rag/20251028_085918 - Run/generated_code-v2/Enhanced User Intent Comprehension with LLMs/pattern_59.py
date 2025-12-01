import streamlit as st
import re

# --- 1. Knowledge Base (KB) Module (Simplified) ---
KNOWLEDGE_BASE = {
    "shipping_policy": "Our standard shipping takes 3-5 business days. Expedited shipping options are available at checkout.",
    "return_policy": "You can return most items within 30 days of purchase for a full refund. Items must be in their original condition.",
    "payment_methods": "We accept Visa, Mastercard, American Express, PayPal, and Google Pay.",
    "product_warranty": "All electronics come with a 1-year manufacturer's warranty.",
    "contact_support": "You can reach our customer support team at support@example.com or call us at 1-800-123-4567.",
    "product_availability_iphone": "The iPhone 15 is currently in stock. Limited quantities available.",
    "product_availability_samsung": "The Samsung Galaxy S24 is available for pre-order, shipping starts next month."
}

# --- 2. External Tools / API Integration Module (Simplified Placeholders) ---
def track_order_api(order_number):
    if order_number and order_number.isdigit() and len(order_number) == 8:
        # Simulate API call
        status_map = {
            "12345678": "Your order #12345678 is out for delivery and expected today.",
            "87654321": "Your order #87654321 was delivered on 2023-10-26."
        }
        return status_map.get(order_number, f"Order # {order_number} not found. Please double-check the number.")
    return "Please provide a valid 8-digit order number to track your order."

def initiate_return_api(order_number, product_name=None):
    if order_number and order_number.isdigit() and len(order_number) == 8:
        if order_number == "12345678":
            return f"Return initiated for order # {order_number} (Product: {product_name or 'unspecified'}). You will receive an email with instructions shortly."
        else:
            return f"Unable to initiate return for order # {order_number}. Please contact support."
    return "Please provide a valid 8-digit order number to initiate a return."

def get_product_details_api(product_name):
    product_details = {
        "iphone 15": "The iPhone 15 features a A17 Bionic chip, 6.1-inch Super Retina XDR display, and advanced dual-camera system. Starting at $799.",
        "samsung galaxy s24": "The Samsung Galaxy S24 boasts a new AI-powered camera, Snapdragon 8 Gen 3 processor, and dynamic AMOLED 2X display. Starting at $899."
    }
    return product_details.get(product_name.lower(), f"Details for '{product_name}' are not currently available.")

# --- 3. Natural Language Understanding (NLU) Module (Simplified) ---
class NLU:
    def __init__(self):
        pass # In a real app, load models here

    def preprocess(self, text):
        return text.strip().lower()

    def classify_intent(self, text):
        text = self.preprocess(text)
        if "track order" in text or "where is my order" in text or "order status" in text:
            return "track_order"
        elif "return" in text or "send back" in text or "refund" in text:
            return "initiate_return"
        elif "product" in text and ("info" in text or "details" in text or "about" in text):
            return "product_inquiry"
        elif "shipping" in text or "delivery" in text or "shipment" in text:
            return "shipping_policy"
        elif "payment" in text or "pay with" in text or "card" in text:
            return "payment_methods"
        elif "warranty" in text:
            return "product_warranty"
        elif "contact" in text or "speak to agent" in text or "human" in text:
            return "speak_to_agent"
        elif "hello" in text or "hi" in text or "hey" in text:
            return "greet"
        elif "thank" in text or "great" in text or "awesome" in text:
            return "thank_you"
        elif "availability" in text or "stock" in text:
            return "product_availability"
        return "general_query"

    def extract_entities(self, text, intent):
        entities = {}
        text = self.preprocess(text)

        # Extract order number
        order_match = re.search(r'\b(\d{8})\b', text) # Assumes 8-digit order number
        if order_match:
            entities["order_number"] = order_match.group(1)

        # Extract product name (simplified)
        product_keywords = ["iphone 15", "samsung galaxy s24", "laptop", "headphone"]
        for keyword in product_keywords:
            if keyword in text:
                entities["product_name"] = keyword
                break
        
        return entities

    def detect_ambiguity(self, intent, entities):
        if intent == "track_order" and "order_number" not in entities:
            return "Please provide your 8-digit order number to track it."
        if intent == "initiate_return" and "order_number" not in entities:
            return "To initiate a return, please provide your 8-digit order number."
        if intent == "product_inquiry" and "product_name" not in entities:
            return "Which product are you interested in?"
        if intent == "product_availability" and "product_name" not in entities:
            return "For which product would you like to check availability?"
        return None

# --- 4. Dialogue Management & Orchestration Module ---
class DialogueManager:
    def __init__(self, nlu_module, kb_module, tool_integration_module):
        self.nlu = nlu_module
        self.kb = kb_module
        self.tools = tool_integration_module
        self.context = {}

    def generate_response(self, user_query):
        intent = self.nlu.classify_intent(user_query)
        entities = self.nlu.extract_entities(user_query, intent)
        
        # Update context
        self.context.update(entities)
        
        # Check for ambiguity and prompt for clarification
        ambiguity_response = self.nlu.detect_ambiguity(intent, self.context)
        if ambiguity_response:
            return ambiguity_response

        # Handle intents
        if intent == "track_order":
            order_number = self.context.get("order_number")
            response = self.tools.track_order_api(order_number)
            self.context = {} # Clear context after fulfilling
            return response
        
        elif intent == "initiate_return":
            order_number = self.context.get("order_number")
            product_name = self.context.get("product_name")
            response = self.tools.initiate_return_api(order_number, product_name)
            self.context = {} # Clear context after fulfilling
            return response

        elif intent == "product_inquiry":
            product_name = self.context.get("product_name")
            if product_name:
                response = self.tools.get_product_details_api(product_name)
                self.context = {} 
                return response
            return self.kb.get("general_query", "Please tell me more about what product you are interested in.")

        elif intent == "product_availability":
            product_name = self.context.get("product_name")
            if product_name:
                key = f"product_availability_{product_name.replace(' ', '_')}"
                response = self.kb.get(key, f"I'm sorry, I don't have availability information for {product_name} at the moment.")
                self.context = {} 
                return response
            return self.kb.get("general_query", "Which product are you looking for?")

        elif intent == "shipping_policy":
            return self.kb.get("shipping_policy", "I'm sorry, I don't have information about shipping policy.")
        
        elif intent == "return_policy":
            return self.kb.get("return_policy", "I'm sorry, I don't have information about return policy.")

        elif intent == "payment_methods":
            return self.kb.get("payment_methods", "I'm sorry, I don't have information about payment methods.")

        elif intent == "product_warranty":
            return self.kb.get("product_warranty", "I'm sorry, I don't have information about product warranty.")

        elif intent == "speak_to_agent":
            self.context = {} 
            return self.kb.get("contact_support", "Connecting you to a human agent now...")

        elif intent == "greet":
            return "Hello! How can I assist you today?"

        elif intent == "thank_you":
            return "You're welcome! Is there anything else I can help you with?"
            
        elif intent == "general_query":
            return "I'm sorry, I didn't quite understand that. Can you please rephrase or ask about something else?"

        return "I'm sorry, I encountered an unexpected issue. Please try again later."


# --- Streamlit UI (5. User Interface) ---
st.title("🛒 Smart E-commerce Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize NLU and Dialogue Manager
nlu_module = NLU()
dialogue_manager = DialogueManager(nlu_module, KNOWLEDGE_BASE, globals()) # globals() for simplified tool access

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = dialogue_manager.generate_response(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
