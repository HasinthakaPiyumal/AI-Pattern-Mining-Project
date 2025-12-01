import streamlit as st
import random
import time

# --- Simulated NLU Module (for demonstration) ---
class SimulatedNLU:
    def __init__(self):
        self.intents = {
            "order_status": ["where is my order", "track my package", "order status", "delivery status"],
            "product_inquiry": ["tell me about", "product details", "specs", "information on", "what is"],
            "return_request": ["return an item", "how to return", "exchange product", "refund"],
            "shipping_info": ["shipping cost", "delivery options", "shipping time"],
            "technical_support": ["troubleshoot", "not working", "technical issue", "help me with"],
            "billing_issue": ["bill problem", "charged wrong", "payment issue", "invoice"],
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
            "goodbye": ["bye", "goodbye", "see you", "farewell"]
        }
        self.responses = {
            "order_status": [
                "I can help with that. Could you please provide your order ID?",
                "To check your order status, please give me your order number."
            ],
            "product_inquiry": [
                "What product are you interested in? I can provide details about it.",
                "Please specify the product you're looking for information on."
            ],
            "return_request": [
                "To initiate a return, please visit our returns page or provide your order ID.",
                "I can guide you through the return process. Do you have your order number handy?"
            ],
            "shipping_info": [
                "Our standard shipping takes 3-5 business days. Would you like to know about expedited options?",
                "Shipping costs vary based on your location and the item. Can I get your delivery address or the item name?"
            ],
            "technical_support": [
                "I'm sorry you're experiencing technical issues. Can you describe the problem in more detail?",
                "Please describe the technical issue you're facing, and I'll connect you with an expert if needed."
            ],
            "billing_issue": [
                "For billing concerns, I'll need your account details or order number to investigate.",
                "Can you tell me more about the billing discrepancy you've noticed?"
            ],
            "greeting": [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?"
            ],
            "goodbye": [
                "Goodbye! Have a great day.",
                "It was a pleasure assisting you. See you next time!"
            ],
            "unknown": [
                "I'm not sure I understand. Could you rephrase your question?",
                "Could you please provide more details or ask your question in a different way?",
                "I apologize, I'm having trouble understanding. Are you looking for something specific?"
            ]
        }

    def predict_intent(self, query):
        query = query.lower()
        best_intent = "unknown"
        max_match = 0
        for intent, keywords in self.intents.items():
            match_count = sum(1 for keyword in keywords if keyword in query)
            if match_count > max_match:
                max_match = match_count
                best_intent = intent
        
        confidence = 0.8 if best_intent != "unknown" else 0.4
        # Simulate ambiguity for demonstration
        if "question about" in query and best_intent == "unknown":
             if random.random() > 0.5: # 50% chance to be ambiguous
                 return {"intent": "unknown", "confidence": 0.6, "clarify": True}
        return {"intent": best_intent, "confidence": confidence, "clarify": False}

    def extract_entities(self, query):
        entities = {}
        # Simplified entity extraction
        if "order id" in query:
            start = query.find("order id") + len("order id")
            # Simple regex-like extraction for numbers after "order id"
            import re
            match = re.search(r'\d+', query[start:])
            if match:
                entities["order_id"] = match.group(0)
        if "product name" in query or "item" in query:
            # Placeholder for actual product name extraction
            pass
        return entities

    def generate_clarifying_question(self, query):
        return random.choice([
            "I'm not entirely sure what you mean. Could you elaborate on your request?",
            "To help me understand better, could you please clarify your question?",
            "It seems I'm having trouble understanding. Are you asking about an order, a product, or something else?"
        ])

# --- Dialogue Manager ---
class DialogueManager:
    def __init__(self, nlu_module):
        self.nlu = nlu_module
        self.dialogue_history = []
        self.user_profiles = {
            "user123": {"name": "Alice", "past_orders": ["ORD1001", "ORD1005"]},
            "user456": {"name": "Bob", "past_orders": ["ORD2003"]}
        }

    def get_personalized_greeting(self, user_id=None):
        if user_id and user_id in self.user_profiles:
            user_name = self.user_profiles[user_id]["name"]
            return f"Welcome back, {user_name}! How can I help you today?"
        return "Hello! How can I assist you today?"

    def get_response(self, user_query, user_id=None):
        nlu_result = self.nlu.predict_intent(user_query)
        intent = nlu_result["intent"]
        confidence = nlu_result["confidence"]
        clarify = nlu_result["clarify"]
        entities = self.nlu.extract_entities(user_query)

        self.dialogue_history.append({"user": user_query, "intent": intent, "entities": entities})

        if clarify and confidence < 0.7: # Threshold for ambiguity
            return self.nlu.generate_clarifying_question(user_query)
        
        response = random.choice(self.nlu.responses.get(intent, self.nlu.responses["unknown"]))

        if user_id and user_id in self.user_profiles:
            user_profile = self.user_profiles[user_id]
            if intent == "order_status" and not entities.get("order_id") and user_profile["past_orders"]:
                response = f"I see you have past orders like {', '.join(user_profile['past_orders'])}. Are you asking about one of those, or a new order?"
            elif intent == "greeting":
                response = self.get_personalized_greeting(user_id)

        return response

# --- Streamlit UI ---
st.set_page_config(page_title="E-commerce Chatbot")
st.title("🛒 E-commerce Customer Support Chatbot")

# Initialize NLU and Dialogue Manager
if "nlu" not in st.session_state:
    st.session_state.nlu = SimulatedNLU()
if "dialogue_manager" not in st.session_state:
    st.session_state.dialogue_manager = DialogueManager(st.session_state.nlu)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": st.session_state.dialogue_manager.get_personalized_greeting("user123")}) # Simulate a logged-in user

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            time.sleep(1) # Simulate processing time
            response = st.session_state.dialogue_manager.get_response(prompt, user_id="user123") # Simulate user ID
            st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
