import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from pydantic import BaseModel
from typing import List, Dict, Callable

# --- 1. Define Actions/Tools ---
class ECommerceActions:
    def track_order(self, order_id: str):
        return f"Looking up order {order_id}. Please wait for details."

    def initiate_return(self, item_name: str):
        return f"Initiating return process for '{item_name}'. You will receive an email shortly with instructions."

    def update_shipping_address(self, new_address: str):
        return f"Request to update shipping address to '{new_address}' has been received. Please verify the changes in your account."

    def product_inquiry(self, product_name: str):
        return f"Fetching details for product '{product_name}'. What specifically would you like to know?"

    def cancel_order(self, order_id: str):
        return f"Attempting to cancel order {order_id}. Please confirm this action."

    def payment_issue(self, issue_description: str):
        return f"Our support team has been notified about your payment issue: '{issue_description}'. We will contact you soon."

    def unknown_intent(self):
        return "I'm sorry, I don't understand your request. Can you please rephrase it or provide more details?"


# --- 2. Chatbot Core Logic ---
class Chatbot:
    def __init__(self):
        # Load Sentence Transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ChromaDB in-memory
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="ecommerce_intents")

        # Define predefined intents and their corresponding actions
        self.intents = {
            "track_order": {"phrases": ["where is my order", "track my package", "order status", "shipping update"], "action": ECommerceActions().track_order},
            "initiate_return": {"phrases": ["return an item", "how to return", "send back a product", "refund process"], "action": ECommerceActions().initiate_return},
            "update_shipping_address": {"phrases": ["change my address", "update delivery address", "new shipping location"], "action": ECommerceActions().update_shipping_address},
            "product_inquiry": {"phrases": ["tell me about this product", "product details", "specifications for", "features of"], "action": ECommerceActions().product_inquiry},
            "cancel_order": {"phrases": ["cancel my order", "stop this order", "undo purchase"], "action": ECommerceActions().cancel_order},
            "payment_issue": {"phrases": ["payment failed", "billing problem", "card declined", "transaction error"], "action": ECommerceActions().payment_issue}
        }

        self.intent_labels = list(self.intents.keys())
        self.intent_phrases = [phrase for intent_data in self.intents.values() for phrase in intent_data["phrases"]]
        self.intent_ids = [f"{label}_{i}" for label, intent_data in self.intents.items() for i in range(len(intent_data["phrases"]))]

        # Add intent phrases to ChromaDB
        if self.collection.count() == 0:
            self.collection.add(
                embeddings=self.embedding_model.encode(self.intent_phrases).tolist(),
                documents=self.intent_phrases,
                metadatas=[{"intent_name": label} for label in self.intent_labels for _ in range(len(self.intents[label]["phrases"]))],
                ids=self.intent_ids
            )
        
        # Basic pipeline for sentiment/entity (optional, for demo clarity in advanced scenarios)
        # self.nlp_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        # self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

    def recognize_intent(self, query: str, threshold: float = 0.65) -> (str, float):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=1,
            include=['distances', 'metadatas']
        )

        if results['distances'] and results['distances'][0] and results['distances'][0][0] < (1 - threshold):
            # ChromaDB returns L2 distance, so closer to 0 is better. (1-threshold) maps similarity to distance.
            closest_intent_name = results['metadatas'][0][0]['intent_name']
            # For simplicity, we'll use the similarity implied by the distance. Lower distance -> higher similarity.
            # A crude similarity estimate from L2 distance for this demo: max(0, 1 - distance/max_possible_distance)
            # Max possible L2 distance between normalized embeddings is 2.
            similarity_score = 1 - (results['distances'][0][0] / 2)
            return closest_intent_name, similarity_score
        return "unknown_intent", 0.0

    def extract_entities(self, query: str, intent: str) -> Dict[str, str]:
        # This is a very basic entity extraction. In a real system, you'd use NER models
        # or more sophisticated regex/pattern matching based on the recognized intent.
        entities = {}
        query_lower = query.lower()

        if intent == "track_order" or intent == "cancel_order":
            # Look for common order ID patterns (e.g., #12345, ORD-67890)
            import re
            match = re.search(r'(?:order|id|#|ord-)\s*([a-zA-Z0-9-]+)', query_lower)
            if match: entities["order_id"] = match.group(1).upper()
            else: entities["order_id"] = None # Indicate missing entity
        elif intent == "initiate_return":
            # For demo, just take some keywords as item_name
            if "phone" in query_lower: entities["item_name"] = "phone"
            elif "laptop" in query_lower: entities["item_name"] = "laptop"
            elif "shirt" in query_lower: entities["item_name"] = "shirt"
            else: entities["item_name"] = None
        elif intent == "update_shipping_address":
            # A very simplistic placeholder. Real address parsing is complex.
            if "to " in query_lower:
                parts = query_lower.split("to ", 1)
                if len(parts) > 1: entities["new_address"] = parts[1].strip().title()
            else: entities["new_address"] = None
        elif intent == "product_inquiry":
            if "about the " in query_lower: entities["product_name"] = query_lower.split("about the ", 1)[1].strip().title()
            elif "for the " in query_lower: entities["product_name"] = query_lower.split("for the ", 1)[1].strip().title()
            else: entities["product_name"] = None
        elif intent == "payment_issue":
            if "my card was declined" in query_lower: entities["issue_description"] = "card declined"
            elif "transaction failed" in query_lower: entities["issue_description"] = "transaction failed"
            else: entities["issue_description"] = query
            
        return entities

    def process_query(self, query: str) -> str:
        intent, confidence = self.recognize_intent(query)
        response = ""
        action_function = self.intents.get(intent, {}).get("action", ECommerceActions().unknown_intent)
        
        if confidence < 0.75 and intent != "unknown_intent": # If confidence is moderate, ask for clarification
            response = f"I'm not entirely sure I understood. Did you mean to {intent.replace('_', ' ')}? Can you elaborate?"
            # Store state for follow-up (simplified for this demo)
            st.session_state.awaiting_clarification = True
            st.session_state.potential_intent = intent
            return response
        
        if intent == "unknown_intent" or confidence < 0.65: # Low confidence or truly unknown
            return action_function() # This calls ECommerceActions().unknown_intent()
        
        # High confidence intent, proceed with action
        entities = self.extract_entities(query, intent)
        
        # Check for missing crucial entities and ask for them
        if intent == "track_order" and not entities.get("order_id"):
            return "Please provide your order ID so I can track your package."
        if intent == "cancel_order" and not entities.get("order_id"):
            return "To cancel your order, I need the order ID. Can you provide it?"
        if intent == "initiate_return" and not entities.get("item_name"):
            return "Which item would you like to return?"
        if intent == "update_shipping_address" and not entities.get("new_address"):
            return "What is the new shipping address you'd like to use?"
        if intent == "product_inquiry" and not entities.get("product_name"):
            return "Which product are you interested in?"

        try:
            # Call the action with extracted entities. Using **entities for keyword arguments.
            response = action_function(**{k: v for k, v in entities.items() if v is not None})
        except TypeError:
            # Handle cases where entities might not perfectly match action arguments
            response = f"I recognized your intent to {intent.replace('_', ' ')}, but I'm having trouble with the details. Can you provide more specific information?"
        
        st.session_state.awaiting_clarification = False # Reset clarification state
        st.session_state.potential_intent = None

        return response


# --- 3. Streamlit Frontend ---
st.set_page_config(page_title="E-commerce Chatbot", layout="centered")
st.title("🛒 E-commerce Customer Support Chatbot")

# Initialize chatbot and session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = Chatbot()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'awaiting_clarification' not in st.session_state:
    st.session_state.awaiting_clarification = False
if 'potential_intent' not in st.session_state:
    st.session_state.potential_intent = None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = st.session_state.chatbot.process_query(prompt)
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

