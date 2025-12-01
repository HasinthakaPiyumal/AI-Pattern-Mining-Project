import streamlit as st
from transformers import pipeline

# --- 1. Intent Understanding Module (Simulated) ---
# In a real scenario, you would load a fine-tuned model here.
# For this prototype, we'll use keyword-based intent detection
# to simulate the output of a more complex model.

# Placeholder for a Hugging Face pipeline (not actually used for keyword matching)
# If you had a fine-tuned model, it would look something like this:
# classifier = pipeline("text-classification", model="your_finetuned_model", tokenizer="your_finetuned_tokenizer")

# Define possible intents and associated keywords
intent_keywords = {
    "check_order_status": ["order", "status", "where is", "my package", "tracking", "delivery"],
    "product_inquiry": ["product", "info", "details", "about", "specifications", "features", "item"],
    "initiate_return": ["return", "send back", "faulty", "wrong item", "damaged"],
    "account_help": ["account", "login", "password", "profile", "billing", "address"],
    "greeting": ["hello", "hi", "hey"],
    "farewell": ["bye", "goodbye", "see you"],
    "thanks": ["thank you", "thanks"]
}

def predict_intent(query):
    query_lower = query.lower()
    detected_intents = []
    for intent, keywords in intent_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                detected_intents.append(intent)
                break # Move to next intent once a keyword is found

    # Simulate confidence: higher confidence if more specific keywords are hit or a direct match
    if len(detected_intents) == 1:
        return detected_intents[0], 0.9 # High confidence
    elif len(detected_intents) > 1:
        return detected_intents, 0.5 # Ambiguous, medium confidence
    else:
        return "general_inquiry", 0.6 # Default, medium confidence

# --- 2. Simulated Backend APIs ---
def get_order_details(order_id="#12345"):
    return f"Your order {order_id} is currently in transit and expected to be delivered by October 26, 2023."

def search_products(query):
    if "laptop" in query.lower():
        return "We have several great laptops! Are you looking for gaming, work, or something else?"
    return f"I can provide information on a wide range of products. What specific product are you interested in?"

def initiate_return_process(item_id="a product"):
    return f"To initiate a return for {item_id}, please visit our returns page at example.com/returns. You will need your order number."

def get_account_help():
    return "For account-related issues, please visit your account settings or contact our support team directly at support@example.com."

# --- 3. Action Executor ---
def execute_action(intent, query=None):
    if intent == "check_order_status":
        # In a real app, extract order_id from query
        return get_order_details()
    elif intent == "product_inquiry":
        return search_products(query)
    elif intent == "initiate_return":
        # In a real app, extract item_id from query
        return initiate_return_process()
    elif intent == "account_help":
        return get_account_help()
    elif intent == "greeting":
        return "Hello! How can I assist you today?"
    elif intent == "farewell":
        return "Goodbye! Have a great day!"
    elif intent == "thanks":
        return "You're welcome!"
    else:
        return "I'm sorry, I couldn't fully understand your request. Can you please rephrase or provide more details?"

# --- 4. Dialogue Manager ---
def manage_dialogue(user_query, chat_history):
    intent, confidence = predict_intent(user_query)

    if isinstance(intent, list): # Ambiguous intent
        st.session_state.last_ambiguous_intents = intent
        return "It seems you might be asking about a few things. Are you looking for information about a specific product, or perhaps checking an order status?"
    elif confidence < 0.7 and intent == "general_inquiry": # Low confidence general inquiry
        return "I'm not quite sure I understood. Could you please provide more context or be more specific?"
    elif st.session_state.get("last_ambiguous_intents"): # Follow-up after ambiguity
        clarified_intent = None
        if "product" in user_query.lower() and "product_inquiry" in st.session_state.last_ambiguous_intents:
            clarified_intent = "product_inquiry"
        elif "order" in user_query.lower() and "check_order_status" in st.session_state.last_ambiguous_intents:
            clarified_intent = "check_order_status"
        
        st.session_state.last_ambiguous_intents = None # Clear ambiguity state
        if clarified_intent:
            return execute_action(clarified_intent, user_query)
        else:
            return "I'm still having trouble understanding. Could you tell me more clearly what you need help with?"
    else:
        return execute_action(intent, user_query)

# --- Streamlit UI ---
st.set_page_config(page_title="E-commerce Chatbot")
st.title("🛒 E-commerce Customer Support Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I help you today?"})

if "last_ambiguous_intents" not in st.session_state:
    st.session_state.last_ambiguous_intents = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        response = manage_dialogue(prompt, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
