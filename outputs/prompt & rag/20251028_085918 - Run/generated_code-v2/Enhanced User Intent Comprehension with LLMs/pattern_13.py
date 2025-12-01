import streamlit as st
from transformers import pipeline

# 1. NLU Module (Intent Classifier) - Using Zero-Shot Classification for demonstration
# In a real-world scenario, this would be a fine-tuned model for specific e-commerce intents.
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_intent(query):
    candidate_labels = [
        "order status inquiry",
        "product inquiry",
        "return request",
        "shipping information",
        "general support",
        "gift recommendation",
        "account management"
    ]
    result = classifier(query, candidate_labels)
    # The label with the highest score is considered the primary intent
    return result["labels"][0]

# 2. Knowledge Base (Simulated)
knowledge_base = {
    "order status inquiry": "To check your order status, please provide your order ID.",
    "return request": "You can initiate a return by visiting our returns page or providing your order ID and reason for return.",
    "product inquiry": "What specific product are you interested in? I can help you with features, availability, or recommendations.",
    "shipping information": "Shipping times vary based on your location and selected method. Standard shipping usually takes 3-5 business days.",
    "general support": "How can I assist you further? Please describe your issue.",
    "gift recommendation": "Tell me a bit about the recipient and their interests, and I can suggest some gift ideas!",
    "account management": "For account-related issues, please visit your account settings or contact our support team directly."
}

# 3. Simulated Backend Services
def simulate_check_order_status(order_id="N/A"):
    if order_id != "N/A":
        return f"Simulating: Checking status for order {order_id}. It appears to be \"In Transit\"."
    return "Please provide your order ID to check its status."

def simulate_process_return(order_id="N/A", reason="N/A"):
    if order_id != "N/A" and reason != "N/A":
        return f"Simulating: Processing return for order {order_id} due to: {reason}. You will receive an email shortly."
    return "Please provide your order ID and reason for return."

def simulate_product_recommendation(interests="N/A"):
    if interests != "N/A":
        return f"Simulating: Based on interests like {interests}, consider these products: [Product A, Product B, Product C]."
    return "What kind of products are you looking for or who is the gift for?"

# 4. Dialogue Management & Action Dispatcher
def get_chatbot_response(user_query):
    intent = get_intent(user_query)
    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("system", f"*Identified Intent: {intent}*"))

    response = "I'm not sure how to handle that. Can you please rephrase?"

    if intent == "order status inquiry":
        # Simple entity extraction simulation - for a real system, use regex or NER
        order_id_match = next((word for word in user_query.split() if word.isdigit() and len(word) > 5), "N/A")
        response = simulate_check_order_status(order_id=order_id_match)
    elif intent == "return request":
        response = knowledge_base.get(intent)
        # Further steps would involve prompting for order ID/reason if needed
    elif intent == "product inquiry":
        response = knowledge_base.get(intent)
        # Further steps would involve prompting for specifics
    elif intent == "shipping information":
        response = knowledge_base.get(intent)
    elif intent == "gift recommendation":
        # Simple entity extraction simulation
        keywords = ["friend", "mom", "dad", "kids", "birthday", "anniversary", "tech", "books", "fashion"]
        extracted_interests = [k for k in keywords if k in user_query.lower()]
        response = simulate_product_recommendation(interests= ", ".join(extracted_interests) if extracted_interests else "N/A")
    elif intent == "general support" or intent == "account management":
        response = knowledge_base.get(intent)
    else:
        response = "I understand you're looking for assistance. Could you please clarify your request?"

    return response

# Streamlit UI
st.set_page_config(page_title="E-commerce Chatbot", layout="centered")
st.title("🛒 E-commerce Customer Support Chatbot")
st.write("Hello! How can I assist you with your order, returns, or products today?")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)

# User input
user_query = st.chat_input("Type your query here...")

if user_query:
    chatbot_response = get_chatbot_response(user_query)
    st.session_state.chat_history.append(("system", chatbot_response))
    st.rerun()


