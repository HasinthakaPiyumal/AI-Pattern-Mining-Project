import streamlit as st
from transformers import pipeline

# --- NLU Module (Simulated) ---
# In a real scenario, this would be a fine-tuned model on e-commerce customer support data.
# We're using a generic text classification model for demonstration.
# You would replace this with your actual fine-tuned model and labels.

# A placeholder for a pre-trained model for intent recognition
# For a real e-commerce scenario, you'd fine-tune a model on your specific intents
# For simplicity, we'll simulate intent mapping.
# Example: classifier = pipeline("text-classification", model="your/finetuned-model")

# For demonstration, we'll use a rule-based simulation of intent recognition
def recognize_intent_simulated(text):
    text_lower = text.lower()
    if "order status" in text_lower or "where is my order" in text_lower:
        return {"intent": "order_status", "confidence": 0.95}
    elif "return" in text_lower or "refund" in text_lower or "send back" in text_lower:
        return {"intent": "return_refund", "confidence": 0.90}
    elif "product information" in text_lower or "details about" in text_lower or "specifications" in text_lower:
        return {"intent": "product_inquiry", "confidence": 0.88}
    elif "payment issue" in text_lower or "billing problem" in text_lower or "charged" in text_lower:
        return {"intent": "payment_issue", "confidence": 0.85}
    elif "account settings" in text_lower or "change password" in text_lower or "update profile" in text_lower:
        return {"intent": "account_management", "confidence": 0.80}
    elif "contact human" in text_lower or "speak to someone" in text_lower or "agent" in text_lower:
        return {"intent": "escalate_to_agent", "confidence": 0.99}
    else:
        return {"intent": "unknown", "confidence": 0.40} # Low confidence for unknown intents


# --- Chatbot Logic ---
def get_chatbot_response(user_input):
    intent_data = recognize_intent_simulated(user_input)
    intent = intent_data["intent"]
    confidence = intent_data["confidence"]

    response = ""

    # Personalization: Track recent intents for context (simple example)
    if "past_intents" not in st.session_state:
        st.session_state.past_intents = []
    st.session_state.past_intents.append(intent)
    if len(st.session_state.past_intents) > 3:
        st.session_state.past_intents.pop(0) # Keep only last 3 intents

    if confidence < 0.6:
        # Clarifying questions for low confidence or unknown intent
        response = "I'm not entirely sure what you mean. Could you please rephrase or provide more details? Are you looking for information about an order, a product, or something else?"
    elif intent == "order_status":
        response = "To check your order status, please provide your order number. \n(Simulating routing to 'Order Tracking' tool)"
    elif intent == "return_refund":
        response = "For returns or refunds, please visit our 'Returns Policy' page or provide your order number to initiate a return. \n(Simulating routing to 'Returns Management' tool)"
    elif intent == "product_inquiry":
        response = "I can help you with product information. Which product are you interested in? \n(Simulating routing to 'Product Catalog Search' tool)"
    elif intent == "payment_issue":
        response = "I understand you're having a payment issue. Please describe it in more detail, and I can try to help or connect you with our billing department. \n(Simulating routing to 'Payment Support' tool)"
    elif intent == "account_management":
        response = "You want to manage your account. What specifically would you like to do, such as changing your password or updating your profile? \n(Simulating routing to 'Account Settings' tool)"
    elif intent == "escalate_to_agent":
        response = "Connecting you to a human agent now. Please wait while I find an available representative. \n(Simulating agent hand-off)"
    else: # Default for sufficiently confident but unhandled specific intents or future intents
        response = f"I understood your intent as '{intent.replace('_', ' ')}'. How can I further assist you with this?"

    return response


# --- Streamlit UI ---
st.set_page_config(page_title="E-commerce Chatbot", layout="centered")
st.title("🛒 E-commerce Customer Support Chatbot")
st.write("Hello! I'm here to help you with your e-commerce queries.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_chatbot_response(prompt)
            st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})



# Optional: Display past intents for debugging/demonstration of personalization
# st.sidebar.subheader("Chatbot State (Simulated Personalization)")
# if "past_intents" in st.session_state:
#     st.sidebar.write("Recent Intents:", st.session_state.past_intents)
