import streamlit as st
import re

# --- Mock NLU Module ---

def mock_intent_classifier(query):
    query_lower = query.lower()
    if "product" in query_lower or "item" in query_lower or "details" in query_lower:
        return "product_inquiry"
    elif "order" in query_lower or "status" in query_lower or "delivery" in query_lower:
        return "order_status"
    elif "return" in query_lower or "exchange" in query_lower or "refund" in query_lower:
        return "return_request"
    elif "troubleshoot" in query_lower or "fix" in query_lower or "problem" in query_lower:
        return "troubleshooting"
    else:
        return "general_query"

def mock_entity_recognizer(query, intent):
    entities = {}
    query_lower = query.lower()

    if intent == "product_inquiry":
        product_keywords = ["laptop", "smartphone", "headphone", "tablet", "camera"]
        for keyword in product_keywords:
            if keyword in query_lower:
                entities["product_name"] = keyword
                break
    elif intent == "order_status" or intent == "return_request":
        order_id_match = re.search(r"#?(\d{6,})", query)
        if order_id_match:
            entities["order_id"] = order_id_match.group(1)

    return entities

# --- Mock Tool/Action Execution Module ---

mock_product_database = {
    "laptop": {"price": "$1200", "features": "16GB RAM, 512GB SSD, Intel i7", "availability": "In Stock"},
    "smartphone": {"price": "$800", "features": "6.1-inch display, 128GB storage, Dual Camera", "availability": "Low Stock"},
    "headphone": {"price": "$150", "features": "Noise Cancelling, Bluetooth 5.0", "availability": "In Stock"},
}

mock_orders_db = {
    "123456": {"status": "Shipped", "item": "Laptop", "delivery_date": "2023-11-15"},
    "789012": {"status": "Processing", "item": "Smartphone"},
}

mock_troubleshooting_kb = {
    "device not turning on": "Please ensure your device is charged. Try holding the power button for 10 seconds.",
    "internet connection issues": "Check your Wi-Fi router. Restart your modem and try again.",
}

def get_product_info(product_name):
    return mock_product_database.get(product_name.lower())

def get_order_status(order_id):
    return mock_orders_db.get(order_id)

def initiate_return(order_id):
    if order_id in mock_orders_db:
        return f"Return initiated for order {order_id}. You will receive an email with instructions."
    return f"Could not find order {order_id} to initiate a return."

def get_troubleshooting_steps(issue):
    for key in mock_troubleshooting_kb:
        if issue.lower() in key:
            return mock_troubleshooting_kb[key]
    return "I'm sorry, I don't have specific troubleshooting steps for that issue. Can you provide more details?"

# --- Dialogue Management & Clarification Module ---

def manage_dialogue(intent, entities, conversation_history):
    response = ""
    action_needed = None

    if intent == "product_inquiry":
        if "product_name" not in entities:
            response = "Which product are you interested in (e.g., laptop, smartphone)?"
            action_needed = "clarify_product"
        else:
            product_info = get_product_info(entities["product_name"])
            if product_info:
                response = f"The {entities['product_name']} is priced at {product_info['price']} with features like {product_info['features']}. It is {product_info['availability']}."
            else:
                response = f"I couldn't find information for {entities['product_name']}. Please check the spelling or try another product."
    elif intent == "order_status":
        if "order_id" not in entities:
            response = "Could you please provide your order ID?"
            action_needed = "clarify_order_id"
        else:
            order_details = get_order_status(entities["order_id"])
            if order_details:
                response = f"Your order {entities['order_id']} for a {order_details['item']} is currently '{order_details['status']}'. Expected delivery: {order_details.get('delivery_date', 'N/A')}."
            else:
                response = f"I couldn't find any order with ID {entities['order_id']}. Please double-check your order ID."
    elif intent == "return_request":
        if "order_id" not in entities:
            response = "To initiate a return, I need your order ID. Can you provide it?"
            action_needed = "clarify_order_id"
        else:
            response = initiate_return(entities["order_id"])
    elif intent == "troubleshooting":
        issue = entities.get("issue", conversation_history[-1]["user"] if conversation_history and conversation_history[-1]["user"] else "")
        steps = get_troubleshooting_steps(issue)
        if "I'm sorry" in steps: # Check for the default 'not found' message
             response = "What specific issue are you experiencing so I can assist with troubleshooting?"
             action_needed = "clarify_troubleshooting_issue"
        else:
            response = steps
    elif intent == "general_query":
        response = "I'm here to help with product inquiries, order status, returns, and troubleshooting. How can I assist you today?"

    return response, action_needed

# --- Orchestration Layer ---

def process_user_query(user_query, conversation_history):
    intent = mock_intent_classifier(user_query)
    entities = mock_entity_recognizer(user_query, intent)

    # Check for ongoing clarification from previous turns
    last_bot_message = conversation_history[-1]["agent"] if conversation_history else ""
    if "Which product" in last_bot_message and intent == "general_query": # If bot asked for product and user just typed a product name
        entities["product_name"] = user_query
        intent = "product_inquiry"
    elif "order ID" in last_bot_message and intent == "general_query": # If bot asked for order ID and user typed a number
        order_id_match = re.search(r"#?(\d{6,})", user_query)
        if order_id_match:
            entities["order_id"] = order_id_match.group(1)
            if "return" in last_bot_message:
                intent = "return_request"
            else:
                intent = "order_status"
    elif "What specific issue" in last_bot_message and intent == "general_query":
        entities["issue"] = user_query
        intent = "troubleshooting"

    response, action_needed = manage_dialogue(intent, entities, conversation_history)
    return response

# --- Streamlit UI Layer ---
st.title("Smart Customer Support Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = process_user_query(prompt, st.session_state.messages)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
