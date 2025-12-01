import streamlit as st

KNOWLEDGE_BASE = [
    {"content": "Our latest smartphone, the XYZ Pro, features a 6.7-inch OLED display, a triple-lens 108MP camera, and a 5000mAh battery. It is available in Black, Silver, and Midnight Blue."},
    {"content": "The shipping policy states that standard shipping takes 3-5 business days. Express shipping is available for an extra charge and takes 1-2 business days. Free shipping is offered on all orders over $100."},
    {"content": "Returns are accepted within 30 days of purchase, provided the item is in its original condition with all packaging. To initiate a return, please visit our returns portal on the website."},
    {"content": "Our customer support can be reached via email at support@ecom.com or by phone at 1-800-555-0123. Operating hours are Monday-Friday, 9 AM - 5 PM EST."},
    {"content": "The new 'SmartWatch Ultra' has GPS tracking, heart rate monitoring, and is water-resistant up to 50 meters. It connects seamlessly with both iOS and Android devices."},
    {"content": "Payments can be made using Visa, MasterCard, American Express, PayPal, and Apple Pay. We also offer a 'Buy Now, Pay Later' option through Affirm."},
    {"content": "To reset your password, click on the 'Forgot Password' link on the login page and follow the instructions sent to your registered email address."}
]

def optimize_retrieval_query(lm_prefix: str, retrieval_query_length: int) -> str:
    tokens = lm_prefix.split()
    optimized_tokens = tokens[-retrieval_query_length:] if len(tokens) > retrieval_query_length else tokens
    return " ".join(optimized_tokens)

def retrieve_documents(query: str, knowledge_base: list) -> list:
    relevant_docs = []
    for doc in knowledge_base:
        if query.lower() in doc["content"].lower():
            relevant_docs.append(doc["content"])
    return relevant_docs

def generate_response(user_query: str, chat_history: list, retrieved_documents: list) -> str:
    response_parts = ["Hello! How can I assist you today?"]

    if retrieved_documents:
        response_parts.append("Here's some information I found that might be relevant:")
        for doc in retrieved_documents:
            response_parts.append(f"- {doc}")
        response_parts.append("Is there anything specific you would like to know about this?")
    else:
        response_parts.append("I couldn't find very specific information on that, but I'll do my best to help.")

    if "shipping" in user_query.lower():
        response_parts.append("Regarding shipping, standard shipping takes 3-5 business days. You can find more details on our website.")
    elif "return" in user_query.lower() or "returns" in user_query.lower():
        response_parts.append("For returns, please visit our returns portal within 30 days of purchase.")
    elif "camera" in user_query.lower() or "smartphone" in user_query.lower():
        response_parts.append("Our XYZ Pro smartphone has a triple-lens 108MP camera.")
    elif "payment" in user_query.lower():
        response_parts.append("We accept various payment methods including Visa, MasterCard, PayPal.")
    elif "customer support" in user_query.lower() or "contact" in user_query.lower():
        response_parts.append("You can reach customer support via email at support@ecom.com or by phone at 1-800-555-0123.")
    elif "password reset" in user_query.lower() or "forgot password" in user_query.lower():
        response_parts.append("To reset your password, use the 'Forgot Password' link on the login page.")
    elif not retrieved_documents:
         response_parts.append("Could you please rephrase or provide more details? I'm here to help!")

    return "\n".join(response_parts)

st.set_page_config(page_title="E-commerce Chatbot (RALM Demo)")
st.title("🛒 E-commerce Customer Support Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

retrieval_query_length = st.slider(
    "Retrieval Query Length (Number of tokens from prefix)",
    min_value=1,
    max_value=50,
    value=10,
    step=1,
    key="retrieval_query_length_slider"
)

if prompt := st.chat_input("Ask me about products, shipping, returns, or anything else!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    lm_prefix_content = "\n".join([msg["content"] for msg in st.session_state.messages])
    
    optimized_query = optimize_retrieval_query(lm_prefix_content, retrieval_query_length)
    retrieved_docs = retrieve_documents(optimized_query, KNOWLEDGE_BASE)

    # Mock LLM generation
    response = generate_response(prompt, st.session_state.messages, retrieved_docs)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
