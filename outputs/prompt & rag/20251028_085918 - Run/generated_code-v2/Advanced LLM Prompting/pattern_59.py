import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load Paraphrasing Model
@st.cache_resource
def load_paraphraser():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_paraphraser()

# Define Intents and Responses
intents_responses = {
    "password_reset": {
        "keywords": ["reset password", "change password", "forgot password", "password recovery"],
        "response": "To reset your password, please visit our website's login page and click on the 'Forgot Password' link. Follow the instructions sent to your registered email."
    },
    "account_balance": {
        "keywords": ["account balance", "check balance", "how much money", "current balance"],
        "response": "You can check your account balance by logging into your online banking portal or through our mobile app. Your current balance will be displayed on the dashboard."
    },
    "contact_support": {
        "keywords": ["contact support", "speak to a human", "customer service", "help desk"],
        "response": "You can reach our customer support team by calling 1-800-555-CHAT during business hours, or by sending an email to support@example.com. We're here to help!"
    },
    "product_info": {
        "keywords": ["product information", "details about", "tell me about"],
        "response": "Please specify which product you are interested in, and I can provide you with more details. You can also browse our product catalog on our website."
    }
}

def paraphrase_text(text, num_return_sequences=3, max_length=60):
    input_text = f"paraphrase: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        input_ids,
        num_return_sequences=num_return_sequences,
        max_length=max_length,
        early_stopping=True,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    
    paraphrased_sentences = []
    for i in range(num_return_sequences):
        decoded_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        if decoded_text.lower() != text.lower(): # Avoid returning the exact original as a paraphrase
            paraphrased_sentences.append(decoded_text)
    return paraphrased_sentences

def recognize_intent(query_list):
    for query in query_list:
        lower_query = query.lower()
        for intent, data in intents_responses.items():
            for keyword in data["keywords"]:
                if keyword in lower_query:
                    return intent, query # Return the recognized intent and the query that matched it
    return "unknown", None

# Streamlit Application Logic
st.title("AI-Powered Customer Support Chatbot (Prompt Paraphrasing Demo)")
st.write("Ask me a question about account balance, password reset, or contacting support.")

user_query = st.text_input("Your query:", key="user_input")

if st.button("Get Response"):
    if user_query:
        st.subheader("Original Query:")
        st.write(user_query)
        
        paraphrases = paraphrase_text(user_query)
        if paraphrases:
            st.subheader("Generated Paraphrases:")
            for i, p in enumerate(paraphrases):
                st.write(f"- {p}")
        else:
            st.write("No meaningful paraphrases generated.")

        all_queries_for_intent = [user_query] + paraphrases
        recognized_intent, matched_query = recognize_intent(all_queries_for_intent)

        st.subheader("Chatbot Response:")
        if recognized_intent != "unknown":
            st.success(f"Intent Recognized: '{recognized_intent}' (matched with: '{matched_query}')")
            st.info(intents_responses[recognized_intent]["response"])
        else:
            st.error("I'm sorry, I couldn't understand your query. Please try rephrasing or contact support.")
            st.info(intents_responses["contact_support"]["response"])
    else:
        st.warning("Please enter your query to get a response.")
