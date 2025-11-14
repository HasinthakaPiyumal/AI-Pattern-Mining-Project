import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# --- 1. Product Catalog (Backend Logic) ---
PRODUCT_CATALOG = [
    {"id": "P001", "name": "Smartwatch X", "description": "A sleek smartwatch with health tracking, notifications, and long battery life.", "category": "Electronics", "price": 199.99},
    {"id": "P002", "name": "Noise-Cancelling Headphones", "description": "Immersive audio experience with active noise cancellation and comfortable earcups.", "category": "Electronics", "price": 249.00},
    {"id": "P003", "name": "Ergonomic Office Chair", "description": "Designed for maximum comfort and support during long working hours. Adjustable features.", "category": "Home & Office", "price": 349.50},
    {"id": "P004", "name": "Organic Coffee Beans", "description": "Premium Arabica beans, medium roast, ethically sourced for a rich and smooth cup.", "category": "Food & Beverage", "price": 15.99},
    {"id": "P005", "name": "Portable Bluetooth Speaker", "description": "Compact and powerful speaker with rich bass and waterproof design, perfect for outdoors.", "category": "Electronics", "price": 79.95},
    {"id": "P006", "name": "Yoga Mat Pro", "description": "Extra thick, non-slip yoga mat for superior comfort and stability during all types of yoga.", "category": "Sports & Outdoors", "price": 45.00},
    {"id": "P007", "name": "Stainless Steel Water Bottle", "description": "Keeps drinks cold for 24 hours and hot for 12 hours. Leak-proof and durable.", "category": "Kitchenware", "price": 22.50},
    {"id": "P008", "name": "Fiction Novel: The Quantum Leap", "description": "A gripping science fiction story exploring parallel universes and time travel.", "category": "Books", "price": 12.99},
    {"id": "P009", "name": "Wireless Charging Pad", "description": "Fast and efficient wireless charger compatible with most Qi-enabled smartphones.", "category": "Electronics", "price": 35.00},
    {"id": "P010", "name": "Gourmet Chocolate Assortment", "description": "A selection of exquisite dark, milk, and white chocolates crafted by master chocolatiers.", "category": "Food & Beverage", "price": 29.99},
]

# --- 2. Initialize Models and Pre-compute Embeddings (Backend Logic) ---
@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_resource
def load_llm_pipeline():
    # Using a smaller, faster model for demonstration purposes
    # For better quality, consider 'gpt2', 'facebook/bart-large-cnn' (for summarization/generation) or larger models
    pipe = pipeline("text-generation", model="distilgpt2", device=0 if st.session_state.get('cuda_available', False) else -1)
    return pipe

embedding_model = load_embedding_model()
llm_pipeline = load_llm_pipeline()

# Pre-compute embeddings for all product descriptions
product_descriptions = [product["description"] for product in PRODUCT_CATALOG]
@st.cache_data
def get_product_embeddings(descriptions, model):
    return model.encode(descriptions, show_progress_bar=False)

product_embeddings = get_product_embeddings(product_descriptions, embedding_model)

# --- Streamlit App (Frontend) ---
st.title("🛍️ AI-Enhanced Conversational E-commerce Recommender")
st.markdown("Hello! I'm your personal shopping assistant. Tell me what you're looking for!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What are you looking for today?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # 1. Embed the user query
        user_query_embedding = embedding_model.encode([prompt])

        # 2. Perform similarity search
        similarities = cosine_similarity(user_query_embedding, product_embeddings)[0]
        top_n_indices = np.argsort(similarities)[::-1][:3] # Get top 3 recommendations
        recommended_products = [PRODUCT_CATALOG[i] for i in top_n_indices]

        # 3. Construct LLM prompt for response generation
        llm_prompt_template = f"""The user is looking for products based on their query: "{prompt}".
        Here are some highly relevant products from our catalog:
        {chr(10).join([f"- {p['name']} ({p['category']}): {p['description']} (Price: ${p['price']:.2f})" for p in recommended_products])}

        As a helpful and friendly e-commerce assistant, recommend these products to the user in a conversational tone. Explain why they are suitable based on their query. Ask a follow-up question to encourage further interaction. Start your response with "Based on what you're looking for, here are a few suggestions:".

        Assistant:"""

        # 4. Generate LLM response
        try:
            llm_response = llm_pipeline(
                llm_prompt_template,
                max_new_tokens=200,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                eos_token_id=llm_pipeline.tokenizer.eos_token_id,
            )[0]['generated_text']

            # Clean up LLM output - sometimes distilgpt2 can be a bit repetitive or include the prompt
            # We'll try to find the actual assistant response part
            response_start_marker = "Assistant:"
            if response_start_marker in llm_response:
                llm_response = llm_response.split(response_start_marker, 1)[1].strip()
            
            # Further cleanup to remove potential prompt remnants or cut-off sentences
            llm_response = llm_response.split('\n\n')[0].strip()
            if llm_response.endswith("\n"):
                llm_response = llm_response[:-1].strip()
            if llm_response.endswith("..."):
                llm_response = llm_response.rsplit('.', 1)[0] + '.' # Try to complete sentence if cut off

            st.markdown(llm_response)
            st.session_state.messages.append({"role": "assistant", "content": llm_response})
        except Exception as e:
            error_message = f"An error occurred while generating a response: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
