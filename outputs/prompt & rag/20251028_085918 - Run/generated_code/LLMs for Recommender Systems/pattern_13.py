# app.py

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import os

# Set OpenAI API key from environment variable or Streamlit secrets
# Ensure you have your OPENAI_API_KEY set in your environment variables
# or in a .streamlit/secrets.toml file if deploying to Streamlit Cloud.
# Example for .streamlit/secrets.toml:
# OPENAI_API_KEY="your_openai_api_key_here"

# For local testing, you can uncomment and set it directly (NOT recommended for production):
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- 1. Data Ingestion & Preprocessing ---
# Mock Product Catalog
products_data = [
    {"id": "P001", "name": "Smartwatch Pro", "description": "Advanced smartwatch with heart rate monitoring, GPS, and long battery life.", "category": "Electronics", "price": 299.99},
    {"id": "P002", "name": "Wireless Noise-Cancelling Headphones", "description": "Premium headphones with industry-leading noise cancellation and crystal-clear audio.", "category": "Electronics", "price": 249.00},
    {"id": "P003", "name": "Ergonomic Office Chair", "description": "Comfortable and supportive chair designed for long hours of work, adjustable lumbar support.", "category": "Home & Office", "price": 350.50},
    {"id": "P004", "name": "Portable Bluetooth Speaker", "description": "Compact speaker with powerful sound and 12-hour battery life, waterproof design.", "category": "Electronics", "price": 79.99},
    {"id": "P005", "name": "Organic Green Tea Sampler", "description": "A selection of finest organic green teas, perfect for relaxation and health benefits.", "category": "Food & Beverage", "price": 25.00},
    {"id": "P006", "name": "Laptop Backpack with USB Charging Port", "description": "Durable and stylish backpack with dedicated laptop compartment and integrated USB charging port.", "category": "Accessories", "price": 89.95},
    {"id": "P007", "name": "Professional DSLR Camera Kit", "description": "High-resolution DSLR camera with multiple lenses and accessories for aspiring photographers.", "category": "Electronics", "price": 1200.00},
    {"id": "P008", "name": "Yoga Mat Eco-Friendly", "description": "Non-slip, durable, and environmentally friendly yoga mat for all types of practice.", "category": "Sports & Outdoors", "price": 45.00},
    {"id": "P009", "name": "Robot Vacuum Cleaner", "description": "Smart robot vacuum with intelligent mapping and powerful suction for automated home cleaning.", "category": "Home Appliances", "price": 499.00},
    {"id": "P010", "name": "Adventure Travel Guidebook", "description": "Comprehensive guide to exploring the world's most thrilling destinations, with tips and itineraries.", "category": "Books", "price": 19.99},
]
products_df = pd.DataFrame(products_data)

# --- 2. Semantic Understanding & Embedding Module ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedding_model()

@st.cache_resource
def get_chroma_client():
    # Initialize ChromaDB client. Using a persistent client to store data locally.
    client = chromadb.PersistentClient(path="./chroma_db")
    return client

client = get_chroma_client()

def get_product_collection():
    try:
        collection = client.get_collection(name="products")
    except: # Collection does not exist, create it
        collection = client.create_collection(name="products")
        st.write("ChromaDB collection created. Populating with product data...")
        documents = products_df["description"].tolist()
        metadatas = products_df.drop(columns="description").to_dict(orient="records")
        ids = products_df["id"].tolist()

        # Generate embeddings in batches if necessary for large datasets
        embeddings = embedder.encode(documents, show_progress_bar=True).tolist()

        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        st.write("ChromaDB populated with product embeddings.")
    return collection

product_collection = get_product_collection()

# --- 3. Hybrid Recommender Core (Semantic Search) ---
def semantic_recommendations(query: str, top_k: int = 5):
    query_embedding = embedder.encode([query]).tolist()
    results = product_collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include = ["metadatas", "distances"]
    )
    recommended_product_ids = [m["id"] for m in results["metadatas"][0]]
    return products_df[products_df["id"].isin(recommended_product_ids)]

# --- 4. Explainable AI Module ---
@st.cache_resource
def load_explanation_llm_chain():
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
    explanation_template = """You are an AI assistant designed to provide concise and helpful explanations for product recommendations.
    Explain why a user might like the following product, considering their implied interest based on a query.

    User Query: {query}
    Product Name: {product_name}
    Product Description: {product_description}

    Provide a brief, natural language explanation (2-3 sentences) focusing on key benefits or features relevant to the query.
    Explanation:"""
    explanation_prompt = PromptTemplate(
        input_variables=["query", "product_name", "product_description"],
        template=explanation_template,
    )
    return LLMChain(llm=llm, prompt=explanation_prompt)

explanation_chain = load_explanation_llm_chain()

def get_explanation(query: str, product: pd.Series) -> str:
    return explanation_chain.run(
        query=query,
        product_name=product["name"],
        product_description=product["description"]
    )

# --- 5. Conversational AI Interface Module (Streamlit & Langchain) ---
st.title("🛒 Intelligent E-commerce Recommender")
st.markdown("Ask me about products, and I'll give you personalized recommendations with explanations!")

@st.cache_resource
def load_conversational_llm_chain():
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
    # Using ConversationBufferMemory to store chat history
    memory = ConversationBufferMemory(return_messages=True)

    # The prompt for the conversational agent
    template = """The following is a friendly conversation between a human and an AI assistant.
The AI is designed to help users find products and provide recommendations.
The AI is talkative and provides lots of specific details from its context.

Current conversation:
{history}
Human: {input}
AI:"""
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    return ConversationChain(llm=llm, memory=memory, prompt=prompt)

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = load_conversational_llm_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def handle_user_input(user_query):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate AI response using the conversational chain
    ai_response = st.session_state.conversation_chain.predict(input=user_query)

    # Check if the query seems like a product search/recommendation request
    # A more sophisticated intent recognition would be here. For simplicity, we assume if keywords like
    # 'recommend', 'looking for', 'find me', or product-related terms are present.
    lower_query = user_query.lower()
    product_keywords = ["recommend", "looking for", "find me", "product", "item", "buy", "want"]
    is_product_search = any(keyword in lower_query for keyword in product_keywords) or \
                        any(product_name.lower() in lower_query for product_name in products_df["name"])

    if is_product_search:
        with st.spinner("Finding recommendations..."):
            recommendations = semantic_recommendations(user_query, top_k=3)
            if not recommendations.empty:
                rec_message = "Here are some recommendations based on your query:\n\n"
                for idx, product in recommendations.iterrows():
                    explanation = get_explanation(user_query, product)
                    rec_message += f"**{product['name']}** ({product['category']}) - ${product['price']:.2f}\n"
                    rec_message += f"*Explanation:* {explanation}\n\n"
                ai_response = rec_message + "Is there anything else I can help you find?"
            else:
                ai_response = "I couldn't find any specific product recommendations for that. Can you try rephrasing?"

    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)

# Chat input
if prompt := st.chat_input("What are you looking for?"):
    handle_user_input(prompt)

