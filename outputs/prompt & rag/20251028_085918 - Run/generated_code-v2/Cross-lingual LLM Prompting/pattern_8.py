import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline
import requests
import uvicorn
import threading
import time

DetectorFactory.seed = 0 # for reproducible results

# Initialize ChromaDB client
chroma_client = chromadb.Client()
collection_name = "crosslingual_customer_support"
try:
    collection = chroma_client.get_or_create_collection(name=collection_name)
except Exception as e:
    print(f"Error getting/creating collection: {e}. Attempting to delete and recreate.")
    chroma_client.delete_collection(name=collection_name)
    collection = chroma_client.get_or_create_collection(name=collection_name)

# Initialize Sentence Transformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Multilingual LLM (Flan-T5 for text2text generation)
# Note: For a real-world production system, consider a larger, more capable LLM and potentially hosting it on a dedicated inference service.
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=-1) # -1 for CPU, 0 for GPU if available

# --- Dummy Cross-lingual Examples for In-Context Learning ---
# These examples will be stored in ChromaDB for retrieval
dummy_examples = [
    {
        "id": "example_1",
        "query_en": "What is your return policy?",
        "response_en": "Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging.",
        "query_es": "¿Cuál es su política de devoluciones?",
        "response_es": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra, siempre que el artículo no haya sido utilizado y esté en su embalaje original."
    },
    {
        "id": "example_2",
        "query_en": "How can I track my order?",
        "response_en": "You can track your order by logging into your account and visiting the 'My Orders' section, or by using the tracking number provided in your shipping confirmation email.",
        "query_es": "¿Cómo puedo rastrear mi pedido?",
        "response_es": "Puede rastrear su pedido iniciando sesión en su cuenta y visitando la sección 'Mis pedidos', o utilizando el número de seguimiento proporcionado en su correo electrónico de confirmación de envío."
    },
    {
        "id": "example_3",
        "query_en": "Do you ship internationally?",
        "response_en": "Yes, we ship to over 100 countries worldwide. Shipping costs and delivery times vary by destination.",
        "query_es": "¿Realizan envíos internacionales?",
        "response_es": "Sí, realizamos envíos a más de 100 países en todo el mundo. Los costos de envío y los tiempos de entrega varían según el destino."
    },
    {
        "id": "example_4",
        "query_en": "What payment methods do you accept?",
        "response_en": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
        "query_fr": "Quels modes de paiement acceptez-vous ?",
        "response_fr": "Nous acceptons Visa, Mastercard, American Express, PayPal et Apple Pay."
    }
]

def initialize_chroma_db():
    ids = [ex["id"] for ex in dummy_examples]
    documents_to_embed = [
        f"English Query: {ex.get('query_en', '')} English Response: {ex.get('response_en', '')} "
        f"Spanish Query: {ex.get('query_es', '')} Spanish Response: {ex.get('response_es', '')} "
        f"French Query: {ex.get('query_fr', '')} French Response: {ex.get('response_fr', '')}"
        for ex in dummy_examples
    ]
    embeddings = embedder.encode(documents_to_embed).tolist()
    
    # Store the full example in metadata, keyed by language
    metadatas = []
    for ex in dummy_examples:
        meta = {}
        if "query_en" in ex: meta["query_en"] = ex["query_en"]
        if "response_en" in ex: meta["response_en"] = ex["response_en"]
        if "query_es" in ex: meta["query_es"] = ex["query_es"]
        if "response_es" in ex: meta["response_es"] = ex["response_es"]
        if "query_fr" in ex: meta["query_fr"] = ex["query_fr"]
        if "response_fr" in ex: meta["response_fr"] = ex["response_fr"]
        metadatas.append(meta)

    existing_ids = collection.peek(100)['ids'] # Check if collection is empty
    if not existing_ids:
        print("Populating ChromaDB with examples...")
        collection.add(embeddings=embeddings, documents=documents_to_embed, metadatas=metadatas, ids=ids)
        print("ChromaDB populated.")
    else:
        print("ChromaDB already contains examples.")

initialize_chroma_db()

# --- FastAPI Application ---
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

def construct_icl_prompt(user_query: str, detected_lang: str, retrieved_examples: list) -> str:
    prompt = "Given the following customer support query examples and their responses in multiple languages:\n\n"

    for i, ex in enumerate(retrieved_examples):
        metadata = ex['metadata']
        prompt += f"Example {i+1}:\n"
        
        # Always include English if available as a pivot for cross-lingual transfer
        if "query_en" in metadata:
            prompt += f"English Query: {metadata['query_en']}\n"
            if "response_en" in metadata: prompt += f"English Response: {metadata['response_en']}\n"
        
        # Include the detected language if available
        query_key = f"query_{detected_lang}"
        response_key = f"response_{detected_lang}"
        if query_key in metadata:
            prompt += f"{detected_lang.capitalize()} Query: {metadata[query_key]}\n"
            if response_key in metadata: prompt += f"{detected_lang.capitalize()} Response: {metadata[response_key]}\n"
        
        # If detected_lang is not English, and Spanish is available, include it too for more cross-lingual context
        if detected_lang != 'en' and "query_es" in metadata and detected_lang != 'es':
            prompt += f"Spanish Query: {metadata['query_es']}\n"
            if "response_es" in metadata: prompt += f"Spanish Response: {metadata['response_es']}\n"

        prompt += "\n"
    
    prompt += f"Now, respond to the following customer query in {detected_lang}:\n"
    prompt += f"{user_query}\n"
    prompt += f"Response in {detected_lang}:"
    return prompt

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    user_query = request.query
    
    try:
        detected_lang = detect(user_query)
    except Exception:
        detected_lang = "en" # Default to English if detection fails
    
    # Embed the user query
    query_embedding = embedder.encode([user_query]).tolist()
    
    # Retrieve top-k relevant examples from ChromaDB
    # Use a generic document for search, and rely on metadata for cross-lingual examples
    retrieved_results = collection.query(
        query_embeddings=query_embedding,
        n_results=2, # Retrieve top 2 examples
        include=['metadatas']
    )
    
    retrieved_examples = retrieved_results['metadatas'][0]
    
    # Construct the ICL prompt
    icl_prompt = construct_icl_prompt(user_query, detected_lang, retrieved_examples)
    
    # Get response from LLM
    try:
        llm_response = llm_pipeline(icl_prompt, max_new_tokens=100, num_return_sequences=1)
        generated_text = llm_response[0]['generated_text']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM inference failed: {e}")
    
    return {"response": generated_text}

# --- Streamlit Frontend ---
def streamlit_app():
    st.set_page_config(page_title="Multilingual E-commerce Chatbot", layout="centered")
    st.title("🌍 Multilingual E-commerce Chatbot")
    st.markdown("Ask a question in English, Spanish, or French and get a cross-lingual assisted response!")

    user_query = st.text_area("Your Question:", height=100)

    if st.button("Get Response"):        
        if user_query:
            st.info("Sending query to chatbot...")
            try:
                # Make a request to the FastAPI backend
                response = requests.post("http://localhost:8000/chat", json={"query": user_query})
                response.raise_for_status() # Raise an exception for HTTP errors
                chatbot_response = response.json()["response"]
                st.success("Chatbot Response:")
                st.write(chatbot_response)
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Please ensure the FastAPI server is running.")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question.")


# --- Running both FastAPI and Streamlit ---
# This section provides instructions on how to run both components.
# They cannot be run directly in a single `python script.py` command
# without complex process management. Instead, run them in separate terminals.

if __name__ == "__main__":
    st.markdown("## How to Run")
    st.markdown("To run this application, you need two separate terminals:")
    st.markdown("1.  **Run the FastAPI backend:**")
    st.code("uvicorn multilingual_chatbot:app --reload --port 8000")
    st.markdown("2.  **Run the Streamlit frontend:**")
    st.code("streamlit run multilingual_chatbot.py")
    st.markdown("Make sure you have all the required libraries installed (`pip install -r requirements.txt`).")
    st.markdown("**Note:** If you are seeing this text, you are running the script directly. For the Streamlit UI, use `streamlit run multilingual_chatbot.py`.")

    # To allow direct execution of the streamlit app when run via `streamlit run`
    # and avoid running FastAPI directly (which uvicorn handles)
    if "streamlit_script_runner" in __import__("sys").modules:
        streamlit_app()