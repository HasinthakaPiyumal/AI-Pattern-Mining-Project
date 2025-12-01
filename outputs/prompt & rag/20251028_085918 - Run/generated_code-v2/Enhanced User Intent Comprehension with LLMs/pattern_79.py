import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import threading
import requests
from dotenv import load_dotenv
import os

# Simulate Hugging Face transformers and sentence-transformers
# In a real application, you would load actual models here.
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

load_dotenv()

# --- Simulated Medical Knowledge Base ---
MEDICAL_DOCUMENTS = [
    "Symptoms of influenza include fever, cough, sore throat, and body aches. It is caused by influenza viruses.",
    "Diabetes mellitus is a chronic metabolic disease characterized by high blood glucose levels. Type 1 diabetes is an autoimmune disease, while type 2 diabetes is often associated with insulin resistance.",
    "Hypertension, or high blood pressure, is a common condition that can lead to serious health problems like heart disease and stroke. Lifestyle changes and medication can help manage it.",
    "Common treatments for bacterial pneumonia include antibiotics such as amoxicillin or azithromycin. Rest and hydration are also important.",
    "A heart attack occurs when blood flow to a part of the heart is blocked, usually by a blood clot. Symptoms include chest pain, shortness of breath, and pain in the left arm.",
    "Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to symptoms like wheezing, shortness of breath, chest tightness, and coughing.",
    "The COVID-19 virus, SARS-CoV-2, can cause a range of symptoms from mild to severe, including fever, cough, fatigue, and loss of taste or smell. Vaccination is recommended to prevent severe illness."
]

# --- ChromaDB Initialization ---
# Using an in-memory client for simplicity. For persistence, configure a path.
client = chromadb.Client()
collection_name = os.getenv("CHROMA_COLLECTION_NAME", "medical_knowledge")

# Delete existing collection if it exists to ensure fresh data for demonstration
try:
    client.delete_collection(name=collection_name)
except:
    pass # Collection might not exist

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a new collection and add documents
vectorstore = Chroma.from_texts(
    texts=MEDICAL_DOCUMENTS,
    embedding=embeddings_model,
    collection_name=collection_name,
    client=client
)

# --- LLM for Intent Understanding and Response Generation (Simulated/Placeholder) ---
# In a real scenario, this would be a fine-tuned medical LLM.
# For demonstration, we'll use a simple text generation pipeline.
# You might need to install 'accelerate' for some models
try:
    llm_pipeline = pipeline("text-generation", model="distilgpt2", max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
except Exception as e:
    st.warning(f"Could not load 'distilgpt2' for text generation. \nError: {e}\nFalling back to a simpler placeholder for LLM response.")
    class SimplePlaceholderLLM:
        def __call__(self, prompt, stop=None, **kwargs):
            if "medical documents" in prompt.lower():
                return f"Based on medical knowledge, I can provide information. Your query was: '{prompt[:100]}...'"
            return f"I understand your query. Here is a simulated response: {prompt[:100]}..."
    llm = SimplePlaceholderLLM()

# --- Langchain RAG Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# --- FastAPI Backend ---
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        response = qa_chain.run(request.query)
        return {"response": response}
    except Exception as e:
        return {"error": str(e), "response": "An error occurred while processing your request."}

# --- Streamlit Frontend ---
def streamlit_app():
    st.set_page_config(page_title="Medical AI Assistant")
    st.title("🩺 Medical AI Assistant")
    st.write("Enter a medical query and get information from the AI assistant.")

    user_query = st.text_area("Your Medical Query:", "What are the symptoms of diabetes?")

    if st.button("Get Information"):
        if user_query:
            st.info("Processing your query...")
            try:
                # Make request to FastAPI backend
                backend_url = os.getenv("FASTAPI_URL", "http://localhost:8000")
                response = requests.post(f"{backend_url}/query", json={"query": user_query})
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                result = response.json()

                if "response" in result:
                    st.success("AI Assistant Response:")
                    st.write(result["response"])
                elif "error" in result:
                    st.error(f"Error from backend: {result['error']}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Make sure it's running.")
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Please enter a query.")

# --- Main entry point to run both FastAPI and Streamlit ---
def run_fastapi():
    # Use a specific port for FastAPI, e.g., 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    # Give FastAPI a moment to start up
    import time
    time.sleep(2)

    # Run Streamlit app
    streamlit_app()