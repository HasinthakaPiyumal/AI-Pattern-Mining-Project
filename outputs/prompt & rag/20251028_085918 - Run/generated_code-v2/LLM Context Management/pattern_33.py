from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from chromadb import Client, Settings
import os

# --- Configuration --- #
CLASSIFIER_MODEL_NAME = "distilbert-base-uncased-finetuned-sst2-english" # Using a sentiment model as a placeholder for complexity classifier
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "distilgpt2"

# --- FastAPI App Initialization --- #
app = FastAPI(title="Adaptive Customer Support AI Assistant")

# --- 1. Query Complexity Classifier --- #
# This will be a pre-trained model for demonstration. 
# In a real scenario, this would be fine-tuned on query complexity labels.
classifier_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME)
classifier_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL_NAME)
query_classifier = pipeline(
    "text-classification", 
    model=classifier_model,
    tokenizer=classifier_tokenizer,
    return_all_scores=True
)

# --- 3. Embedding Model --- #
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --- 2. Knowledge Bases (ChromaDB for Vector Stores) --- #
# In-memory ChromaDB for demonstration
chroma_client = Client(Settings(persist_directory="./chroma_db"))

# FAQ Knowledge Base (simplified as a dict for direct lookup)
faq_db = {
    "how do i change my email": "You can change your email in your profile settings under the 'Account' tab.",
    "reset password": "To reset your password, click on 'Forgot Password' on the login page and follow the instructions.",
    "subscription cost": "Our basic subscription costs $9.99/month. Please visit our pricing page for more details."
}

# CRM Data Knowledge Base (using ChromaDB for semantic search)
crm_collection_name = "crm_data_kb"
try:
    crm_kb = chroma_client.get_collection(name=crm_collection_name)
except:
    crm_kb = chroma_client.create_collection(name=crm_collection_name)
    crm_kb.add(
        documents=[
            "Customer ID 123: Ticket #XYZ status is 'In Progress', assigned to John Doe.",
            "Customer ID 456: Subscription expires on 2024-12-31.",
            "Customer ID 123: Has a premium support plan."
        ],
        metadatas=[
            {"source": "CRM_ticket"}, 
            {"source": "CRM_subscription"}, 
            {"source": "CRM_plan"}
        ],
        ids=["crm_doc1", "crm_doc2", "crm_doc3"]
    )

# Technical Documentation Knowledge Base (using ChromaDB)
tech_docs_collection_name = "tech_docs_kb"
try:
    tech_docs_kb = chroma_client.get_collection(name=tech_docs_collection_name)
except:
    tech_docs_kb = chroma_client.create_collection(name=tech_docs_collection_name)
    tech_docs_kb.add(
        documents=[
            "API Error 500: Internal Server Error. Check server logs for detailed tracebacks. Common causes include misconfigured environment variables or database connection issues.",
            "API Error 401: Unauthorized. Ensure your API key is valid and has the necessary permissions. Regenerate if necessary.",
            "Integration Guide: Step-by-step instructions for integrating with our payment gateway. Requires API key and webhook configuration."
        ],
        metadatas=[
            {"source": "API_Errors"}, 
            {"source": "API_Errors"}, 
            {"source": "Integration_Guide"}
        ],
        ids=["tech_doc1", "tech_doc2", "tech_doc3"]
    )

# --- 4. Retrieval-Augmented Generation (RAG) System --- #
# Simplified LLM for generation (distilgpt2)
generation_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
generation_model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL_NAME)
generation_pipeline = pipeline(
    "text-generation", 
    model=generation_model, 
    tokenizer=generation_tokenizer,
    max_new_tokens=100
)

def retrieve_and_generate(query: str, kb_collection, num_results: int = 2) -> str:
    results = kb_collection.query(
        query_texts=[query],
        n_results=num_results
    )
    context = " ".join(results['documents'][0]) if results['documents'] else ""
    
    if not context:
        return "I couldn't find relevant information for your query in the knowledge base."
    
    prompt = f"Based on the following context, answer the query:\nContext: {context}\nQuery: {query}\nAnswer:"
    response = generation_pipeline(prompt)[0]["generated_text"]
    return response.split("Answer:", 1)[-1].strip()

def multi_step_rag(query: str) -> str:
    # Simulate multi-step RAG: initial query to tech docs, then potentially refine or ask for more details
    initial_retrieval = tech_docs_kb.query(
        query_texts=[query],
        n_results=1
    )
    context = initial_retrieval['documents'][0][0] if initial_retrieval['documents'] else ""
    
    if not context:
        return "I couldn't find specific technical documentation for your complex query. Please provide more details or consider escalating to human support."
    
    # Simulate a 