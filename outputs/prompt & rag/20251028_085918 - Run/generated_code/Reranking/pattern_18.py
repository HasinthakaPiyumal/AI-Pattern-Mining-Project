"""
Configuration file for MedInfo-Assist.
"""

import os

class Config:
    # OpenAI API Key for Language Models and Embeddings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

    # Embedding model name (e.g., for Sentence Transformers or OpenAI embeddings)
    EMBEDDING_MODEL_NAME = "text-embedding-ada-002"  # OpenAI embedding model
    # Fallback for local models if no API key or different setup needed
    LOCAL_EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

    # Main Language Model for generation
    GENERATION_MODEL_NAME = "gpt-4o-mini" # OpenAI powerful and cost-effective model

    # Vector Database settings
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "medical_literature"

    # Reranker model (if using a fine-tuned model)
    RERANKER_MODEL_PATH = "./models/predictive_reranker_model.pkl"

    # Conditional Retrieval Model path
    CONDITIONAL_RETRIEVAL_MODEL_PATH = "./models/conditional_retrieval_classifier.pkl"

    # Number of documents to retrieve initially
    TOP_K_RETRIEVAL = 10

    # Number of documents to keep after reranking
    TOP_K_RERANKED = 5

    # Placeholder for a simple keyword list for conditional retrieval (can be expanded)
    MEDICAL_KEYWORDS = ["treatment", "diagnosis", "syndrome", "drug", "medication", "pathology", "clinical", "patient", "therapy", "complication", "prognosis"]
