import os

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo") # or "google/flan-t5-large"
    PRODUCT_CATALOG_PATH = "data/products.json"
    FAISS_INDEX_PATH = "data/product_embeddings.bin"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    TOP_K_CANDIDATES = 50
    DEFAULT_NUM_RECOMMENDATIONS = 5
