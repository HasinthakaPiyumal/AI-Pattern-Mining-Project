import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI API Key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Path for ChromaDB persistence
    # This will create a 'chroma_db' directory in the project root
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

    # Embedding model name for sentence-transformers
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # LLM model name (e.g., for Langchain)
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")

    # You can add other configurations here as needed, e.g., external API endpoints, thresholds

    @classmethod
    def validate_config(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        print("Configuration loaded successfully.")

# Example usage (for testing purposes, remove in production if not needed)
if __name__ == "__main__":
    Config.validate_config()
    print(f"OpenAI API Key (first 5 chars): {Config.OPENAI_API_KEY[:5]}...")
    print(f"ChromaDB Path: {Config.CHROMA_DB_PATH}")
    print(f"Embedding Model: {Config.EMBEDDING_MODEL_NAME}")
    print(f"LLM Model: {Config.LLM_MODEL_NAME}")
