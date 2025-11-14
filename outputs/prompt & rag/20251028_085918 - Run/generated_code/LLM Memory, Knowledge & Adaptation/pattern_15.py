import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Add other configurations like ChromaDB path, etc.
    CHROMA_DB_PATH = "./chroma_db"
