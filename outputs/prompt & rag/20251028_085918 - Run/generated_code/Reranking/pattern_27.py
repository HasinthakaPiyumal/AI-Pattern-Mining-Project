import os

# Configuration for the Medical Assistant Chatbot

# API Keys for Language Models (e.g., OpenAI, Google Gemini)
# It's recommended to load these from environment variables for security
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

# Default Language Model settings
DEFAULT_LM_MODEL = "gpt-3.5-turbo" # Example: "gpt-4", "gemini-pro"
TEMPERATURE = 0.7
MAX_TOKENS = 500

# Retrieval System Configuration
# In a real application, this would point to actual database endpoints or APIs
MEDICAL_DB_MOCK_FILE = "medical_knowledge_base_mock.py" # Path to mock knowledge base
RETRIEVAL_TOP_K = 5 # Number of documents to retrieve initially

# Reranker Configuration
RERANKER_MODEL = "zero-shot-lm" # Options: "zero-shot-lm", "trained-model" (placeholder)

# Prompt engineering settings
SYSTEM_PROMPT = (
    "You are a highly knowledgeable and accurate medical assistant AI, designed to provide information "
    "to healthcare professionals. Always cite the provided documents for your answers and state if you "
    "cannot find the answer in the provided context." 
)

# Enable/disable conditional retrieval (for future advanced implementation)
CONDITIONAL_RETRIEVAL_ENABLED = True
