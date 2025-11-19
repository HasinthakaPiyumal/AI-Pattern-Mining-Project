import os

class Config:
    # Placeholder for API keys and other configurations
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    # Add other LLM API keys here if needed
    # Example: HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    # Configuration for simulated knowledge base
    KNOWLEDGE_BASE_PATH = "data/medical_knowledge.json" # Not used in this basic sim, but good practice

    # Thresholds for self-reflection
    CONFIDENCE_THRESHOLD = 0.7
    RETRIEVAL_ITERATION_LIMIT = 3
