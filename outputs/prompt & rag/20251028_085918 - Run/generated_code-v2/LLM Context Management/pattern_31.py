INDEX_DIR = "faiss_indices"
# Example data for initial setup and updates
INITIAL_NEWS_DATA = [
    "The capital of France is Paris.",
    "The Eiffel Tower is in Paris.",
    "The Louvre Museum is a famous landmark in Paris.",
    "Emmanuel Macron is the current president of France."
]

UPDATED_NEWS_DATA = [
    "The capital of France is Paris.",
    "The Eiffel Tower is in Paris.",
    "The Louvre Museum is a famous landmark in Paris.",
    "Gabriel Attal was appointed Prime Minister of France in January 2024.", # Updated information
    "The 2024 Summer Olympics will be held in Paris."
]

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Note: For a real application, consider a larger model or a hosted API.
# This model is small and can run locally for demonstration.
LLM_MODEL_NAME = "google/flan-t5-small"