"""
Configuration settings for the E-commerce Product Recommender System.
"""

# LLM Settings
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY" # Replace with your actual OpenAI API key
LLM_MODEL_NAME = "gpt-4" # Or any other suitable LLM model (e.g., "gpt-3.5-turbo")

# Recommender System Settings
DATASET_SIZE = 1000 # Simulated dataset size
NUM_PRODUCTS = 100
NUM_USERS = 50
LATENT_DIM = 64 # Default latent dimension for simulated architectures

# Genetic Algorithm NAS Settings
POPULATION_SIZE = 10
NUM_GENERATIONS = 5
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8

# Blackbox Optimizer Settings
BLACKBOX_OPTIMIZER_ITERATIONS = 3
