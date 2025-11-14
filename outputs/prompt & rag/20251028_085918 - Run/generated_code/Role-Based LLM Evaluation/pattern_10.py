import os

# Set your OpenAI API key here or as an environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# Ensure the API key is set
if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
    raise ValueError("OPENAI_API_KEY not set. Please set it in config.py or as an environment variable.")
