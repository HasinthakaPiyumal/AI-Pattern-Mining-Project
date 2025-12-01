"""Configuration file for the Automated Medical Diagnostic Assistant Evaluation Dataset Generator."""

import os

# --- LLM Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
# Other potential LLM configurations (e.g., model names, temperatures)

# --- Data Paths ---
MEDICAL_KNOWLEDGE_BASE_PATH = "./data/medical_knowledge_base.txt"
GENERATED_DATASET_PATH = "./data/medical_qa_dataset.json"

# --- Question Generation Parameters ---
QUESTION_GENERATION_TEMPERATURE = 0.7
QUESTION_GENERATION_MODEL = "gpt-4o"

# --- Answer Generation Parameters ---
ANSWER_GENERATION_MODEL = "gpt-4o"

# --- Dataset Generation Parameters ---
NUM_QUESTIONS_TO_GENERATE = 10