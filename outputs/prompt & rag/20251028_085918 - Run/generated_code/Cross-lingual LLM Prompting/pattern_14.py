from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, DetectorFactory
import gradio as gr
import re

# Ensure consistent language detection results
DetectorFactory.seed = 0

# 1. Multilingual Language Model (LLM)
# Using a smaller model for demonstration. For broader coverage, consider 'facebook/mbart-large-50-many-to-many-mmt'
MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("Model loaded successfully.")

# 2. Product Knowledge Base (primarily in English)
product_knowledge_base = {
    "AquaFlow Water Bottle": {
        "id": "AWB001",
        "name": "AquaFlow Water Bottle",
        "description": "A durable and insulated water bottle.",
        "features": "1-liter capacity, double-walled insulation, BPA-free plastic, keeps water cold for 24 hours.",
        "price": "$25"
    },
    "EcoMug Coffee Cup": {
        "id": "ECC002",
        "name": "EcoMug Coffee Cup",
        "description": "A stylish and eco-friendly coffee mug.",
        "features": "350ml capacity, stainless steel, spill-proof lid, keeps coffee hot for 6 hours.",
        "price": "$18"
    },
    "LuxeComfort Headphones": {
        "id": "LCH003",
        "name": "LuxeComfort Headphones",
        "description": "Premium over-ear headphones with noise cancellation.",
        "features": "Active noise cancellation, 30-hour battery life, ergonomic design, Bluetooth 5.0.",
        "price": "$199"
    }
}

def get_product_info(query):
    """Retrieves relevant product information based on keywords in the query."""
    query_lower = query.lower()
    for product_name, details in product_knowledge_base.items():
        if product_name.lower() in query_lower or details['id'].lower() in query_lower:
            return f"Product Info (English): {product_name}: {details['features']} Price: {details['price']}."
    return ""

# 4. InCLT (In-Context Learning with Cross-Lingual Transfer) Prompting Module
def construct_inclt_prompt(customer_query, detected_language, retrieved_product_info):
    """Constructs the prompt using the InCLT pattern."""
    base_prompt = (
        "You are a helpful multilingual customer support assistant. You can answer questions about products in different languages, even if the information comes from another language.\n\n"
        "Here are some examples:\n\n"
        "User (English): Tell me about the 'AquaFlow Water Bottle'.\n"
        "Product Info (English): AquaFlow Water Bottle: 1-liter capacity, double-walled insulation, BPA-free plastic, keeps water cold for 24 hours. Price: $25.\n"
        "Assistant (English): The AquaFlow Water Bottle has a 1-liter capacity, double-walled insulation, and is made from BPA-free plastic. It can keep your water cold for up to 24 hours and costs $25.\n\n"
        "User (Spanish): ¿Cuál es la capacidad de la 'EcoMug Coffee Cup'?\n"
        "Product Info (English): EcoMug Coffee Cup: 350ml capacity, stainless steel, spill-proof lid, keeps coffee hot for 6 hours. Price: $18.\n"
        "Assistant (Spanish): La Taza de Café EcoMug tiene una capacidad de 350ml, es de acero inoxidable y tiene una tapa a prueba de derrames. Mantiene el café caliente durante 6 horas.\n\n"
    )

    # Add the current user query and retrieved product info
    prompt = f"{base_prompt}User ({detected_language.capitalize()}): {customer_query}\n"
    if retrieved_product_info:
        prompt += f"{retrieved_product_info}\n"
    prompt += f"Assistant ({detected_language.capitalize()}): "
    
    return prompt

# 5. Chatbot Logic (Main Application Flow)
def chatbot_response(user_query, history):
    """Generates a response to the user query using the LLM and InCLT prompting."""
    print(f"Received query: {user_query}")
    
    try:
        detected_language = detect(user_query)
        print(f"Detected language: {detected_language}")
    except Exception as e:
        detected_language = 'en' # Default to English if detection fails
        print(f"Language detection failed: {e}. Defaulting to English.")

    # Retrieve relevant product info
    retrieved_product_info = get_product_info(user_query)
    print(f"Retrieved product info: {retrieved_product_info if retrieved_product_info else 'None'}")

    # Construct the InCLT prompt
    prompt = construct_inclt_prompt(user_query, detected_language, retrieved_product_info)
    print(f"Constructed prompt:\n{prompt}")

    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Adjust max_new_tokens based on expected response length. 
    # A small model like opus-mt might struggle with very long generations.
    output_sequences = model.generate(
        inputs["input_ids"],
        num_beams=5, 
        min_length=1, 
        max_new_tokens=150, # Limit generation to prevent excessively long or repetitive output
        early_stopping=True
    )
    
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    print(f"LLM Raw Output: {generated_text}")

    # Post-process to extract only the assistant's response part if the model generates the prompt again
    # This part can be tricky with seq2seq models if they don't strictly adhere to the prompt format.
    # For Helsinki-NLP/opus-mt, it's primarily a translation model, so it might just translate the last sentence.
    # A more advanced LLM would be better at following the 