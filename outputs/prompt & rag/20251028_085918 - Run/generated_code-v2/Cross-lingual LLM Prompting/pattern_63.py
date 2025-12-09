import json
from langdetect import detect, LangDetectException
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# --- 1. In-context Example Database ---
# These examples demonstrate cross-lingual understanding by providing translations
# of the same query/answer pairs across multiple languages.
INCLT_EXAMPLES = [
    {
        "description": "Order Status Inquiry",
        "query_en": "Where is my order #XYZ123?",
        "answer_en": "Your order #XYZ123 is currently being prepared for shipment and is expected to arrive within 3-5 business days.",
        "query_es": "¿Dónde está mi pedido #XYZ123?",
        "answer_es": "Su pedido #XYZ123 está siendo preparado para el envío y se espera que llegue en 3-5 días hábiles.",
        "query_fr": "Où est ma commande #XYZ123?",
        "answer_fr": "Votre commande #XYZ123 est en cours de préparation pour l'expédition et devrait arriver dans les 3-5 jours ouvrables."
    },
    {
        "description": "Return Policy Question",
        "query_en": "What is your return policy?",
        "answer_en": "Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging.",
        "query_es": "¿Cuál es su política de devolución?",
        "answer_es": "Nuestra política de devolución permite devoluciones dentro de los 30 días posteriores a la compra, siempre que el artículo no haya sido utilizado y esté en su embalaje original.",
        "query_fr": "Quelle est votre politique de retour ?",
        "answer_fr": "Notre politique de retour autorise les retours dans les 30 jours suivant l'achat, à condition que l'article soit inutilisé et dans son emballage d'origine."
    },
    {
        "description": "Product Information",
        "query_en": "Tell me about product 'Eco-Friendly Water Bottle'.",
        "answer_en": "The Eco-Friendly Water Bottle is made from recycled materials, has a 750ml capacity, and comes with a leak-proof design. It's perfect for daily use and outdoor activities.",
        "query_es": "Háblame del producto 'Botella de Agua Ecológica'.",
        "answer_es": "La Botella de Agua Ecológica está hecha de materiales reciclados, tiene una capacidad de 750ml y viene con un diseño a prueba de fugas. Es perfecta para el uso diario y actividades al aire libre.",
        "query_fr": "Parlez-moi du produit 'Bouteille d'eau écologique'.",
        "answer_fr": "La Bouteille d'eau écologique est fabriquée à partir de matériaux recyclés, a une capacité de 750 ml et est dotée d'un design anti-fuite. Elle est parfaite pour un usage quotidien et les activités de plein air."
    }
]

# --- 2. Prompt Engineering Module (InCLT Implementation) ---
def get_icl_prompt(user_query: str, detected_lang: str) -> str:
    """Constructs a prompt with InCLT examples for the LLM."""
    prompt_parts = []
    prompt_parts.append(
        "You are a helpful customer support assistant. Below are examples of how to respond to customer queries in multiple languages. These examples demonstrate cross-lingual understanding.\n\n"
    )

    for i, example in enumerate(INCLT_EXAMPLES):
        prompt_parts.append(f"Example {i+1}:\n")
        if "query_en" in example and "answer_en" in example:
            prompt_parts.append(f"Customer Query (English): {example['query_en']}\n")
            prompt_parts.append(f"Assistant Response (English): {example['answer_en']}\n")
        if "query_es" in example and "answer_es" in example:
            prompt_parts.append(f"Customer Query (Spanish): {example['query_es']}\n")
            prompt_parts.append(f"Assistant Response (Spanish): {example['answer_es']}\n")
        if "query_fr" in example and "answer_fr" in example:
            prompt_parts.append(f"Customer Query (French): {example['query_fr']}\n")
            prompt_parts.append(f"Assistant Response (French): {example['answer_fr']}\n")
        prompt_parts.append("\n")

    prompt_parts.append(
        f"Now, analyze the following customer query in its original language and provide an insightful response IN ENGLISH. We will then translate your English response.\n\n"
    )
    prompt_parts.append(f"Customer Query ({detected_lang.upper()}): {user_query}\n")
    prompt_parts.append(f"Assistant Response (English):")

    return "".join(prompt_parts)


# --- 3. Language Detection Module ---
def detect_language(text: str) -> str:
    """Detects the language of the given text."""
    try:
        return detect(text)
    except LangDetectException:
        return "en"  # Default to English if detection fails


# --- 4. Multilingual Large Language Model (LLM) and Translation ---
# Using a small, general-purpose LLM for demonstration.
# In a real application, a much larger and truly multilingual LLM would be used.
# The LLM will generate an English response, which will then be translated.

# Placeholder for a generative LLM (e.g., GPT-2). For actual multilingual generation,
# consider models like 'Helsinki-NLP/opus-mt-en-mul' or larger LLMs fine-tuned for multilingual tasks.
# device = 0 if torch.cuda.is_available() else -1
llm_pipeline = pipeline(
    "text-generation",
    model="gpt2",
    # device=device, # Uncomment if you have a GPU
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1
)

# Translation pipelines - dynamically loaded based on target language
translators = {}

# Mapping langdetect codes to Opus-MT model codes
LANG_MAP = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    # Add more mappings as needed
}

def get_translator(target_lang_code: str):
    """Returns a translation pipeline for en->target_lang."""
    if target_lang_code not in translators:
        if target_lang_code not in LANG_MAP: # If target_lang is not supported for translation, fallback to English
            print(f"Warning: No specific translator for {target_lang_code}. Falling back to English LLM response.")
            return None
        model_name = f"Helsinki-NLP/opus-mt-en-{LANG_MAP[target_lang_code]}"
        try:
            # translators[target_lang_code] = pipeline("translation", model=model_name, device=device) # Uncomment if GPU
            translators[target_lang_code] = pipeline("translation", model=model_name)
            print(f"Loaded translator: {model_name}")
        except Exception as e:
            print(f"Error loading translator {model_name}: {e}")
            return None
    return translators[target_lang_code]


def generate_response(prompt: str, detected_lang: str) -> str:
    """Generates an English response using the LLM and then translates it if needed."""
    # 1. LLM generates response in English
    llm_output = llm_pipeline(prompt)[0]["generated_text"]
    # Extract only the assistant's part after the prompt
    english_response = llm_output.split("Assistant Response (English):")[-1].strip()
    
    # Clean up any potential prompt repetition by the LLM
    if "Customer Query" in english_response:
        english_response = english_response.split("Customer Query")[0].strip()

    print(f"\nLLM (English) Raw Output:\n{llm_output}")
    print(f"Extracted English Response: {english_response}")

    # 2. Translate if the detected language is not English
    if detected_lang != "en":
        translator = get_translator(detected_lang)
        if translator:
            try:
                translated_text = translator(english_response)[0]["translation_text"]
                return translated_text
            except Exception as e:
                print(f"Error during translation: {e}. Returning English response.")
                return english_response
        else:
            return english_response # Return English if no translator available
    return english_response


# --- 5. User Interface (Command-line) ---
def main():
    print("Welcome to the Multilingual Customer Support Chatbot!")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Thank you for chatting! Goodbye.")
            break

        if not user_input.strip():
            print("Chatbot: Please enter a query.")
            continue

        detected_lang = detect_language(user_input)
        print(f"Detected language: {detected_lang.upper()}")

        icl_prompt = get_icl_prompt(user_input, detected_lang)
        # print(f"\n--- Generated Prompt ---\n{icl_prompt}\n------------------------\n") # Uncomment to see the full prompt

        chatbot_response = generate_response(icl_prompt, detected_lang)
        print(f"Chatbot ({detected_lang.upper()}): {chatbot_response}")

if __name__ == "__main__":
    main()