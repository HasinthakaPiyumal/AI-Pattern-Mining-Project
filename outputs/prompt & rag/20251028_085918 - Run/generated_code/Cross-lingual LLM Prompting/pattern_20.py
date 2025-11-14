import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import re

# --- Configuration ---
MODEL_NAME = "google/flan-t5-base" # A good multilingual base model
MAX_LENGTH = 100 # Max length for generated responses
TEMPERATURE = 0.7 # For more diverse but controlled responses

# --- In-Context Learning Examples (Source + Target Language Mix) ---
# These examples demonstrate how to respond to queries in different languages
# and implicitly guide the model on which language to use for response
# based on the query and existing examples.
# The key is showing the model how to transfer knowledge by example.
INCLT_EXAMPLES = [
    {
        "query_lang": "English",
        "query": "I need help with my recent order, it hasn't arrived yet.",
        "response_lang": "Spanish",
        "response": "Necesito ayuda con mi pedido reciente, aún no ha llegado."
    },
    {
        "query_lang": "Spanish",
        "query": "¿Cuál es el estado de mi reembolso?",
        "response_lang": "English",
        "response": "What is the status of my refund?"
    },
    {
        "query_lang": "English",
        "query": "Can you update my shipping address?",
        "response_lang": "English",
        "response": "Please provide your new shipping address and I'll update it for you promptly."
    },
    {
        "query_lang": "Spanish",
        "query": "Mi producto está defectuoso, ¿cómo puedo devolverlo?",
        "response_lang": "Spanish",
        "response": "Lamento escuchar eso. Para devolver un producto defectuoso, por favor, sigue este enlace a nuestra política de devoluciones o contacta con nuestro equipo de soporte."
    },
    {
        "query_lang": "English",
        "query": "How do I activate my new service?",
        "response_lang": "Spanish", # Example of responding in a different language
        "response": "Para activar su nuevo servicio, por favor, visite nuestra página de configuración o siga las instrucciones en su correo electrónico de bienvenida."
    },
    {
        "query_lang": "Spanish",
        "query": "Necesito cambiar mi método de pago.",
        "response_lang": "English",
        "response": "To change your payment method, please go to your account settings and select 'Payment Options'."
    }
]

def identify_language(text: str) -> str:
    """
    A simple placeholder function to identify the language of the user's query.
    In a real application, a more robust language detection library (e.g., 'langdetect', 'fasttext')
    would be used for accurate results.
    """
    # Simple heuristic: count common Spanish vs. English indicators
    text_lower = text.lower()
    spanish_indicators = sum(1 for char in text_lower if char in "áéíóúñ¿¡")
    spanish_words = ["qué", "cómo", "dónde", "cuándo", "por qué", "hola", "gracias", "favor", "quiere", "no"]
    spanish_word_count = sum(1 for word in spanish_words if re.search(r'\b' + word + r'\b', text_lower))

    english_words = ["what", "how", "where", "when", "why", "hello", "thank", "please", "want", "no"]
    english_word_count = sum(1 for word in english_words if re.search(r'\b' + word + r'\b', text_lower))

    if spanish_indicators > 0 or spanish_word_count > english_word_count + 1: # Give Spanish a slight edge for special chars
        return "Spanish"
    elif english_word_count > spanish_word_count:
        return "English"
    return "English" # Default if not clearly Spanish

def get_target_response_language(user_lang: str) -> str:
    """
    Determines the target response language. For this demonstration,
    we alternate the response language to clearly show cross-lingual transfer.
    In a real chatbot, you might default to the user's input language or a configured preference.
    """
    return "Spanish" if user_lang == "English" else "English"

def construct_inclt_prompt(user_query: str, user_lang: str, target_response_lang: str) -> str:
    """
    Constructs a prompt using InCLT examples to guide the LLM for cross-lingual understanding
    and response generation in the target language.
    """
    prompt_header = (
        "Given the following customer support interactions, "
        "provide an appropriate response based on the context.\n"
        "You need to demonstrate cross-lingual understanding and respond in the specified target language."
    )
    
    prompt_examples = []
    for example in INCLT_EXAMPLES:
        prompt_examples.append(
            f"User ({example['query_lang']}): {example['query']}\n"
            f"Assistant ({example['response_lang']}): {example['response']}"
        )
    
    # Add the user's current query and specify the desired response language
    user_query_part = (
        f"User ({user_lang}): {user_query}\n"
        f"Assistant ({target_response_lang}):"
    )
    
    return "\n\n".join([prompt_header] + prompt_examples + [user_query_part])

def run_chatbot():
    print(f"Loading LLM: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        
        # Using pipeline for simpler text generation
        text_generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_LENGTH,
            temperature=TEMPERATURE,
            do_sample=True, # Enable sampling for temperature to have an effect
            device=0 if torch.cuda.is_available() else -1 # Use GPU if available
        )
        print("LLM loaded. Type 'exit' to quit.")
    except Exception as e:
        print(f"Error loading LLM: {e}")
        print("Please ensure you have an internet connection and the model can be downloaded.")
        return

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        user_lang = identify_language(user_input)
        print(f"[DEBUG] Detected input language: {user_lang}")
        
        target_response_lang = get_target_response_language(user_lang)
        print(f"[DEBUG] Target response language: {target_response_lang}")

        prompt = construct_inclt_prompt(user_input, user_lang, target_response_lang)
        # print(f"\n--- FULL PROMPT SENT TO LLM ---\n{prompt}\n-------------------------------\n") # Uncomment for debugging

        try:
            outputs = text_generator(prompt, num_return_sequences=1)
            response_text = outputs[0]['generated_text'].strip()
            print(f"Chatbot ({target_response_lang}): {response_text}")
        except Exception as e:
            print(f"Error generating response: {e}")
            print("Could not generate a response. Please try again.")

if __name__ == "__main__":
    run_chatbot()