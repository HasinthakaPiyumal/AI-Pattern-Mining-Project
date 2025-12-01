from langdetect import detect, LangDetectException
import random

# 2. Multilingual In-Context Learning (InCL) Example Database
in_context_examples = {
    "shipping_delay": [
        {
            "EN": {"query": "Where is my order?", "response": "Your order is delayed.", "category": "shipping_delay"},
            "FR": {"query": "Où est ma commande ?", "response": "Votre commande est retardée.", "category": "shipping_delay"},
            "ES": {"query": "¿Dónde está mi pedido?", "response": "Su pedido está retrasado.", "category": "shipping_delay"}
        },
        {
            "EN": {"query": "My package is late.", "response": "We apologize for the delay. Please provide your order number.", "category": "shipping_delay"},
            "FR": {"query": "Mon colis est en retard.", "response": "Nous nous excusons pour le retard. Veuillez fournir votre numéro de commande.", "category": "shipping_delay"},
            "DE": {"query": "Mein Paket ist verspätet.", "response": "Wir entschuldigen uns für die Verzögerung. Bitte geben Sie Ihre Bestellnummer an.", "category": "shipping_delay"}
        }
    ],
    "return_request": [
        {
            "EN": {"query": "I want to return an item.", "response": "Please provide your order number and reason for return.", "category": "return_request"},
            "FR": {"query": "Je veux retourner un article.", "response": "Veuillez fournir votre numéro de commande et la raison du retour.", "category": "return_request"},
            "ES": {"query": "Quiero devolver un artículo.", "response": "Por favor, proporcione su número de pedido y el motivo de la devolución.", "category": "return_request"}
        },
        {
            "EN": {"query": "How do I send back a product?", "response": "You can initiate a return through our website within 30 days of purchase.", "category": "return_request"},
            "FR": {"query": "Comment puis-je renvoyer un produit ?", "response": "Vous pouvez initier un retour via notre site web dans les 30 jours suivant l'achat.", "category": "return_request"},
            "DE": {"query": "Wie schicke ich ein Produkt zurück?", "response": "Sie können eine Rücksendung innerhalb von 30 Tagen nach dem Kauf über unsere Website veranlassen.", "category": "return_request"}
        }
    ]
}

# 1. Language Detection Module
def detect_language(text):
    try:
        return detect(text).upper()
    except LangDetectException:
        return "EN" # Default to English if detection fails

# 3. Prompt Engineering Module (InCLT Implementation)
def create_inclt_prompt(query, detected_lang, target_langs, num_examples=2):
    prompt_parts = ["Here are examples of customer inquiries and their categories/responses:"]
    
    # For simplicity, we'll try to guess a relevant category or pick randomly
    relevant_category = None
    for category, examples_list in in_context_examples.items():
        if any(keyword in query.lower() for keyword in category.split("_")):
            relevant_category = category
            break
    if not relevant_category and in_context_examples:
        relevant_category = random.choice(list(in_context_examples.keys()))

    if relevant_category:
        selected_examples = random.sample(in_context_examples[relevant_category], min(num_examples, len(in_context_examples[relevant_category])))
        for example in selected_examples:
            for lang in sorted(list(example.keys())):
                if lang in target_langs or lang == detected_lang:
                    prompt_parts.append(f"\n{lang} Query: \"{example[lang]["query"]}\"\n{lang} Response: \"{example[lang]["response"]}\"")

    prompt_parts.append(f"\n\nCustomer Query ({detected_lang}): \"{query}\"\nAssistant Response:")
    return "\n".join(prompt_parts)

# 4. Multilingual Large Language Model (LLM) - Simulated
def call_multilingual_llm(prompt):
    # In a real application, this would interface with a transformer model or API
    # For demonstration, we'll return a simple mock response.
    if "Where is my order?" in prompt and "shipping_delay" in prompt:
        return "Your order is currently in transit and is expected to arrive within 2-3 business days. You can track it using the provided tracking number."
    elif "return an item" in prompt and "return_request" in prompt:
        return "To initiate a return, please visit our returns portal on the website and follow the instructions. You will need your order number."
    else:
        return "Thank you for contacting customer support. How can I further assist you today?"

# 5. Response Generation and Translation (Simplified Post-processing)
def format_and_translate_response(response_text, target_lang):
    # In a real system, this would translate the LLM's output to target_lang if needed
    # For this example, we assume the LLM responds in a globally understandable way or the target_lang is handled implicitly.
    return response_text

def customer_support_assistant(query):
    detected_lang = detect_language(query)
    
    # Define target languages for InCLT. Always include detected_lang.
    # For this example, let's always include EN, FR, ES, DE if available in examples.
    available_langs = set()
    for category_examples in in_context_examples.values():
        for ex in category_examples:
            available_langs.update(ex.keys())
    
    target_langs = list(available_langs)
    if detected_lang not in target_langs:
        target_langs.append(detected_lang)

    inclt_prompt = create_inclt_prompt(query, detected_lang, target_langs, num_examples=1) # Using 1 example for brevity in output
    llm_response = call_multilingual_llm(inclt_prompt)
    final_response = format_and_translate_response(llm_response, detected_lang)
    
    print(f"Detected Language: {detected_lang}")
    print("\n--- Generated Prompt ---")
    print(inclt_prompt)
    print("\n--- LLM Response ---")
    print(final_response)
    return final_response

if __name__ == "__main__":
    print("\n--- Test Case 1: English Shipping Inquiry ---")
    customer_support_assistant("Where is my order?")

    print("\n--- Test Case 2: French Return Inquiry ---")
    customer_support_assistant("Je veux retourner un article.")

    print("\n--- Test Case 3: Spanish Shipping Inquiry ---")
    customer_support_assistant("¿Dónde está mi pedido?")

    print("\n--- Test Case 4: General Inquiry (English) ---")
    customer_support_assistant("I need help with something else.")
