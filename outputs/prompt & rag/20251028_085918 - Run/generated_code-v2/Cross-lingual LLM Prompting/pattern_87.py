"""Simulated Multilingual Customer Support Chatbot with InCLT Prompting."""

import json

# Simulated Product Database (primarily in English)
PRODUCT_DATABASE = {
    "PROD101": {
        "name": "Smartwatch X9",
        "description": "The Smartwatch X9 features a vibrant AMOLED display, 7-day battery life, and advanced health tracking. Compatible with Android and iOS devices.",
        "specifications": "Screen: 1.4-inch AMOLED, Battery: 200mAh, Water Resistance: IP68",
        "language": "en"
    },
    "PROD102": {
        "name": "Wireless Earbuds Pro",
        "description": "Experience crystal clear audio with the Wireless Earbuds Pro. Active Noise Cancellation, comfortable fit, and 30 hours of total playtime with the charging case.",
        "specifications": "Bluetooth: 5.2, ANC: Yes, Battery: 6 hours (earbuds), 24 hours (case)",
        "language": "en"
    }
}

# Simulated In-Context Learning Transfer (InCLT) Examples
# These examples demonstrate how to respond in one language (target) using context from another (source)
IN_CONTEXT_EXAMPLES = [
    {
        "query_lang": "es",
        "product_desc_lang": "en",
        "query": "¿Cuánto dura la batería del Smartwatch X9?",
        "product_description_snippet": "The Smartwatch X9 features... 7-day battery life...",
        "response": "El Smartwatch X9 tiene una duración de batería de 7 días. (The Smartwatch X9 has a 7-day battery life.)"
    },
    {
        "query_lang": "fr",
        "product_desc_lang": "en",
        "query": "Les écouteurs Wireless Earbuds Pro ont-ils la suppression du bruit?",
        "product_description_snippet": "Experience crystal clear audio with the Wireless Earbuds Pro. Active Noise Cancellation...",
        "response": "Oui, les écouteurs Wireless Earbuds Pro sont équipés d'une suppression active du bruit. (Yes, the Wireless Earbuds Pro headphones are equipped with active noise cancellation.)"
    },
    {
        "query_lang": "de",
        "product_desc_lang": "en",
        "query": "Ist die Smartwatch X9 wasserdicht?",
        "product_description_snippet": "...Water Resistance: IP68",
        "response": "Ja, die Smartwatch X9 ist IP68 wasserdicht. (Yes, the Smartwatch X9 is IP68 waterproof.)"
    }
]

def detect_language(text: str) -> str:
    """Simulates language detection. In a real app, use a proper library like langdetect."""
    text = text.lower()
    if any(word in text for word in ["hola", "qué", "cuánto", "gracias", "batería"]):
        return "es" # Spanish
    elif any(word in text for word in ["bonjour", "comment", "quel", "merci", "écouteurs"]):
        return "fr" # French
    elif any(word in text for word in ["hallo", "wie", "was", "danke", "wasserdicht"]):
        return "de" # German
    return "en" # Default to English

def get_product_details(product_id: str) -> dict or None:
    """Simulates retrieving product details from a database."""
    return PRODUCT_DATABASE.get(product_id)

def build_inclt_prompt(
    customer_query: str,
    source_language: str,
    product_details: dict or None,
    in_context_examples: list
) -> str:
    """Constructs a prompt using the InCLT pattern."""
    prompt_parts = []

    # System instruction
    prompt_parts.append(
        "You are a helpful multilingual customer support chatbot for an e-commerce platform. "
        "Your goal is to answer customer questions accurately, leveraging provided product information. "
        "When answering, ensure you respond in the language of the customer's query, even if "
        "the product information is in a different language. Use the cross-lingual examples below "
        "to guide your understanding and response generation."
    )
    prompt_parts.append("""
Here are some examples of how to answer queries in different languages, using product context that might be in English:
""")

    # InCLT Examples
    for example in in_context_examples:
        prompt_parts.append(f"""
Customer Query ({example['query_lang'].upper()}): {example['query']}
Product Information (EN): {example['product_description_snippet']}
Chatbot Response ({example['query_lang'].upper()}): {example['response']}
""")

    # Current Product Context
    if product_details:
        prompt_parts.append("""
--- Current Product Information ---
""")
        for key, value in product_details.items():
            prompt_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        prompt_parts.append("""
-----------------------------------
""")
    else:
        prompt_parts.append("No specific product details available for this query.")

    # Customer's Actual Query
    prompt_parts.append(f"Customer Query ({source_language.upper()}): {customer_query}")
    prompt_parts.append(f"Chatbot Response ({source_language.upper()}):")

    return "\n".join(prompt_parts)

def simulate_llm_response(prompt: str) -> str:
    """Simulates an LLM's response based on the prompt.
    In a real application, this would call an actual LLM API.
    """
    # Simple keyword-based simulation to show cross-lingual understanding
    if "¿Cuánto dura la batería" in prompt and "7-day battery life" in prompt:
        return "El Smartwatch X9 tiene una duración de batería de 7 días."
    elif "suppression du bruit" in prompt and "Active Noise Cancellation" in prompt:
        return "Oui, les écouteurs Wireless Earbuds Pro sont équipés d'une suppression active du bruit."
    elif "wasserdicht" in prompt and "IP68" in prompt:
        return "Ja, die Smartwatch X9 ist IP68 wasserdicht."
    elif "price" in prompt.lower():
        return "I'm sorry, I don't have information about the price. Please check the product page."
    elif "hola" in prompt.lower():
        return "¡Hola! ¿En qué puedo ayudarte hoy?"
    elif "qué idiomas" in prompt.lower():
        return "Puedo ayudarte en español, inglés, francés y alemán. ¿Cómo puedo ayudarte?"
    return "I understand you're asking about the product. Based on the information, I can tell you that it's designed for optimal performance. Please let me know if you have specific questions, and I'll do my best to answer them in your language, leveraging all available product details."

def main():
    print("--- InCLT Multilingual Customer Support Chatbot Demo ---")

    # Example 1: Spanish query about Smartwatch X9 battery life
    customer_query_1 = "¿Cuánto dura la batería del Smartwatch X9?"
    product_id_1 = "PROD101"
    print(f"\nCustomer: {customer_query_1}")
    
    source_lang_1 = detect_language(customer_query_1)
    product_details_1 = get_product_details(product_id_1)
    
    prompt_1 = build_inclt_prompt(
        customer_query_1, source_lang_1, product_details_1, IN_CONTEXT_EXAMPLES
    )
    # print(f"\n--- Generated Prompt 1 ---\n{prompt_1}\n--------------------------") # Uncomment to see the full prompt
    response_1 = simulate_llm_response(prompt_1)
    print(f"Chatbot: {response_1}")

    # Example 2: French query about Wireless Earbuds Pro ANC
    customer_query_2 = "Les écouteurs Wireless Earbuds Pro ont-ils la suppression du bruit?"
    product_id_2 = "PROD102"
    print(f"\nCustomer: {customer_query_2}")

    source_lang_2 = detect_language(customer_query_2)
    product_details_2 = get_product_details(product_id_2)
    
    prompt_2 = build_inclt_prompt(
        customer_query_2, source_lang_2, product_details_2, IN_CONTEXT_EXAMPLES
    )
    # print(f"\n--- Generated Prompt 2 ---\n{prompt_2}\n--------------------------") # Uncomment to see the full prompt
    response_2 = simulate_llm_response(prompt_2)
    print(f"Chatbot: {response_2}")

    # Example 3: German query about Smartwatch X9 water resistance
    customer_query_3 = "Ist die Smartwatch X9 wasserdicht?"
    product_id_3 = "PROD101"
    print(f"\nCustomer: {customer_query_3}")
    
    source_lang_3 = detect_language(customer_query_3)
    product_details_3 = get_product_details(product_id_3)
    
    prompt_3 = build_inclt_prompt(
        customer_query_3, source_lang_3, product_details_3, IN_CONTEXT_EXAMPLES
    )
    # print(f"\n--- Generated Prompt 3 ---\n{prompt_3}\n--------------------------") # Uncomment to see the full prompt
    response_3 = simulate_llm_response(prompt_3)
    print(f"Chatbot: {response_3}")
    
    # Example 4: English query without specific product ID but general info
    customer_query_4 = "Hello, what languages can you help me in?"
    product_id_4 = None # No specific product
    print(f"\nCustomer: {customer_query_4}")
    
    source_lang_4 = detect_language(customer_query_4)
    product_details_4 = get_product_details(product_id_4) # Will be None
    
    prompt_4 = build_inclt_prompt(
        customer_query_4, source_lang_4, product_details_4, IN_CONTEXT_EXAMPLES
    )
    response_4 = simulate_llm_response(prompt_4)
    print(f"Chatbot: {response_4}")


if __name__ == "__main__":
    main()