# This script demonstrates the 'InCLT Crosslingual Transfer Prompting' pattern for a multilingual customer support chatbot.
# It simulates an LLM leveraging both source and target languages (English as internal pivot) for in-context learning.

# 1. In-context Examples Data
# This list holds pre-defined cross-lingual examples used to construct the prompt.
# Each example includes the customer query in multiple languages, an internal context description (often in English),
# and the expected response in multiple languages. This structure facilitates cross-lingual transfer.
CROSS_LINGUAL_EXAMPLES = [
    {
        "id": "order_status_spanish",
        "customer_query": {"es": "¿Dónde está mi pedido?", "en": "Where is my order?"},
        "context_info": "Customer inquiring about order status. Needs order number.",
        "response": {"es": "Por favor, proporcióneme su número de pedido para que pueda ayudarle.", "en": "Please provide your order number so I can assist you.", "de": "Bitte geben Sie Ihre Bestellnummer an, damit ich Ihnen helfen kann.", "fr": "Veuillez me fournir votre numéro de commande afin que je puisse vous aider."
        }
    },
    {
        "id": "return_policy_german",
        "customer_query": {"de": "Wie ist Ihre Rückgaberichtlinie?", "en": "What is your return policy?"},
        "context_info": "Customer asking about return policy. Needs a general explanation.",
        "response": {"de": "Unsere Rückgaberichtlinie erlaubt Rücksendungen innerhalb von 30 Tagen nach dem Kauf.", "en": "Our return policy allows returns within 30 days of purchase.", "es": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra.", "fr": "Notre politique de retour autorise les retours dans les 30 jours suivant l'achat."
        }
    },
    {
        "id": "product_info_french",
        "customer_query": {"fr": "Ce produit est-il disponible en d'autres couleurs ?", "en": "Is this product available in other colors?"},
        "context_info": "Customer asking about product variations. Needs to check inventory.",
        "response": {"fr": "Oui, ce produit est disponible en rouge, bleu et vert. Quelle couleur préférez-vous ?", "en": "Yes, this product is available in red, blue, and green. Which color do you prefer?", "es": "Sí, este producto está disponible en rojo, azul y verde. ¿Qué color prefiere?", "de": "Ja, dieses Produkt ist in Rot, Blau und Grün erhältlich. Welche Farbe bevorzugen Sie?"
        }
    }
]

# 2. Prompt Generator Function
# This function constructs the prompt for the LLM, incorporating the InCLT pattern.
# It includes instructions for the LLM, followed by in-context examples that show
# a customer query in a source language, an internal cross-lingual understanding (via English),
# and the desired response in the source language. This helps the LLM learn how to transfer knowledge.
def generate_inclt_prompt(customer_query: str, customer_language: str, examples: list) -> str:
    prompt_parts = []
    prompt_parts.append(f"You are a helpful multilingual customer support assistant for an e-commerce platform.")
    prompt_parts.append(f"Your goal is to understand customer queries in any language and provide accurate responses in the customer's language.")
    prompt_parts.append(f"To achieve this, you will leverage cross-lingual understanding by observing examples that demonstrate concepts in both the source (customer's) language and an internal target language (English for clarification).")
    prompt_parts.append(f"\nHere are some examples of how to handle customer queries:")

    # Add in-context examples demonstrating cross-lingual transfer
    for example in examples[:2]: # Using first 2 examples for brevity in the demo prompt
        # Prioritize examples that directly match the customer's language, otherwise use English for consistency
        query_in_customer_lang = example["customer_query"].get(customer_language, example["customer_query"]["en"])
        response_in_customer_lang = example["response"].get(customer_language, example["response"]["en"])

        prompt_parts.append(f"\n--- Example ---")
        prompt_parts.append(f"Customer ({customer_language.upper()}): \"{query_in_customer_lang}\"")
        prompt_parts.append(f"Internal Thought (Cross-lingual understanding via English): \"{example['customer_query']['en']} - {example['context_info']}\"")
        prompt_parts.append(f"Response ({customer_language.upper()}): \"{response_in_customer_lang}\"")

    prompt_parts.append(f"\n--- New Customer Query ---")
    prompt_parts.append(f"Customer ({customer_language.upper()}): \"{customer_query}\"")
    prompt_parts.append(f"Internal Thought (Cross-lingual understanding via English): ") # Placeholder for LLM's thought
    prompt_parts.append(f"Response ({customer_language.upper()}): ") # Placeholder for LLM's response

    return "\n".join(prompt_parts)

# 3. LLM Simulator Function
# This function simulates the behavior of a multilingual LLM. In a real application,
# this would involve an actual API call to a large language model. For this demo,
# it parses the query and generates a response based on keywords and the specified language.
def simulate_llm_response(full_prompt: str) -> str:
    # Extract the customer query and language from the end of the prompt
    customer_query_line = [line for line in full_prompt.split("\n") if line.startswith("Customer (") and line.endswith("): ")][-1]
    query_start_index = customer_query_line.find('): "') + 4
    query_end_index = customer_query_line.rfind('"')
    customer_query = customer_query_line[query_start_index:query_end_index]

    lang_start_index = customer_query_line.find('(') + 1
    lang_end_index = customer_query_line.find(')')
    customer_language_code = customer_query_line[lang_start_index:lang_end_index].lower()

    # Simulate internal thought process based on the query and a simple keyword matching
    internal_thought = f"The customer query is: \"{customer_query}\". "
    simulated_response = ""

    if "order" in customer_query.lower() or "pedido" in customer_query.lower() or "bestellung" in customer_query.lower() or "commande" in customer_query.lower():
        internal_thought += "This seems to be about an order. Need to ask for order number."
        simulated_response = {
            "es": "Para ayudarle con su pedido, por favor, indíqueme su número de pedido.",
            "en": "To help you with your order, please provide your order number.",
            "de": "Um Ihnen bei Ihrer Bestellung zu helfen, geben Sie bitte Ihre Bestellnummer an.",
            "fr": "Pour vous aider avec votre commande, veuillez me donner votre numéro de commande."
        }.get(customer_language_code, "I need your order number to help with your request.")
    elif "return" in customer_query.lower() or "rückgabe" in customer_query.lower() or "devolución" in customer_query.lower() or "retour" in customer_query.lower():
        internal_thought += "This is about return policy. Provide general return information."
        simulated_response = {
            "es": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra.",
            "en": "Our return policy allows returns within 30 days of purchase.",
            "de": "Unsere Rückgaberichtlinie erlaubt Rücksendungen innerhalb von 30 Tagen nach dem Kauf.",
            "fr": "Notre politique de retour autorise les retours dans les 30 jours suivant l'achat."
        }.get(customer_language_code, "Our return policy allows returns within 30 days of purchase.")
    elif "product" in customer_query.lower() or "producto" in customer_query.lower() or "produkt" in customer_query.lower() or "produit" in customer_query.lower() or "color" in customer_query.lower() or "farbe" in customer_query.lower() or "couleur" in customer_query.lower():
        internal_thought += "This is about product details/variations. Provide available options."
        simulated_response = {
            "es": "Sí, este producto está disponible en rojo, azul y verde. ¿Qué color prefiere?",
            "en": "Yes, this product is available in red, blue, and green. Which color do you prefer?",
            "de": "Ja, dieses Produkt ist in Rot, Blau und Grün erhältlich. Welche Farbe bevorzugen Sie?",
            "fr": "Oui, ce produit est disponible en rouge, bleu et vert. Quelle couleur préférez-vous ?"
        }.get(customer_language_code, "Yes, this product is available in various colors.")
    else:
        internal_thought += "General query. Attempting to provide a helpful, polite response."
        simulated_response = {
            "es": "Gracias por contactarnos. ¿En qué más puedo ayudarle?",
            "en": "Thank you for contacting us. How else may I assist you?",
            "de": "Vielen Dank für Ihre Kontaktaufnahme. Wie kann ich Ihnen sonst noch helfen?",
            "fr": "Merci de nous avoir contactés. Comment puis-je vous aider davantage ?"
        }.get(customer_language_code, "Thank you for contacting us. How else may I assist you?")

    # Append the simulated internal thought and response to the prompt's final parts
    lines = full_prompt.split("\n")
    output_lines = []
    for line in lines:
        output_lines.append(line)
        if line.strip().startswith("Internal Thought (Cross-lingual understanding via English):"):
            output_lines.append(f" {internal_thought}")
        elif line.strip().startswith(f"Response ({customer_language_code.upper()}):"):
            output_lines.append(f" {simulated_response}")
    
    return "\n".join(output_lines)

# 4. Main Chatbot Interaction (Demo)
# This function simulates a user interacting with the chatbot, demonstrating
# how the InCLT prompt is generated and processed by the simulated LLM.
def run_chatbot_demo():
    print("Welcome to the Multilingual E-commerce Chatbot Demo (InCLT Prompting)")
    print("Type 'exit' to end the conversation.")
    print("Supported languages for demo: Spanish (es), German (de), French (fr), English (en)")

    while True:
        customer_input = input("\nCustomer (e.g., 'es: ¿Dónde está mi pedido?', 'en: What is your return policy?'): ")
        if customer_input.lower() == 'exit':
            print("Exiting chatbot demo. Goodbye!")
            break

        # Parse customer language and query from input
        if ":" in customer_input:
            parts = customer_input.split(":", 1)
            if len(parts) == 2:
                lang_code, query = parts
                customer_language = lang_code.strip().lower()
                customer_query = query.strip()
            else:
                print("Invalid input format. Please use 'lang_code: query'.")
                continue
        else:
            print("Please specify language, e.g., 'es: ¿Dónde está mi pedido?'.")
            continue

        if customer_language not in [lang for ex in CROSS_LINGUAL_EXAMPLES for lang in ex['customer_query']] + ['en']:
            print(f"Unsupported language for this demo: {customer_language}. Please use es, de, fr, or en.")
            continue

        print(f"\nGenerating InCLT prompt for query: \"{customer_query}\" in {customer_language.upper()}...")
        prompt = generate_inclt_prompt(customer_query, customer_language, CROSS_LINGUAL_EXAMPLES)
        
        print("\n--- Generated Prompt (sent to LLM) ---")
        print(prompt)
        print("---------------------------------------")

        print("\n--- Simulating LLM Response ---")
        simulated_full_response = simulate_llm_response(prompt)
        print(simulated_full_response)
        print("-------------------------------")

        # Extract just the final chatbot response for the user to see
        response_prefix = f"Response ({customer_language.upper()}): "
        response_lines = [line for line in simulated_full_response.split("\n") if line.strip().startswith(response_prefix)]
        if response_lines:
            final_bot_response = response_lines[-1].replace(response_prefix, "").strip()
            print(f"\nChatbot ({customer_language.upper()}): {final_bot_response}")
        else:
            print("\nChatbot: Could not parse a specific response from the simulation.")

# Entry point to run the chatbot demo
if __name__ == "__main__":
    run_chatbot_demo()