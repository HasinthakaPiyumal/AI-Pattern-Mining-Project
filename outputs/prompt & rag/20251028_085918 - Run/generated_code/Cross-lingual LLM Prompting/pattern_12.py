# Simulated In-Context Examples for InCLT (In-Context Learning Transfer)
# These examples demonstrate pairs of customer queries and ideal responses
# in both a source and a target language, as per the InCLT pattern.
# In a real application, these would be retrieved from a curated dataset.
IN_CONTEXT_EXAMPLES = [
    {
        "source_lang": "en",
        "source_query": "My order hasn't arrived yet.",
        "source_response": "Please provide your order number so I can check its status.",
        "target_lang": "es",
        "target_query": "¿Mi pedido no ha llegado todavía?",
        "target_response": "Por favor, proporcione su número de pedido para que pueda verificar su estado."
    },
    {
        "source_lang": "en",
        "source_query": "How do I return an item?",
        "source_response": "You can initiate a return through your account's 'Order History' section.",
        "target_lang": "fr",
        "target_query": "Comment retourner un article ?",
        "target_response": "Vous pouvez initier un retour via la section 'Historique des commandes' de votre compte."
    },
    {
        "source_lang": "es",
        "source_query": "Tengo un problema con la facturación.",
        "source_response": "Necesitaré su número de factura para investigar esto.",
        "target_lang": "en",
        "target_query": "I have a billing issue.",
        "target_response": "I will need your invoice number to look into this."
    },
    {
        "source_lang": "fr",
        "source_query": "Je voudrais changer mon adresse de livraison.",
        "source_response": "Veuillez vous connecter à votre compte et mettre à jour votre adresse dans les paramètres du profil.",
        "target_lang": "en",
        "target_query": "I would like to change my delivery address.",
        "target_response": "Please log in to your account and update your address in the profile settings."
    }
]

def construct_inclt_prompt(user_query: str, user_lang: str) -> str:
    """
    Constructs a prompt for a multilingual LLM using the InCLT (In-Context Learning Transfer)
    pattern. It includes the user's query and a set of cross-lingual in-context examples.

    Args:
        user_query: The customer's query in their original language.
        user_lang: The ISO 639-1 code for the user's language (e.g., "en", "es", "fr").

    Returns:
        A string representing the constructed prompt for the LLM.
    """
    prompt_parts = [
        f"You are a helpful customer support assistant. The user's query is in {user_lang}.",
        "Here are some examples of customer queries and ideal responses in various languages to help you understand and generate a good response:",
        ""
    ]

    # Add the cross-lingual in-context examples to the prompt
    for example in IN_CONTEXT_EXAMPLES:
        prompt_parts.append(f"Example ({example['source_lang']}): Q: \"{example['source_query']}\" A: \"{example['source_response']}\"")
        prompt_parts.append(f"Example ({example['target_lang']}): Q: \"{example['target_query']}\" A: \"{example['target_response']}\"")
        prompt_parts.append("###") # Separator for examples

    prompt_parts.append(f"Now, please provide an ideal, polite, and helpful response in {user_lang} to the following user query:")
    prompt_parts.append(f"User Query ({user_lang}): \"{user_query}\"")
    prompt_parts.append(f"Chatbot Response ({user_lang}):")
    
    return "\n".join(prompt_parts)

def simulate_multilingual_llm_response(prompt: str, user_lang: str) -> str:
    """
    A highly simplified simulation of a multilingual LLM's response generation.
    In a real-world application, this function would make an API call to an actual
    multilingual Large Language Model (e.g., Google's Gemini, OpenAI's GPT-4).

    For demonstration, it provides a rule-based response based on keywords and user_lang.

    Args:
        prompt: The constructed prompt containing the user query and InCLT examples.
        user_lang: The expected language of the response.

    Returns:
        A simulated response from the chatbot in the specified user_lang.
    """
    # Lowercasing the prompt for simpler keyword matching
    lower_prompt = prompt.lower()

    # Simulate responses based on common keywords, considering the target language
    if "order" in lower_prompt or "pedido" in lower_prompt or "commande" in lower_prompt:
        if user_lang == "en":
            return "To check your order status, please provide your order ID." 
        elif user_lang == "es":
            return "Para verificar el estado de su pedido, por favor proporcione su número de pedido."
        elif user_lang == "fr":
            return "Pour vérifier le statut de votre commande, veuillez fournir votre numéro de commande."

    if "return" in lower_prompt or "devolver" in lower_prompt or "retourner" in lower_prompt:
        if user_lang == "en":
            return "You can find our return policy and initiate a return in the 'Order History' section of your account." 
        elif user_lang == "es":
            return "Puede encontrar nuestra política de devolución e iniciar una devolución en la sección 'Historial de Pedidos' de su cuenta."
        elif user_lang == "fr":
            return "Vous pouvez consulter notre politique de retour et initier un retour dans la section 'Historique des commandes' de votre compte."

    if "billing" in lower_prompt or "facturación" in lower_prompt or "facture" in lower_prompt:
        if user_lang == "en":
            return "For billing inquiries, please provide your invoice number or account details." 
        elif user_lang == "es":
            return "Para consultas de facturación, por favor proporcione su número de factura o detalles de cuenta."
        elif user_lang == "fr":
            return "Pour les demandes de facturation, veuillez fournir votre numéro de facture ou les détails de votre compte."
            
    # Generic fallback response if no specific keyword is matched
    if user_lang == "en":
        return "Thank you for reaching out. How may I assist you further today?"
    elif user_lang == "es":
        return "Gracias por contactarnos. ¿En qué más puedo ayudarle hoy?"
    elif user_lang == "fr":
        return "Merci de nous avoir contactés. Comment puis-je vous aider davantage aujourd'hui ?"
    else:
        return "Thank you for your inquiry. We will get back to you shortly."

def chatbot_interact(user_query: str, user_lang: str) -> str:
    """
    Simulates a complete chatbot interaction using the InCLT prompting strategy.

    Args:
        user_query: The customer's query.
        user_lang: The language of the customer's query.

    Returns:
        The simulated chatbot's response.
    """
    print(f"\n--- User ({user_lang}): {user_query} ---")
    
    # 1. Construct the prompt with InCLT examples
    prompt = construct_inclt_prompt(user_query, user_lang)
    
    # For demonstration, print a snippet of the constructed prompt
    print("\n--- Constructed Prompt for LLM (abbreviated for display) ---")
    print(prompt[:600] + "...\n" if len(prompt) > 600 else prompt + "\n") 
    
    # 2. Simulate the LLM generating a response based on the prompt
    llm_response = simulate_multilingual_llm_response(prompt, user_lang)
    
    print(f"--- Chatbot ({user_lang}): {llm_response} ---")
    return llm_response

# --- Example Usage ---
if __name__ == "__main__":
    print("\n--- Starting Multilingual Customer Support Chatbot Simulation ---\n")

    # Example 1: English Query
    chatbot_interact("My delivery is taking too long, what's my order status?", "en")

    # Example 2: Spanish Query
    chatbot_interact("Tengo un problema con el producto que recibí, ¿cómo lo devuelvo?", "es")

    # Example 3: French Query
    chatbot_interact("J'ai une question sur ma dernière facture, où puis-je la trouver ?", "fr")

    # Example 4: English Query (another type)
    chatbot_interact("I need to update my shipping address.", "en")

    # Example 5: Spanish Query (generic)
    chatbot_interact("¿Podrían ayudarme con algo más?", "es")

    print("\n--- End of Simulation ---\n")