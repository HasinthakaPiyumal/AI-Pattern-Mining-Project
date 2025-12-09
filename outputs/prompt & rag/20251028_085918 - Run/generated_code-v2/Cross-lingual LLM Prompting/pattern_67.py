# Simulate FAQ knowledge base
ENGLISH_FAQ_DATABASE = {
    "shipping_cost": "What are the shipping costs?",
    "return_policy": "What is your return policy?",
    "payment_methods": "What payment methods do you accept?",
    "order_tracking": "How can I track my order?",
    "product_warranty": "Do products come with a warranty?",
    "customer_service": "How do I contact customer service?",
}

# Simulate a mechanism to retrieve relevant English FAQs based on a query in any language
# In a real system, this would involve embedding both query and FAQs and using a vector store (e.g., FAISS).
def simulate_faq_retrieval(query_lang, query_text):
    """
    Simulates retrieving relevant English FAQs for a given query.
    In a real scenario, a multilingual embedding model and a vector store would be used.
    For this simulation, we'll use simple keyword matching for illustration.
    """
    retrieved_faqs = []
    query_text_lower = query_text.lower()

    if "envío" in query_text_lower or "shipping" in query_text_lower:
        retrieved_faqs.append(ENGLISH_FAQ_DATABASE["shipping_cost"])
    if "devolución" in query_text_lower or "return" in query_text_lower:
        retrieved_faqs.append(ENGLISH_FAQ_DATABASE["return_policy"])
    if "pago" in query_text_lower or "payment" in query_text_lower:
        retrieved_faqs.append(ENGLISH_FAQ_DATABASE["payment_methods"])
    if "rastrear" in query_text_lower or "track" in query_text_lower:
        retrieved_faqs.append(ENGLISH_FAQ_DATABASE["order_tracking"])
    
    # Ensure some FAQs are always retrieved for demonstration
    if not retrieved_faqs:
        retrieved_faqs.append(ENGLISH_FAQ_DATABASE["customer_service"])
    
    return list(set(retrieved_faqs)) # Remove duplicates

class InCLT_Prompt_Builder:
    """
    Builds prompts leveraging InCLT Crosslingual Transfer Prompting.
    This means including examples that show how a target language query maps
    to a source language (English) FAQ and then generates a target language response.
    """
    def __init__(self):
        # These are the hand-crafted cross-lingual in-context examples
        # demonstrating the pattern.
        self.icl_examples = [
            {
                "query_es": "¿Cuál es el costo de envío?",
                "faq_en": ENGLISH_FAQ_DATABASE["shipping_cost"],
                "answer_es": "El costo de envío varía según la ubicación y el tamaño del paquete. Puedes ver el costo exacto al finalizar tu compra."
            },
            {
                "query_fr": "Quelle est votre politique de retour ?",
                "faq_en": ENGLISH_FAQ_DATABASE["return_policy"],
                "answer_fr": "Notre politique de retour permet les retours dans les 30 jours suivant l'achat, à condition que l'article soit dans son état d'origine."
            },
            {
                "query_de": "Welche Zahlungsmethoden akzeptieren Sie?",
                "faq_en": ENGLISH_FAQ_DATABASE["payment_methods"],
                "answer_de": "Wir akzeptieren Visa, Mastercard, American Express und PayPal."
            },
        ]

    def build_prompt(self, customer_query_lang, customer_query_text, retrieved_english_faqs):
        """
        Constructs the full prompt for the multilingual LLM.
        """
        prompt_parts = []

        # Add In-Context Learning examples
        for example in self.icl_examples:
            # Dynamically select the query and answer based on the example's language
            query_key = f"query_{list(example.keys())[0].split('_')[1]}" # e.g., 'query_es'
            answer_key = f"answer_{list(example.keys())[0].split('_')[1]}" # e.g., 'answer_es'

            prompt_parts.append(f"Customer Query ({list(example.keys())[0].split('_')[1].upper()}): {example[query_key]}")
            prompt_parts.append(f"Relevant English FAQ: {example['faq_en']}")
            prompt_parts.append(f"Assistant Response ({list(example.keys())[0].split('_')[1].upper()}): {example[answer_key]}\n")

        # Add the actual customer query and retrieved FAQs
        prompt_parts.append(f"Customer Query ({customer_query_lang.upper()}): {customer_query_text}")
        prompt_parts.append(f"Relevant English FAQs: {', '.join(retrieved_english_faqs)}")
        prompt_parts.append(f"Assistant Response ({customer_query_lang.upper()}):")

        return "\n".join(prompt_parts)

def multilingual_llm_inference(prompt_text):
    """
    Simulates interaction with a multilingual LLM.
    In a real application, this would be an API call (e.g., OpenAI, Cohere)
    or an inference with a local `transformers` model.
    """
    print(f"\n--- Simulating LLM Inference with Prompt ---\n{prompt_text}\n-------------------------------------------\n")
    
    # Very basic, illustrative response generation based on prompt keywords
    if "shipping_cost" in prompt_text and "español" in prompt_text.lower():
        return "El costo de envío se calcula al finalizar la compra. Depende de la ubicación y el peso."
    elif "return_policy" in prompt_text and "français" in prompt_text.lower():
        return "Notre politique de retour vous permet de retourner les articles dans les 30 jours."
    elif "payment_methods" in prompt_text and "deutsch" in prompt_text.lower():
        return "Wir akzeptieren gängige Kreditkarten und PayPal."
    elif "order_tracking" in prompt_text and "rastrear" in prompt_text.lower():
        return "Puedes rastrear tu pedido usando el número de seguimiento proporcionado en tu correo electrónico."
    else:
        return "Lo siento, no pude encontrar una respuesta específica. Por favor, contacta a nuestro servicio al cliente para más ayuda."

def customer_support_assistant(query_lang, query_text):
    """
    Main function for the customer support assistant demonstrating InCLT.
    """
    # Step 1: Simulate FAQ retrieval
    retrieved_faqs = simulate_faq_retrieval(query_lang, query_text)

    # Step 2: Build the InCLT prompt
    prompt_builder = InCLT_Prompt_Builder()
    prompt = prompt_builder.build_prompt(query_lang, query_text, retrieved_faqs)

    # Step 3: Simulate LLM inference
    llm_response = multilingual_llm_inference(prompt)

    return llm_response

# Example Usage:
# user_query_es = "¿Cuánto cuesta el envío a Madrid?"
# print(f"Customer ({"ES".upper()}): {user_query_es}")
# assistant_response_es = customer_support_assistant("es", user_query_es)
# print(f"Assistant ({"ES".upper()}): {assistant_response_es}\n")

# user_query_fr = "Comment puis-je suivre ma commande ?"
# print(f"Customer ({"FR".upper()}): {user_query_fr}")
# assistant_response_fr = customer_support_assistant("fr", user_query_fr)
# print(f"Assistant ({"FR".upper()}): {assistant_response_fr}\n")

# user_query_de = "Gibt es eine Garantie für die Produkte?"
# print(f"Customer ({"DE".upper()}): {user_query_de}")
# assistant_response_de = customer_support_assistant("de", user_query_de)
# print(f"Assistant ({"DE".upper()}): {assistant_response_de}\n")

# user_query_en = "What are your payment options?"
# print(f"Customer ({"EN".upper()}): {user_query_en}")
# assistant_response_en = customer_support_assistant("en", user_query_en)
# print(f"Assistant ({"EN".upper()}): {assistant_response_en}\n")