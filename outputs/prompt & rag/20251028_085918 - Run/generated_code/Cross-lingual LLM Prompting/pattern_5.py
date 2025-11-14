
from langdetect import detect, DetectorFactory

# Ensure consistent language detection results
DetectorFactory.seed = 0

# --- 1. Mock Multilingual LLM (Placeholder for a real Hugging Face model) ---
def mock_multilingual_llm(prompt: str, target_language: str) -> str:
    """
    A mock LLM function. In a real application, this would interact with a 
    Hugging Face Transformers model (e.g., mBERT, XLM-R, or fine-tuned T5/BART).
    It simulates generating a response in the target language based on the prompt.
    """
    print(f"\n--- Mock LLM Input Prompt ({target_language}) ---")
    print(prompt)
    print("---------------------------------------------------")

    # Simulate a basic understanding and response generation
    if "shipping" in prompt.lower() and "status" in prompt.lower():
        if target_language == "en":
            return "Your shipping status can be checked on our website using your order number."
        elif target_language == "es":
            return "Puede verificar el estado de su envío en nuestro sitio web utilizando su número de pedido."
        elif target_language == "fr":
            return "Vous pouvez vérifier le statut de votre expédition sur notre site web avec votre numéro de commande."
    elif "refund" in prompt.lower() and "policy" in prompt.lower():
        if target_language == "en":
            return "Our refund policy allows returns within 30 days of purchase with a valid receipt."
        elif target_language == "es":
            return "Nuestra política de reembolso permite devoluciones dentro de los 30 días posteriores a la compra con un recibo válido."
        elif target_language == "fr":
            return "Notre politique de remboursement autorise les retours dans les 30 jours suivant l'achat avec un reçu valide."
    elif "product" in prompt.lower() and "information" in prompt.lower():
        if target_language == "en":
            return "Please specify which product you are interested in for more detailed information."
        elif target_language == "es":
            return "Por favor, especifique qué producto le interesa para obtener información más detallada."
        elif target_language == "fr":
            return "Veuillez préciser le produit qui vous intéresse pour des informations plus détaillées."

    if target_language == "en":
        return "I am sorry, I couldn't find a specific answer to your query. Please rephrase or ask another question."
    elif target_language == "es":
        return "Lo siento, no pude encontrar una respuesta específica a su consulta. Por favor, reformule o haga otra pregunta."
    elif target_language == "fr":
        return "Je suis désolé, je n'ai pas pu trouver de réponse spécifique à votre question. Veuillez reformuler ou poser une autre question."


# --- 2. Language Detection Module ---
def detect_query_language(text: str) -> str:
    """
    Detects the language of the input text.
    """
    try:
        return detect(text)
    except Exception:
        return "en" # Default to English if detection fails


# --- 3. Mock Knowledge Base ---
# In a real system, this would be a vector database with embeddings 
# for cross-lingual semantic search.
KNOWLEDGE_BASE = {
    "shipping_status_en": {
        "language": "en",
        "question": "How do I check my shipping status?",
        "answer": "You can track your order using the tracking number provided in your shipping confirmation email on our website."
    },
    "shipping_status_es": {
        "language": "es",
        "question": "¿Cómo verifico el estado de mi envío?",
        "answer": "Puede rastrear su pedido utilizando el número de seguimiento proporcionado en su correo electrónico de confirmación de envío en nuestro sitio web."
    },
    "shipping_status_fr": {
        "language": "fr",
        "question": "Comment vérifier le statut de ma commande ?",
        "answer": "Vous pouvez suivre votre commande en utilisant le numéro de suivi fourni dans votre e-mail de confirmation d'expédition sur notre site web."
    },
    "refund_policy_en": {
        "language": "en",
        "question": "What is your refund policy?",
        "answer": "We offer a 30-day money-back guarantee for most products, provided they are in their original condition with a valid proof of purchase."
    },
    "refund_policy_es": {
        "language": "es",
        "question": "¿Cuál es su política de reembolso?",
        "answer": "Ofrecemos una garantía de devolución de dinero de 30 días para la mayoría de los productos, siempre que estén en su estado original con un comprobante de compra válido."
    },
    "refund_policy_fr": {
        "language": "fr",
        "question": "Quelle est votre politique de remboursement ?",
        "answer": "Nous offrons une garantie de remboursement de 30 jours pour la plupart des produits, à condition qu'ils soient dans leur état d'origine avec une preuve d'achat valide."
    },
    "product_info_en": {
        "language": "en",
        "question": "Tell me more about product X.",
        "answer": "Product X is a high-performance gadget designed for tech enthusiasts, featuring a long-lasting battery and intuitive user interface."
    },
    "product_info_es": {
        "language": "es",
        "question": "Háblame más sobre el producto X.",
        "answer": "El Producto X es un dispositivo de alto rendimiento diseñado para entusiastas de la tecnología, con una batería de larga duración y una interfaz de usuario intuitiva."
    },
    "product_info_fr": {
        "language": "fr",
        "question": "Parlez-moi davantage du produit X.",
        "answer": "Le Produit X est un gadget haute performance conçu pour les passionnés de technologie, doté d'une batterie longue durée et d'une interface utilisateur intuitive."
    },
}

def retrieve_relevant_info(query: str, query_language: str, target_language: str) -> str:
    """
    Simulates retrieving relevant information from the knowledge base.
    In a real scenario, this would involve semantic search in a vector DB.
    For this mock, it does a simple keyword match across languages.
    """
    relevant_articles = []
    query_lower = query.lower()

    # Attempt to find relevant articles in the query language first
    for key, article in KNOWLEDGE_BASE.items():
        if article["language"] == query_language and \
           (query_lower in article["question"].lower() or query_lower in article["answer"].lower()):
            relevant_articles.append(article["answer"])
            # Try to also get the corresponding article in the target language if different
            if query_language != target_language:
                target_key = key.replace(f"_{query_language}", f"_{target_language}")
                if target_key in KNOWLEDGE_BASE:
                    relevant_articles.append(KNOWLEDGE_BASE[target_key]["answer"])
            break # Found a direct match, stop looking for more for simplicity

    # If no direct match in query language, try to find in target language
    if not relevant_articles:
        for key, article in KNOWLEDGE_BASE.items():
            if article["language"] == target_language and \
               (query_lower in article["question"].lower() or query_lower in article["answer"].lower()):
                relevant_articles.append(article["answer"])
                break

    return " ".join(relevant_articles) if relevant_articles else ""


# --- 4. Prompt Engineering Module (with InCLT Crosslingual Transfer) ---

# In-context examples demonstrating cross-lingual transfer
IN_CONTEXT_EXAMPLES = [
    {
        "source_query": "Quel est le statut de ma commande ?",
        "source_language": "fr",
        "target_language_info": "You can track your order using the tracking number provided in your shipping confirmation email on our website.", # English info
        "target_response": "Vous pouvez suivre votre commande en utilisant le numéro de suivi fourni dans votre e-mail de confirmation d'expédition sur notre site web."
    },
    {
        "source_query": "¿Cuál es la política de devoluciones?",
        "source_language": "es",
        "target_language_info": "We offer a 30-day money-back guarantee for most products, provided they are in their original condition with a valid proof of purchase.", # English info
        "target_response": "Ofrecemos una garantía de devolución de dinero de 30 días para la mayoría de los productos, siempre que estén en su estado original con un comprobante de compra válido."
    },
    {
        "source_query": "How can I return an item?",
        "source_language": "en",
        "target_language_info": "Para devolver un artículo, visite nuestra página de devoluciones y siga las instrucciones. Necesitará su número de pedido y el correo electrónico utilizado para la compra.", # Spanish info
        "target_response": "To return an item, please visit our returns page and follow the instructions. You will need your order number and the email used for purchase."
    }
]

def construct_inclt_prompt(
    customer_query: str,
    query_language: str,
    relevant_knowledge: str,
    target_response_language: str,
    in_context_examples: list
) -> str:
    """
    Constructs a prompt using the InCLT Crosslingual Transfer Prompting pattern.
    This involves providing examples in both source and target languages.
    """
    prompt_parts = [
        "You are a helpful multilingual customer support assistant.",
        f"The customer is asking in {query_language}. Provide a concise answer in {target_response_language}."
    ]

    # Add in-context examples
    if in_context_examples:
        prompt_parts.append("\nHere are some examples of how to answer cross-lingual queries:")
        for example in in_context_examples:
            if example["source_language"] == query_language or example["target_response"] == target_response_language:
                prompt_parts.append(f"Customer ({example['source_language']}): {example['source_query']}")
                prompt_parts.append(f"Relevant Info (from KB, potentially {example['target_language_info']}): {example['target_language_info']}")
                prompt_parts.append(f"Assistant ({target_response_language}): {example['target_response']}\n")

    # Add the retrieved relevant knowledge
    if relevant_knowledge:
        prompt_parts.append(f"\nRelevant Information from Knowledge Base (in {target_response_language} or related): {relevant_knowledge}\n")

    # Add the actual customer query
    prompt_parts.append(f"Customer ({query_language}): {customer_query}")
    prompt_parts.append(f"Assistant ({target_response_language}):")

    return "\n".join(prompt_parts)


# --- 5. Chatbot Orchestration Logic ---
def customer_support_chatbot(customer_query: str) -> str:
    """
    Orchestrates the chatbot's response generation using InCLT prompting.
    """
    print(f"\nCustomer Query: '{customer_query}'")

    # 1. Detect query language
    query_language = detect_query_language(customer_query)
    print(f"Detected Language: {query_language}")

    # For simplicity, we'll assume the target response language is the same as the query language
    # In a real scenario, this might be a configurable setting or based on agent preference.
    target_response_language = query_language

    # 2. Retrieve relevant information from Knowledge Base
    # This mock retrieves info in the query_language and potentially the target_response_language
    relevant_info = retrieve_relevant_info(customer_query, query_language, target_response_language)
    print(f"Retrieved Info: {relevant_info[:100]}...") # Truncate for display

    # 3. Construct the InCLT prompt
    prompt = construct_inclt_prompt(
        customer_query,
        query_language,
        relevant_info,
        target_response_language,
        IN_CONTEXT_EXAMPLES
    )

    # 4. Send prompt to Multilingual LLM and get response
    llm_response = mock_multilingual_llm(prompt, target_response_language)

    print(f"\nChatbot Response ({target_response_language}): {llm_response}")
    return llm_response


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Starting Multilingual Customer Support Chatbot ---")

    # Example 1: English query
    customer_support_chatbot("What is the status of my shipment?")

    # Example 2: Spanish query
    customer_support_chatbot("¿Cuál es su política de reembolso?")

    # Example 3: French query (no direct KB match, relies more on LLM general knowledge and prompt structure)
    customer_support_chatbot("J'ai une question sur les retours.")

    # Example 4: English query with a slightly different phrasing
    customer_support_chatbot("I want to know about your return policy.")

    # Example 5: Spanish query with a specific product inquiry
    customer_support_chatbot("Háblame más sobre el producto X.")

    print("\n--- Chatbot Demonstration Finished ---")
