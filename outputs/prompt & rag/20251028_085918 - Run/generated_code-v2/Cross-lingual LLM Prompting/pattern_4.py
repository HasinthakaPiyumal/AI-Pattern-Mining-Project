import os
# from openai import OpenAI # Uncomment for real OpenAI API usage

# --- Mock LLM Interaction (replace with actual API call) ---
# In a real application, you would replace MockLLM with an actual LLM client
# like OpenAI's client, Google Gemini client, or a Hugging Face transformers pipeline.
class MockLLM:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name

    def generate_response(self, prompt, temperature=0.7):
        """
        Simulates an LLM response based on the prompt.
        In a real scenario, this would call an actual LLM API (e.g., OpenAI, Google Gemini).
        """
        print(f"\n--- Mock LLM Receiving Prompt ({self.model_name}) ---")
        # print(prompt) # Uncomment to see the full prompt sent to the mock LLM
        print("Prompt excerpt: " + prompt[:200].replace('\n', ' ') + "...")
        print("--------------------------------------------------")

        # Basic keyword-based mock responses for demonstration
        if "garantía" in prompt.lower() and "producto x" in prompt.lower():
            return "El Producto X tiene una garantía limitada de 1 año."
        elif "devolver" in prompt.lower() and "artículo" in prompt.lower():
            return "Puede devolver los artículos en los 30 días siguientes a la compra con el recibo original."
        elif "shipping" in prompt.lower() and "product y" in prompt.lower() and "fr" in prompt.lower():
            return "L'expédition pour le produit Y prend généralement 5-7 jours ouvrables."
        elif "price" in prompt.lower() and "product z" in prompt.lower() and "de" in prompt.lower():
            return "Der Preis für Produkt Z beträgt 99,99 EUR."
        elif "faq" in prompt.lower() and "delivery time" in prompt.lower() and "es" in prompt.lower():
            return "FAQ: ¿Cuál es el tiempo de entrega típico para productos electrónicos? El tiempo de entrega típico es de 5 a 7 días hábiles."
        elif "faq" in prompt.lower() and "track my order" in prompt.lower() and "fr" in prompt.lower():
            return "FAQ: Comment puis-je suivre ma commande? Vous pouvez suivre votre commande en visitant la page 'Suivre ma commande' sur notre site web."
        else:
            return "Lo siento, no pude encontrar una respuesta específica. ¿Podría reformular su pregunta o proporcionar más detalles? (Mock Response)"

# Initialize mock LLM
llm = MockLLM()

# Uncomment and configure for real OpenAI API usage
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# class RealLLM:
#     def generate_response(self, prompt, temperature=0.7):
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=temperature,
#         )
#         return response.choices[0].message.content
# llm = RealLLM()

# --- Knowledge Base (Simulated) ---
# Represents source language (English) product information/FAQs.
# Pre-translated entries are used for creating effective InCLT examples.
knowledge_base = {
    "product_X_warranty": {
        "en_query": "What is the warranty for product X?",
        "en_answer": "Product X has a 1-year limited warranty.",
        "es_query": "¿Cuál es la garantía para el producto X?",
        "es_answer": "El Producto X tiene una garantía limitada de 1 año.",
    },
    "product_Y_shipping": {
        "en_query": "How long for shipping of Product Y?",
        "en_answer": "Shipping for Product Y usually takes 5-7 business days.",
        "fr_query": "Combien de temps prend l'expédition du Produit Y ?",
        "fr_answer": "L'expédition pour le Produit Y prend généralement 5-7 jours ouvrables.",
    },
    "return_policy": {
        "en_query": "How do I return an item?",
        "en_answer": "You can return items within 30 days of purchase with the original receipt.",
        "es_query": "¿Cómo devuelvo un artículo?",
        "es_answer": "Puede devolver los artículos en los 30 días siguientes a la compra con el recibo original.",
        "fr_query": "Comment puis-je retourner un article ?",
        "fr_answer": "Vous pouvez retourner les articles dans les 30 jours suivant l'achat avec le reçu original.",
    }
}

# --- In-Context Learning Examples for Cross-lingual Transfer (InCLT) ---
# These examples are crucial for the InCLT pattern, leveraging both source (EN)
# and target languages to demonstrate desired cross-lingual transfer behavior.
icl_examples = [
    {
        "source_lang": "en",
        "target_lang": "es",
        "source_query": knowledge_base["product_X_warranty"]["en_query"],
        "source_answer": knowledge_base["product_X_warranty"]["en_answer"],
        "target_query": knowledge_base["product_X_warranty"]["es_query"],
        "target_answer": knowledge_base["product_X_warranty"]["es_answer"],
    },
    {
        "source_lang": "en",
        "target_lang": "fr",
        "source_query": knowledge_base["return_policy"]["en_query"],
        "source_answer": knowledge_base["return_policy"]["en_answer"],
        "target_query": knowledge_base["return_policy"]["fr_query"],
        "target_answer": knowledge_base["return_policy"]["fr_answer"],
    },
    {
        "source_lang": "en",
        "target_lang": "es",
        "source_query": knowledge_base["return_policy"]["en_query"],
        "source_answer": knowledge_base["return_policy"]["en_answer"],
        "target_query": knowledge_base["return_policy"]["es_query"],
        "target_answer": knowledge_base["return_policy"]["es_answer"],
    },
]

def generate_icl_prompt(customer_query: str, target_language: str, examples: list) -> str:
    """
    Constructs a prompt using the InCLT Crosslingual Transfer Prompting pattern.
    Includes system instruction, carefully selected in-context examples in both source and target languages,
    and the current customer query.
    """
    prompt_parts = []

    # 1. System Instruction: Defines the chatbot's role and emphasizes cross-lingual understanding.
    prompt_parts.append(
        "You are a multilingual customer support chatbot for a global e-commerce platform. "
        "Your task is to provide accurate, concise, and helpful responses to customer queries "
        "in the *requested target language*. "
        "Crucially, leverage the provided 'In-Context Learning Examples'. "
        "These examples demonstrate how to transfer information and answer patterns "
        "from a source language to a target language. "
        "Pay close attention to how concepts and answers are expressed across languages in the examples. "
        "Your final answer should only be the direct response to the customer's query in the target language."
    )
    prompt_parts.append("\n--- In-Context Learning Examples for Cross-lingual Transfer ---")

    # 2. In-Context Examples (InCLT): Iterates through provided examples to build the ICL part of the prompt.
    # These examples show the LLM how to translate/transfer knowledge.
    for ex in examples:
        # In a more advanced system, example selection would be dynamic and based on query similarity.
        # Here, we include relevant examples for the target language or general ones.
        prompt_parts.append(f"\nEXAMPLE {examples.index(ex) + 1}:")
        prompt_parts.append(f"  Source Language ({ex['source_lang'].upper()}):")
        prompt_parts.append(f"    Query: {ex['source_query']}")
        prompt_parts.append(f"    Answer: {ex['source_answer']}")
        prompt_parts.append(f"  Target Language ({ex['target_lang'].upper()}):")
        prompt_parts.append(f"    Query: {ex['target_query']}")
        prompt_parts.append(f"    Answer: {ex['target_answer']}") # This is key for InCLT: showing the desired output format and content transfer

    # 3. Current Query: The actual question the chatbot needs to answer.
    prompt_parts.append("\n--- Customer Query ---")
    prompt_parts.append(f"Customer Query ({target_language.upper()}): {customer_query}")
    prompt_parts.append(f"Provide the answer in {target_language.upper()}:")

    return "\n".join(prompt_parts)

def detect_language(text: str) -> str:
    """
    A simplified mock language detection. In a real system, use a robust library
    like 'langdetect', 'fasttext', or a dedicated NLP model.
    """
    text_lower = text.lower()
    if any(word in text_lower for word in ["garantía", "devolver", "español", "qué", "cómo", "el", "la"]):
        return "es"
    if any(word in text_lower for word in ["garantie", "retourner", "français", "comment", "le", "la"]):
        return "fr"
    if any(word in text_lower for word in ["garantie", "zurückgeben", "deutsch", "was", "wie", "der", "die", "das"]):
        return "de"
    return "en" # Default to English

def handle_customer_query(query: str) -> str:
    """
    Orchestrates the process of handling a customer query:
    1. Detects the query language.
    2. Selects relevant InCLT examples.
    3. Constructs the prompt using generate_icl_prompt.
    4. Calls the LLM to get a response.
    """
    detected_lang = detect_language(query)
    print(f"\nDetected language: {detected_lang.upper()}")

    # Filter InCLT examples to prioritize those matching the detected target language.
    # This helps guide the LLM more effectively.
    relevant_examples = [ex for ex in icl_examples if ex["target_lang"] == detected_lang]
    if not relevant_examples:
        print("No specific InCLT examples found for this target language, using all available.")
        relevant_examples = icl_examples # Fallback if no specific examples are pre-defined

    prompt = generate_icl_prompt(query, detected_lang, relevant_examples)
    response = llm.generate_response(prompt)
    return response

def generate_cross_lingual_faq_entry(source_faq_en: str, target_lang: str) -> str:
    """
    Demonstrates generating a cross-lingual FAQ entry using the InCLT principle.
    It takes an English FAQ (source) and generates its equivalent in the target language,
    leveraging the chatbot's ability for cross-lingual transfer learned from examples.
    """
    print(f"\n--- Generating Cross-lingual FAQ for '{source_faq_en}' in {target_lang.upper()} ---")

    # For FAQ generation, we provide specific examples of English FAQ -> Target Language FAQ
    faq_icl_examples = [
        {
            "source_lang": "en",
            "target_lang": "es",
            "source_text": "FAQ: How long does shipping typically take for electronics?",
            "target_text": "FAQ: ¿Cuál es el tiempo de entrega típico para productos electrónicos? El tiempo de entrega típico es de 5 a 7 días hábiles.",
        },
        {
            "source_lang": "en",
            "target_lang": "fr",
            "source_text": "FAQ: What is your return policy for damaged items?",
            "target_text": "FAQ: Quelle est votre politique de retour pour les articles endommagés ? Vous pouvez retourner les articles endommagés dans les 14 jours suivant la réception.",
        }
    ]

    prompt_parts = []
    prompt_parts.append(
        f"You are an AI assistant specialized in generating cross-lingual FAQ entries. "
        f"Translate and rephrase the given English FAQ into {target_lang.upper()} "
        f"while maintaining its meaning, adapting it to a standard FAQ format in the target language, "
        f"and providing a concise answer. Use the following examples for guidance on cross-lingual transfer."
    )
    prompt_parts.append("\n--- In-Context FAQ Examples ---")
    for ex in faq_icl_examples:
        if ex["target_lang"] == target_lang:
            prompt_parts.append(f"\nSource FAQ ({ex['source_lang'].upper()}): {ex['source_text']}")
            prompt_parts.append(f"Target FAQ ({ex['target_lang'].upper()}): {ex['target_text']}")

    prompt_parts.append("\n--- Your Task ---")
    prompt_parts.append(f"Source FAQ (EN): {source_faq_en}")
    prompt_parts.append(f"Generate Target FAQ ({target_lang.upper()}):")

    prompt = "\n".join(prompt_parts)
    generated_faq = llm.generate_response(prompt)
    return generated_faq


# --- Main execution flow ---
if __name__ == "__main__":
    print("Welcome to the Multilingual Customer Support Chatbot!")
    print("Type 'exit' to end the chat.")
    print("\nTry asking questions in Spanish, French, or English, for example:")
    print("  - '¿Cuál es la garantía para el producto X?' (Spanish)")
    print("  - 'Comment puis-je retourner un article ?' (French)")
    print("  - 'What is the shipping time for Product Y?' (English)")

    while True:
        user_input = input("\nYou (Type your query): ")
        if user_input.lower() == 'exit':
            break

        response = handle_customer_query(user_input)
        print(f"Chatbot: {response}")

    print("\n--- Demonstrating Cross-lingual FAQ Generation ---")
    faq_to_translate_en_1 = "What is the typical delivery time for electronics?"
    generated_faq_es = generate_cross_lingual_faq_entry(faq_to_translate_en_1, "es")
    print(f"\nGenerated Spanish FAQ:\n{generated_faq_es}")

    faq_to_translate_en_2 = "How can I track my order?"
    generated_faq_fr = generate_cross_lingual_faq_entry(faq_to_translate_en_2, "fr")
    print(f"\nGenerated French FAQ:\n{generated_faq_fr}")

    print("\nExiting chatbot. Goodbye!")