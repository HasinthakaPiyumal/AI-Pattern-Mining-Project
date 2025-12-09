
import gradio as gr
import random

# --- Configuration ---
TARGET_LANGUAGE = "en"  # Primary language for internal processing and knowledge base

# --- Mock Services (In a real app, these would be external APIs or sophisticated models) ---

def mock_translate(text: str, src_lang: str, dest_lang: str) -> str:
    """
    A mock translation function. In a real application, this would use a library
    like googletrans, deep_translator, or an API like Google Cloud Translation, DeepL.
    For demonstration, it simply prefixes the text with language codes.
    """
    print(f"Mock translating '{text}' from {src_lang} to {dest_lang}")
    if src_lang == dest_lang:
        return text
    
    # Simulate some basic translation behavior for common languages
    if src_lang == "fr" and dest_lang == "en":
        if "bonjour" in text.lower(): return "hello, how can I help you?"
        if "problème avec ma commande" in text.lower(): return "I have a problem with my order."
        if "où est mon colis" in text.lower(): return "where is my package?"
        if "remboursement" in text.lower(): return "I want a refund."
    elif src_lang == "es" and dest_lang == "en":
        if "hola" in text.lower(): return "hello, how can I help you?"
        if "problema con mi pedido" in text.lower(): return "I have a problem with my order."
        if "dónde está mi paquete" in text.lower(): return "where is my package?"
        if "reembolso" in text.lower(): return "I want a refund."
    elif src_lang == "en" and dest_lang == "fr":
        if "hello" in text.lower(): return "Bonjour, comment puis-je vous aider ?"
        if "order problem" in text.lower(): return "Problème de commande."
        if "where is my package" in text.lower(): return "Où est mon colis ?"
        if "refund" in text.lower(): return "Remboursement."
    elif src_lang == "en" and dest_lang == "es":
        if "hello" in text.lower(): return "Hola, ¿cómo puedo ayudarte?"
        if "order problem" in text.lower(): return "Problema con mi pedido."
        if "where is my package" in text.lower(): return "Dónde está mi paquete?"
        if "refund" in text.lower(): return "Reembolso."

    return f"[{dest_lang.upper()}_TRANSLATION_OF_{text}]"

def mock_llm_response(prompt: str) -> str:
    """
    A mock LLM that generates a response based on keywords.
    In a real application, this would involve calling a hosted LLM API
    (e.g., OpenAI, Google Gemini, Hugging Face transformers model).
    """
    print(f"Mock LLM processing prompt:\n---\n{prompt}\n---")
    
    # Simple keyword-based response simulation
    if "where is my package" in prompt.lower() or "track my order" in prompt.lower():
        return "Your package is currently in transit and is expected to arrive within 2-3 business days. You can track it using the tracking number [XYZ123]."
    elif "refund" in prompt.lower() or "return" in prompt.lower():
        return "To initiate a refund or return, please visit our returns portal on the website and follow the instructions. You will need your order number."
    elif "product information" in prompt.lower() or "details about" in prompt.lower():
        return "Could you please specify which product you are interested in? I can provide details on features, specifications, and availability."
    elif "contact support" in prompt.lower() or "speak to a human" in prompt.lower():
        return "I understand. You can connect with a human agent by calling our hotline at 1-800-555-0123 during business hours, or by using the live chat option on our website."
    elif "hello" in prompt.lower() or "hi" in prompt.lower():
        return "Hello! How can I assist you with your e-commerce needs today?"
    else:
        return "I'm sorry, I couldn't fully understand your request. Could you please rephrase it or provide more details? Our support team can also be reached for complex issues."

# --- Knowledge Base (Simplified) ---
# In a real system, this would be a vector database or a more complex retrieval system.
KNOWLEDGE_BASE = {
    "package tracking": {
        "en": "Your package is currently in transit and is expected to arrive within 2-3 business days. You can track it using the tracking number [XYZ123].",
        "fr": "Votre colis est actuellement en transit et devrait arriver dans les 2-3 jours ouvrables. Vous pouvez le suivre en utilisant le numéro de suivi [XYZ123].",
        "es": "Su paquete está actualmente en tránsito y se espera que llegue dentro de 2-3 días hábiles. Puede rastrearlo usando el número de seguimiento [XYZ123]."
    },
    "refund policy": {
        "en": "To initiate a refund or return, please visit our returns portal on the website and follow the instructions. You will need your order number.",
        "fr": "Pour initier un remboursement ou un retour, veuillez visiter notre portail de retours sur le site web et suivre les instructions. Vous aurez besoin de votre numéro de commande.",
        "es": "Para iniciar un reembolso o una devolución, visite nuestro portal de devoluciones en el sitio web y siga las instrucciones. Necesitará su número de pedido."
    },
    "contact human support": {
        "en": "You can connect with a human agent by calling our hotline at 1-800-555-0123 during business hours, or by using the live chat option on our website.",
        "fr": "Vous pouvez contacter un agent humain en appelant notre service client au 1-800-555-0123 pendant les heures ouvrables, ou en utilisant l'option de chat en direct sur notre site web.",
        "es": "Puede contactar a un agente humano llamando a nuestra línea directa al 1-800-555-0123 durante el horario comercial, o utilizando la opción de chat en vivo en nuestro sitio web."
    }
}

# --- InCLT Prompting Logic ---

def generate_inclt_prompt(user_query_src: str, user_query_tgt: str, source_lang: str, examples: list) -> str:
    """
    Generates the InCLT prompt by combining source and target language examples
    with the current user query.

    Args:
        user_query_src: The user's original query in the source language.
        user_query_tgt: The user's query translated into the target language.
        source_lang: The original source language of the user's query.
        examples: A list of tuples, where each tuple is
                  (src_example_query, tgt_example_query, tgt_example_response, src_example_response).

    Returns:
        A formatted prompt string for the LLM.
    """
    prompt_parts = [
        "You are a multilingual e-commerce customer support assistant. ",
        f"The user's query is in {source_lang}. Provide a helpful response in {source_lang}. ",
        f"I will provide examples in both {source_lang} and {TARGET_LANGUAGE} to guide your understanding.\n"
    ]

    for src_ex_q, tgt_ex_q, tgt_ex_r, src_ex_r in examples:
        prompt_parts.append(f"\n--- Example ---")
        prompt_parts.append(f"Source Query ({src_lang}): {src_ex_q}")
        prompt_parts.append(f"Target Query ({TARGET_LANGUAGE}): {tgt_ex_q}")
        prompt_parts.append(f"Target Response ({TARGET_LANGUAGE}): {tgt_ex_r}")
        prompt_parts.append(f"Source Response ({src_lang}): {src_ex_r}")
        prompt_parts.append(f"--- End Example ---")

    prompt_parts.append(f"\n--- Current Query ---")
    prompt_parts.append(f"Source Query ({source_lang}): {user_query_src}")
    prompt_parts.append(f"Target Query ({TARGET_LANGUAGE}): {user_query_tgt}")
    prompt_parts.append(f"Target Response ({TARGET_LANGUAGE}): ") # LLM will complete this
    
    return "\n".join(prompt_parts)

# --- Main Application Logic ---

def process_customer_query(user_query: str, source_language_code: str) -> str:
    """
    Processes a customer query using the InCLT Crosslingual Transfer Prompting pattern.
    """
    if not user_query.strip():
        return "Please enter a query."

    print(f"\n--- Processing new query ---")
    print(f"User Query (Source: {source_language_code}): {user_query}")

    # 1. Translate the user query from source_language_code to TARGET_LANGUAGE
    user_query_translated_to_target = mock_translate(user_query, source_language_code, TARGET_LANGUAGE)
    print(f"User Query (Target: {TARGET_LANGUAGE}): {user_query_translated_to_target}")

    # 2. Prepare In-Context Learning Examples
    # Create a pool of example structures
    example_pool = [
        ("Où est mon colis ?", "Where is my package?", KNOWLEDGE_BASE["package tracking"]["en"], KNOWLEDGE_BASE["package tracking"]["fr"]),
        ("J'ai un problème avec ma commande.", "I have a problem with my order.", "Please provide your order number and a description of the issue.", "Veuillez fournir votre numéro de commande et une description du problème."),
        ("Quels sont les détails du produit X?", "What are the details of product X?", "Product X features a 12MP camera, 6-inch OLED display, and 256GB storage.", "Le produit X dispose d'un appareil photo de 12MP, d'un écran OLED de 6 pouces et de 256 Go de stockage."),
        ("Quiero un reembolso.", "I want a refund.", KNOWLEDGE_BASE["refund policy"]["en"], KNOWLEDGE_BASE["refund policy"]["es"]),
        ("Necesito hablar con un agente.", "I need to speak to an agent.", KNOWLEDGE_BASE["contact human support"]["en"], KNOWLEDGE_BASE["contact human support"]["es"]),
        ("¿Dónde está mi paquete?", "Where is my package?", KNOWLEDGE_BASE["package tracking"]["en"], KNOWLEDGE_BASE["package tracking"]["es"]),
    ]

    # Dynamically pick a few relevant examples (e.g., 2-3)
    selected_examples = random.sample(example_pool, k=min(3, len(example_pool)))
    
    # 3. Generate the InCLT prompt for the LLM
    llm_prompt = generate_inclt_prompt(user_query, user_query_translated_to_target, source_language_code, selected_examples)

    # 4. Get response from the Multilingual LLM (in TARGET_LANGUAGE)
    llm_raw_response_target_lang = mock_llm_response(llm_prompt)
    
    # 5. Translate the LLM's response back to the original source_language_code
    final_response_src_lang = mock_translate(llm_raw_response_target_lang, TARGET_LANGUAGE, source_language_code)

    print(f"Final Response (Source: {source_language_code}): {final_response_src_lang}")
    print(f"--- Query processing complete ---\n")

    return final_response_src_lang

# --- Gradio Interface ---

iface = gr.Interface(
    fn=process_customer_query,
    inputs=[
        gr.Textbox(label="Customer Query (e.g., in French, Spanish, or English)", placeholder="e.g., Où est mon colis ? / ¿Cómo puedo obtener un reembolso? / Where is my package?"),
        gr.Dropdown(label="Source Language", choices=["fr", "es", "en"], value="fr")
    ],
    outputs="text",
    title="Multilingual E-commerce Customer Support Assistant (InCLT Prompting)",
    description=(
        "This assistant uses 'InCLT Crosslingual Transfer Prompting' to boost cross-lingual capabilities. "
        "Type your query in French (fr), Spanish (es), or English (en) and select the source language."
    ),
    examples=[
        ["Où est mon colis ?", "fr"],
        ["J'ai un problème avec ma commande.", "fr"],
        ["¿Cómo puedo obtener un reembolso?", "es"],
        ["Necesito hablar con un agente.", "es"],
        ["Where is my package?", "en"],
        ["I want to return a product.", "en"],
    ]
)

if __name__ == "__main__":
    iface.launch()
