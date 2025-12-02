import streamlit as st

# 4. In-Context Example Data Store
in_context_examples = [
    {
        "source_lang": "en",
        "source_query": "My order is late.",
        "target_lang": "es",
        "target_query": "Mi pedido está retrasado.",
        "source_answer": "We apologize for the delay. Please provide your order number and we will investigate.",
        "target_answer": "Lamentamos la demora. Por favor, proporcione su número de pedido y lo investigaremos."
    },
    {
        "source_lang": "es",
        "source_query": "¿Cómo puedo cambiar mi dirección de envío?",
        "target_lang": "en",
        "target_query": "How can I change my shipping address?",
        "source_answer": "Puede cambiar su dirección de envío en la sección 'Mi Cuenta' o contactando a nuestro soporte.",
        "target_answer": "You can change your shipping address in the 'My Account' section or by contacting our support."
    },
    {
        "source_lang": "en",
        "source_query": "I need a refund.",
        "target_lang": "fr",
        "target_query": "J'ai besoin d'un remboursement.",
        "source_answer": "Please provide your order details to process your refund request.",
        "target_answer": "Veuillez fournir les détails de votre commande pour traiter votre demande de remboursement."
    },
    {
        "source_lang": "fr",
        "source_query": "Mon produit est défectueux.",
        "target_lang": "en",
        "target_query": "My product is defective.",
        "source_answer": "Nous sommes désolés d'apprendre cela. Veuillez décrire le défaut et nous vous aiderons.",
        "target_answer": "We are sorry to hear that. Please describe the defect and we will assist you."
    },
]

# 2. Language Detection Module (Placeholder)
def detect_language(text: str) -> str:
    # In a real application, use a library like 'langdetect' or a dedicated NLP service
    # For demonstration, we'll make a simple guess or default
    text_lower = text.lower()
    if "order" in text_lower or "late" in text_lower or "refund" in text_lower:
        return "en"
    elif "pedido" in text_lower or "retrasado" in text_lower or "cambiar" in text_lower:
        return "es"
    elif "produit" in text_lower or "défectueux" in text_lower or "remboursement" in text_lower:
        return "fr"
    return "en" # Default to English

# 3. InCLT Prompt Constructor
def construct_inclt_prompt(
    user_query: str,
    source_lang: str,
    target_lang: str,
    examples: list
) -> str:
    prompt_parts = ["You are a multilingual customer support assistant. Answer the user's question concisely."]
    prompt_parts.append("\\nHere are some examples of customer queries and their answers in both source and target languages to help you:")

    for example in examples:
        # Select examples where either source_lang or target_lang matches the current interaction's languages
        if (example["source_lang"] == source_lang and example["target_lang"] == target_lang) or \
           (example["source_lang"] == target_lang and example["target_lang"] == source_lang) or \
           (example["source_lang"] == source_lang and example["target_lang"] == source_lang) or \
           (example["source_lang"] == target_lang and example["target_lang"] == target_lang):

            # Present in-context example in both source and target languages
            prompt_parts.append(f"\\n---\\nSource ({example["source_lang"].upper()}) Query: {example["source_query"]}")
            prompt_parts.append(f"Source ({example["source_lang"].upper()}) Answer: {example["source_answer"]}")
            prompt_parts.append(f"Target ({example["target_lang"].upper()}) Query: {example["target_query"]}")
            prompt_parts.append(f"Target ({example["target_lang"].upper()}) Answer: {example["target_answer"]}")

    prompt_parts.append(f"\\n---\\nUser Query ({source_lang.upper()}): {user_query}")
    prompt_parts.append(f"Assistant Response ({target_lang.upper()}):")

    return "\n".join(prompt_parts)

# 5. LLM Interaction Layer (Simulated)
def simulated_llm_response(prompt: str) -> str:
    # This is a highly simplified simulation. A real LLM would parse the prompt
    # and generate a contextually relevant response.
    # For demonstration, we'll try to extract the user query and provide a generic response.

    if "User Query (EN):" in prompt:
        query_start = prompt.rfind("User Query (EN):") + len("User Query (EN):")
        query_end = prompt.find("\nAssistant Response (EN):")
        user_query_en = prompt[query_start:query_end].strip()
        if "order is late" in user_query_en.lower():
            return "We are checking the status of your order. Please provide your order ID."
        elif "refund" in user_query_en.lower():
            return "To process a refund, we need your order number and reason for return."
        return f"Thank you for your inquiry in English. We are processing your request related to: '{user_query_en}'."

    elif "User Query (ES):" in prompt:
        query_start = prompt.rfind("User Query (ES):") + len("User Query (ES):")
        query_end = prompt.find("\nAssistant Response (ES):")
        user_query_es = prompt[query_start:query_end].strip()
        if "pedido retrasado" in user_query_es.lower():
            return "Estamos verificando el estado de su pedido. Por favor, proporcione su número de pedido."
        elif "cambiar dirección" in user_query_es.lower():
            return "Puede actualizar su dirección en la configuración de su cuenta."
        return f"Gracias por su consulta en español. Estamos procesando su solicitud relacionada con: '{user_query_es}'."

    elif "User Query (FR):" in prompt:
        query_start = prompt.rfind("User Query (FR):") + len("User Query (FR):")
        query_end = prompt.find("\nAssistant Response (FR):")
        user_query_fr = prompt[query_start:query_end].strip()
        if "produit défectueux" in user_query_fr.lower():
            return "Nous sommes désolés. Veuillez nous donner plus de détails sur le défaut."
        elif "remboursement" in user_query_fr.lower():
            return "Pour le remboursement, veuillez nous fournir les détails de votre commande."
        return f"Merci pour votre demande en français. Nous traitons votre requête concernant : '{user_query_fr}'."

    return "I am a simulated LLM and cannot fully understand your query based on the prompt structure. Please rephrase."


# 1. User Interface (UI) - Streamlit
st.set_page_config(page_title="Multilingual Customer Support Chatbot")
st.title("🌍 Multilingual Customer Support Chatbot")
st.markdown(" leverages InCLT Crosslingual Transfer Prompting")

user_input = st.text_area("Enter your query here:", height=100)

available_target_languages = {"English": "en", "Spanish": "es", "French": "fr"}
selected_target_lang_name = st.selectbox(
    "Select desired response language:",
    list(available_target_languages.keys())
)
target_lang_code = available_target_languages[selected_target_lang_name]

if st.button("Get Support"):
    if user_input:
        st.info("Processing your request...")

        # Detect source language
        source_lang_code = detect_language(user_input)
        st.write(f"Detected source language: **{source_lang_code.upper()}**")

        # Construct InCLT prompt
        full_prompt = construct_inclt_prompt(
            user_input,
            source_lang_code,
            target_lang_code,
            in_context_examples
        )

        st.subheader("Generated Prompt for LLM (for debugging/illustration):")
        st.code(full_prompt, language="python")

        # Get simulated LLM response
        llm_response = simulated_llm_response(full_prompt)

        st.subheader(f"Assistant Response ({target_lang_code.upper()}):")
        st.success(llm_response)
    else:
        st.warning("Please enter your query to get support.")
