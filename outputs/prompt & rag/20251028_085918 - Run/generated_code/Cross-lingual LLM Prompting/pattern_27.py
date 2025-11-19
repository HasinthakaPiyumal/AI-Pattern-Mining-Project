import streamlit as st
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0 # for consistent language detection

# --- Knowledge Base (Simplified for demonstration) ---
# In a real application, this would be a more sophisticated database
# with proper indexing and retrieval, potentially using a vector DB.
KNOWLEDGE_BASE = {
    "en": {
        "How do I track my order?": "You can track your order using the link provided in your shipping confirmation email.",
        "What is your return policy?": "Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging.",
        "How can I contact customer support?": "You can contact customer support via email at support@example.com or call us at 1-800-123-4567."
    },
    "es": {
        "¿Cómo rastreo mi pedido?": "Puedes rastrear tu pedido usando el enlace proporcionado en tu correo electrónico de confirmación de envío.",
        "¿Cuál es su política de devoluciones?": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra, siempre que el artículo no haya sido utilizado y esté en su embalaje original.",
        "¿Cómo puedo contactar al servicio al cliente?": "Puede contactar al servicio al cliente por correo electrónico a support@example.com o llamarnos al 1-800-123-4567."
    }
}

# --- Language Detection Module ---
def detect_query_language(text):
    try:
        return detect(text)
    except:
        return "en" # Default to English if detection fails

# --- Prompt Engineering Module (InCLT Prompting) ---
def create_inclt_prompt(user_query, detected_lang, kb):
    # Basic example retrieval based on keywords (for simplicity)
    # In a real system, this would involve semantic search or embeddings.
    
    # Select a few relevant in-context examples
    in_context_examples = []
    if detected_lang == "es":
        # Example 1: English Q&A + Spanish Q&A
        in_context_examples.append(
            """English Q: How do I track my order?
English A: You can track your order using the link provided in your shipping confirmation email.
Spanish Q: ¿Cómo rastreo mi pedido?
Spanish A: Puedes rastrear tu pedido usando el enlace proporcionado en tu correo electrónico de confirmación de envío."""
        )
        # Example 2: Another pair
        in_context_examples.append(
            """English Q: What is your return policy?
English A: Our return policy allows returns within 30 days of purchase.
Spanish Q: ¿Cuál es su política de devoluciones?
Spanish A: Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra."""
        )
    else:
        # Default to English-only examples if the target language isn't Spanish or similar logic
        # Or, ideally, fetch relevant examples in target language if available
        for q_en, a_en in kb["en"].items():
            in_context_examples.append(f"English Q: {q_en}\nEnglish A: {a_en}")
            if len(in_context_examples) >= 2: break # Limit examples

    # Construct the full prompt
    prompt_template = """
As a helpful e-commerce customer support assistant, answer the following question. Use the provided examples to understand the context and generate a helpful response in the user's language.

Examples:
{examples}

User's Query ({lang}): {query}
Assistant's Response ({lang}):"""

    return prompt_template.format(
        examples="\n\n".join(in_context_examples),
        lang=detected_lang.upper(),
        query=user_query
    )

# --- Large Language Model (LLM) Integration (Placeholder) ---
def get_llm_response(prompt):
    # This is a placeholder for actual LLM API call (e.g., OpenAI, Gemini, Hugging Face)
    # In a real application, you would integrate with an LLM SDK here.
    
    # Simulate LLM behavior based on the prompt content and query
    if "¿Cómo rastreo mi pedido?" in prompt and "es" in prompt.lower():
        return "Puedes rastrear tu pedido visitando la sección 'Mis pedidos' en nuestro sitio web y haciendo clic en el enlace de seguimiento junto a tu compra." 
    elif "How do I track my order?" in prompt and "en" in prompt.lower():
        return "You can track your order by visiting the 'My Orders' section on our website and clicking the tracking link next to your purchase."
    elif "¿Cuál es su política de devoluciones?" in prompt and "es" in prompt.lower():
        return "Nuestra política de devoluciones es de 30 días. Para iniciar una devolución, visita la sección de devoluciones en nuestro sitio web."
    elif "What is your return policy?" in prompt and "en" in prompt.lower():
        return "Our return policy is 30 days. To initiate a return, please visit the returns section on our website."
    elif "¿Cómo puedo contactar al servicio al cliente?" in prompt and "es" in prompt.lower():
        return "Nuestro equipo de soporte está disponible por correo electrónico a support@example.com."
    elif "How can I contact customer support?" in prompt and "en" in prompt.lower():
        return "Our support team is available via email at support@example.com."
    else:
        return f"I understand you asked in {detect_query_language(prompt)}. As a multilingual assistant, I am processing your query.\n(Simulated LLM response for: '{user_query}')"

# --- Streamlit UI and Chatbot Orchestration Logic ---
st.set_page_config(page_title="Multilingual Customer Support Chatbot")
st.title("🌍 Multilingual Customer Support Chatbot")
st.markdown("Ask your questions in any supported language and get a smart response!")

user_query = st.text_input("Your Question:", "")

if st.button("Get Answer") and user_query:
    with st.spinner("Detecting language and generating response..."):
        # 1. Language Detection
        detected_lang = detect_query_language(user_query)
        st.info(f"Detected Language: **{detected_lang.upper()}**")

        # 2. Prompt Engineering (InCLT)
        inclt_prompt = create_inclt_prompt(user_query, detected_lang, KNOWLEDGE_BASE)
        # st.expander("View Generated Prompt").code(inclt_prompt) # For debugging/demonstration

        # 3. LLM Integration
        llm_response = get_llm_response(inclt_prompt)

        # 4. Display Response
        st.subheader("Chatbot Response:")
        st.write(llm_response)

elif st.button("Get Answer") and not user_query:
    st.warning("Please enter a question.")

st.markdown("""
**How to test:**
*   Try asking in English: `How do I track my order?`
*   Try asking in Spanish: `¿Cómo rastreo mi pedido?`
*   Try other questions from the knowledge base in both languages.
""")