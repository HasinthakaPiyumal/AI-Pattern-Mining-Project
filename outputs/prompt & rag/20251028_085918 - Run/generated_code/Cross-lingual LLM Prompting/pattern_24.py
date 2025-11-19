import streamlit as st
# For a real application, you would use a robust language detection library like 'langdetect'
# from langdetect import detect

# For a real application, you would integrate with an actual LLM service (e.g., Hugging Face, OpenAI)
# from transformers import pipeline

# --- 1. InCLT Examples --- 
# These examples incorporate both source and target languages to enhance cross-lingual understanding.
INCLT_EXAMPLES = [
    {
        "source_lang": "en",
        "target_lang": "es",
        "query_source": "My package is delayed. Can you check its status?",
        "answer_target": "¿Podría proporcionar su número de pedido para que pueda verificar el estado de su paquete?"
    },
    {
        "source_lang": "es",
        "target_lang": "en",
        "query_source": "Quiero cancelar mi suscripción. ¿Cómo lo hago?",
        "answer_target": "Please confirm your account email address so I can assist you with the cancellation."
    },
    {
        "source_lang": "en",
        "target_lang": "fr",
        "query_source": "How do I reset my password if I forgot it?",
        "answer_target": "Veuillez visiter le lien 'Mot de passe oublié' sur notre page de connexion pour réinitialiser votre mot de passe."
    },
    {
        "source_lang": "fr",
        "target_lang": "en",
        "query_source": "J'ai un problème avec ma facture, pouvez-vous m'aider ?",
        "answer_target": "Could you please provide your invoice number and explain the issue?"
    }
]

# --- 2. Simplified Language Detection (Mock) ---
def detect_language(text):
    # This is a simplified mock-up for demonstration purposes.
    # In a production environment, you would use a dedicated library like 'langdetect' or an ML model.
    text_lower = text.lower()
    if "hola" in text_lower or "gracias" in text_lower or "español" in text_lower or "pedido" in text_lower:
        return "es"
    if "bonjour" in text_lower or "merci" in text_lower or "français" in text_lower or "facture" in text_lower:
        return "fr"
    return "en" # Default to English if no specific keywords are found

# --- 3. In-Context Learning (InCLT) Prompting Module ---
def build_inclt_prompt(user_query, detected_lang, target_response_lang, examples):
    prompt_parts = [
        f"You are a helpful multilingual customer support assistant. Your task is to understand the user's query and respond accurately in {target_response_lang}. Below are some cross-lingual examples to guide your understanding and response generation. These examples show how to understand a query in one language and respond appropriately, potentially in another if instructed, or in the same language if that's the expected target. Pay close attention to the language pairings and expected responses."
    ]

    for i, ex in enumerate(examples):
        prompt_parts.append(f"\n--- Example {i+1} ---")
        prompt_parts.append(f"User ({ex['source_lang']}): {ex['query_source']}")
        prompt_parts.append(f"Assistant ({ex['target_lang']}): {ex['answer_target']}")

    prompt_parts.append(f"\n---\nNow, process the following user query. Respond in {target_response_lang} based on your understanding and the provided examples:")
    prompt_parts.append(f"User ({detected_lang}): {user_query}")
    prompt_parts.append(f"Assistant ({target_response_lang}):")

    return "\n".join(prompt_parts)

# --- 4. Simplified Multilingual Large Language Model (LLM) Integration (Mock) ---
def get_llm_response(prompt, target_response_lang):
    # This function simulates an LLM's response based on the prompt and desired target language.
    # In a real application, this would involve calling a sophisticated LLM via an API or a local model.

    # Simple keyword-based response mapping for demonstration.
    # The actual LLM would leverage the in-context examples to generate a more nuanced response.
    response_map = {
        "en": {
            "package delayed": "Could you please provide your order number so I can check the status?",
            "cancel subscription": "Please confirm your account email address for cancellation.",
            "reset password": "Please visit the 'Forgot Password' link on our login page.",
            "invoice issue": "Could you provide your invoice number and describe the problem?",
            "default": "How can I assist you further today?"
        },
        "es": {
            "paquete retrasado": "¿Podría proporcionar su número de pedido para que pueda verificar el estado?",
            "cancelar suscripción": "Por favor, confirme su dirección de correo electrónico para la cancelación.",
            "restablecer contraseña": "Vaya al enlace 'Olvidé mi contraseña' en nuestra página de inicio de sesión.",
            "problema factura": "¿Podría proporcionar su número de factura y describir el problema?",
            "default": "¿En qué más puedo ayudarle hoy?"
        },
        "fr": {
            "colis retardé": "Pourriez-vous nous fournir votre numéro de commande pour vérifier le statut ?",
            "annuler abonnement": "Veuillez confirmer l'adresse e-mail de votre compte pour l'annulation.",
            "réinitialiser mot de passe": "Veuillez visiter le lien 'Mot de passe oublié' sur notre page de connexion.",
            "problème facture": "Pourriez-vous fournir votre numéro de facture et décrire le problème ?",
            "default": "Comment puis-je vous aider davantage aujourd'hui ?"
        }
    }

    lower_prompt = prompt.lower()
    chosen_response_map = response_map.get(target_response_lang, response_map["en"]) # Default to English if lang not found

    response = chosen_response_map["default"]
    for keyword, resp_text in chosen_response_map.items():
        if keyword != "default" and keyword in lower_prompt:
            response = resp_text
            break
    return response

# --- 5. Streamlit User Interface (Frontend) ---
st.set_page_config(layout="wide")
st.title("🌍 Multilingual Customer Support Chatbot (InCLT Demo)")
st.markdown("This chatbot demonstrates **Cross-Lingual In-Context Learning Prompting (InCLT)**. It constructs prompts with examples in both source and target languages to improve understanding and response generation for multilingual queries.")

st.sidebar.header("Chatbot Settings")
target_response_lang = st.sidebar.selectbox(
    "Select Assistant's Response Language",
    ("en", "es", "fr"),
    index=0,
    format_func=lambda x: {"en": "English", "es": "Español", "fr": "Français"}.get(x, x),
    help="The language in which the chatbot should generate its final response."
)

st.write("Type your customer support query below. The chatbot will use cross-lingual in-context examples to understand and respond in the selected language.")

user_input = st.text_area("Your Customer Query:", height=150, placeholder=f"e.g., 'My internet is not working' or 'Mi pedido está retrasado'.")

if st.button("Get Chatbot Response"):
    if user_input:
        detected_lang = detect_language(user_input)
        st.info(f"**Detected Language of your query:** `{detected_lang.upper()}`")

        inclt_prompt = build_inclt_prompt(user_input, detected_lang, target_response_lang, INCLT_EXAMPLES)

        st.subheader("🔬 **Generated InCLT Prompt (sent to LLM):**")
        st.code(inclt_prompt, language="markdown")
        st.markdown("*(This is the prompt constructed with cross-lingual examples, ready to be sent to a Large Language Model.)* ")

        st.subheader(f"🤖 **Chatbot's Response (in {target_response_lang.upper()}):**")
        llm_response = get_llm_response(inclt_prompt, target_response_lang)
        st.success(llm_response)
    else:
        st.warning("Please enter a query to get a chatbot response.")

st.markdown("--- ")
st.markdown("**Note:** This is a simplified demonstration. A real-world application would integrate with advanced LLMs and robust language detection services.")