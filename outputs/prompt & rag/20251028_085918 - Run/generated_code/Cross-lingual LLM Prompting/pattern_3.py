import streamlit as st
from transformers import pipeline

# 1. LLM Loading
@st.cache_resource
def load_translator():
    # Load the English-to-Spanish translation model as specified in the architecture.
    # This model will be used to demonstrate cross-lingual transfer in a translation-like context.
    return pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

translator = load_translator()

# 2. In-Context Example Manager (Pre-defined examples)
# These examples include both source (English) and target (Spanish) language information
# to facilitate the InCLT Crosslingual Transfer Prompting pattern.
IN_CONTEXT_EXAMPLES = [
    {
        "en_query": "How do I reset my password?",
        "en_answer": "You can reset your password by visiting the login page and clicking \"Forgot Password\".",
        "es_query": "¿Cómo reseteo mi contraseña?", # Spanish translation of the query
        "es_answer": "Puedes restablecer tu contraseña visitando la página de inicio de sesión y haciendo clic en \"Olvidé mi contraseña\"."
    },
    {
        "en_query": "My order is delayed.",
        "en_answer": "Please provide your order number for us to check the status.",
        "es_query": "Mi pedido está retrasado.",
        "es_answer": "Por favor, proporciona tu número de pedido para que podamos verificar el estado."
    },
    {
        "en_query": "What is your return policy?",
        "en_answer": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
        "es_query": "¿Cuál es su política de devoluciones?",
        "es_answer": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra con un recibo válido."
    }
]

# 3. Prompt Constructor
# This function dynamically generates the prompt for the LLM, incorporating the
# InCLT pattern by including both source and target language examples.
# The goal is to guide the EN->ES translation model to produce a more relevant
# Spanish customer support answer based on the provided cross-lingual context.
def build_inclt_prompt(user_query: str, examples: list) -> str:
    prompt_parts = [
        "Based on the following English customer queries and their corresponding Spanish answers, ",
        "provide a suitable Spanish answer for the final English query.\n"
    ]

    for example in examples:
        # Interleave source and target language examples as per InCLT pattern
        prompt_parts.append(f"Customer: {example['en_query']}\nAgent (EN): {example['en_answer']}\n")
        prompt_parts.append(f"Cliente: {example['es_query']}\nAgente (ES): {example['es_answer']}\n")

    prompt_parts.append(f"---\n")
    prompt_parts.append(f"Customer: {user_query}\nAgent (ES):") # The model is expected to complete this part in Spanish

    return "".join(prompt_parts)

# 4. Streamlit Chatbot Interface
st.set_page_config(page_title="InCLT Chatbot Demo", layout="centered")
st.title("Multilingual Customer Support Chatbot (InCLT Demo)")
st.subheader("Enhancing Cross-lingual Knowledge Transfer with Prompting")

st.markdown(
    """
    This chatbot demonstrates the **InCLT Crosslingual Transfer Prompting** pattern.
    Enter an **English query**, and the chatbot will generate a **Spanish response**.
    We use a translation model (`Helsinki-NLP/opus-mt-en-es`) with **in-context examples**
    (containing both English and Spanish) to guide its generation of a more contextually
    appropriate Spanish answer, showcasing a form of cross-lingual knowledge transfer.
    The model's ability to 'transfer' knowledge is observed in how it interprets the
    English query and formulates a Spanish response influenced by the provided patterns.
    """
)

user_input = st.text_area("Enter your customer support query in English:", key="user_query_input")

if st.button("Get Spanish Response", key="get_response_button"):
    if user_input:
        # Construct the prompt using the InCLT pattern
        full_prompt = build_inclt_prompt(user_input, IN_CONTEXT_EXAMPLES)

        st.write("### Prompt Fed to the Translator Model:")
        st.code(full_prompt, language="markdown")

        with st.spinner("Generating Spanish response..."):
            try:
                # Perform LLM inference (translation from English-rich prompt to Spanish)
                # max_length is crucial to allow for a comprehensive generated response.
                # clean_up_tokenization_spaces helps ensure readable output.
                translation_output = translator(full_prompt, max_length=200, clean_up_tokenization_spaces=True)
                
                if translation_output and translation_output[0] and 'translation_text' in translation_output[0]:
                    raw_translated_text = translation_output[0]['translation_text']
                    
                    # Attempt to parse the actual Spanish answer from the full translated prompt.
                    # This parsing is a heuristic due to using a translation model for a generative task.
                    marker = "Agente (ES):"
                    marker_index = raw_translated_text.rfind(marker)

                    if marker_index != -1:
                        spanish_response = raw_translated_text[marker_index + len(marker):].strip()
                        st.write("### Chatbot's Spanish Response:")
                        st.success(spanish_response)
                    else:
                        st.warning("Could not clearly parse the Spanish answer from the model's output. Full translated text below:")
                        st.code(raw_translated_text, language="spanish")
                else:
                    st.error("Could not get a translation from the model. The output was empty or malformed.")
            except Exception as e:
                st.error(f"An error occurred during response generation: {e}")
    else:
        st.warning("Please enter an English query to get a Spanish response.")