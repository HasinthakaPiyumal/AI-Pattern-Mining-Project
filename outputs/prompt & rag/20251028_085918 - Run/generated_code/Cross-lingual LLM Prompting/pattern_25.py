import streamlit as st
from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

# Ensure consistent language detection results
DetectorFactory.seed = 0

# --- Configuration --- #
TARGET_LANG = "en" # Internal processing language

# --- Load Translation Models (Cached) ---
@st.cache_resource
def load_translation_models():
    # Example: English <-> Spanish translation models
    # In a real application, you'd load models for all supported languages
    # or use a more comprehensive multilingual model like NLLB.
    
    # Spanish to English
    es_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    es_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")

    # English to Spanish
    en_es_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    en_es_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    
    # English to French
    en_fr_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    en_fr_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    
    # French to English
    fr_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    fr_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    
    return {
        "es-en": {"tokenizer": es_en_tokenizer, "model": es_en_model},
        "en-es": {"tokenizer": en_es_tokenizer, "model": en_es_model},
        "en-fr": {"tokenizer": en_fr_tokenizer, "model": en_fr_model},
        "fr-en": {"tokenizer": fr_en_tokenizer, "model": fr_en_model},
    }

translation_models = load_translation_models()

def translate(text, src_lang, dest_lang):
    if src_lang == dest_lang:
        return text

    model_key = f"{src_lang}-{dest_lang}"
    if model_key not in translation_models:
        st.warning(f"Translation model for {src_lang} to {dest_lang} not loaded. Using original text.")
        return text

    tokenizer = translation_models[model_key]["tokenizer"]
    model = translation_models[model_key]["model"]
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    translated_tokens = model.generate(**inputs, max_new_tokens=500)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# --- Load LLM (Cached) ---
@st.cache_resource
def load_llm():
    # Using a smaller T5 model for demonstration. For a real chatbot,
    # consider larger models like Llama, Mistral, or API-based LLMs (e.g., OpenAI, Gemini).
    llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return llm_tokenizer, llm_model

llm_tokenizer, llm_model = load_llm()

def get_llm_response(prompt):
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = llm_model.generate(**inputs, max_new_tokens=200, num_beams=5, early_stopping=True)
    return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- In-Context Learning Examples (Static for demo) ---
# In a real application, these might be dynamically retrieved from a database
# based on semantic similarity to the user's query.
INCLT_EXAMPLES = [
    {
        "source_lang": "es",
        "query": "Mi internet no funciona.",
        "in_context_prompt_part": "Customer: Mi internet no funciona.\nAgent (English): My internet is not working. How can I help you?",
        "target_response": "Please restart your router and modem. If the issue persists, contact technical support.",
        "target_response_es": "Por favor, reinicie su router y módem. Si el problema persiste, contacte a soporte técnico."
    },
    {
        "source_lang": "es",
        "query": "Necesito cambiar mi dirección de envío.",
        "in_context_prompt_part": "Customer: Necesito cambiar mi dirección de envío.\nAgent (English): I need to change my shipping address. Can I help you with that?",
        "target_response": "Sure, please provide your new shipping address and order number.",
        "target_response_es": "Claro, por favor proporcione su nueva dirección de envío y número de pedido."
    },
    {
        "source_lang": "fr",
        "query": "Ma commande est en retard.",
        "in_context_prompt_part": "Customer: Ma commande est en retard.\nAgent (English): My order is late. What is your order number?",
        "target_response": "Could you please provide your order number so I can check its status?",
        "target_response_fr": "Pourriez-vous me fournir votre numéro de commande afin que je puisse vérifier son statut ?"
    },
    {
        "source_lang": "fr",
        "query": "Comment réinitialiser mon mot de passe ?",
        "in_context_prompt_part": "Customer: Comment réinitialiser mon mot de passe ?\nAgent (English): How do I reset my password? I can guide you through the process.",
        "target_response": "You can reset your password by visiting the 'Forgot Password' link on our login page.",
        "target_response_fr": "Vous pouvez réinitialiser votre mot de passe en visitant le lien 'Mot de passe oublié' sur notre page de connexion."
    },
]

def get_in_context_examples(query_lang, num_examples=2):
    # Filter examples relevant to the query language (or a general set)
    relevant_examples = [ex for ex in INCLT_EXAMPLES if ex["source_lang"] == query_lang or ex["source_lang"] == TARGET_LANG]
    
    # For a simple demo, just take the first 'num_examples' if available
    # In a real system, you'd use semantic similarity to retrieve the *most* relevant ones.
    return relevant_examples[:num_examples]


def construct_in_clt_prompt(original_query, query_lang, translated_query, in_context_examples):
    system_instruction = (
        "You are a multilingual customer support agent. "
        f"Understand the customer's query in any language, process it internally in {TARGET_LANG}, "
        "and provide a helpful response. Below are examples of how to respond."
    )

    example_prompts = []
    for example in in_context_examples:
        example_prompts.append(example["in_context_prompt_part"])

    # Combine system instruction, examples, and the current query
    prompt_parts = [system_instruction] + example_prompts
    
    # Add the current query, translated to target lang for LLM processing context
    if query_lang == TARGET_LANG:
        prompt_parts.append(f"Customer: {original_query}\nAgent (English):")
    else:
        prompt_parts.append(
            f"Customer ({query_lang}): {original_query}\n" +
            f"Customer (English - internal translation): {translated_query}\nAgent (English):"
        )
    
    return "\n\n".join(prompt_parts)

# --- Streamlit UI --- #
st.title("🌍 Multilingual Customer Support Chatbot")
st.markdown("Ask your questions in Spanish, French, or English!")

user_query = st.text_area("Your Question:", "")

if st.button("Get Response"):
    if user_query:
        with st.spinner("Detecting language and generating response..."):
            try:
                # 1. Language Detection
                query_lang = detect(user_query)
                st.write(f"Detected Language: {query_lang}")

                # 2. Translate Query to Target Language (if necessary)
                translated_query = translate(user_query, query_lang, TARGET_LANG)
                if query_lang != TARGET_LANG:
                    st.write(f"Translated Query (to English): {translated_query}")

                # 3. Get In-Context Examples
                in_context_examples = get_in_context_examples(query_lang)

                # 4. Construct InCLT Prompt
                full_prompt = construct_in_clt_prompt(user_query, query_lang, translated_query, in_context_examples)
                
                # st.text_area("Full Prompt sent to LLM:", full_prompt, height=300)

                # 5. Get LLM Response in Target Language
                llm_raw_response = get_llm_response(full_prompt)
                st.write(f"LLM Raw Response (in English): {llm_raw_response}")

                # 6. Translate Response back to Original Language (if necessary)
                final_response = translate(llm_raw_response, TARGET_LANG, query_lang)
                
                st.success("**Chatbot Response:**")
                st.write(final_response)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Please ensure you have an active internet connection to download models, or restart the app if issues persist.")
    else:
        st.warning("Please enter a question.")
