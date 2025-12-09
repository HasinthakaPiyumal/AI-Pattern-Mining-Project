import streamlit as st
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import torch

# Fix for langdetect to ensure consistent results
DetectorFactory.seed = 0

# --- 1. Load Models ---
@st.cache_resource
def load_models():
    # Embedding model for multilingual retrieval
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Translation models (English <-> Spanish for demonstration)
    # In a production setting, this would be extended for more languages or use a single, more robust multilingual translation model.
    en_es_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    en_es_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")

    es_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    es_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")

    # Multilingual Large Language Model for generating responses
    llm_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    return embedder, en_es_tokenizer, en_es_model, es_en_tokenizer, es_en_model, llm_tokenizer, llm_model

embedder, en_es_tokenizer, en_es_model, es_en_tokenizer, es_en_model, llm_tokenizer, llm_model = load_models()

# --- 2. Knowledge Base (English) ---
# A simple in-memory knowledge base, primarily in English (source language).
knowledge_base = {
    "product_returns": "Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging. Refunds are processed within 5-7 business days.",
    "shipping_options": "We offer standard shipping (3-5 business days) and express shipping (1-2 business days). Shipping costs vary based on destination and package weight.",
    "account_management": "To update your account information, log in to your profile and navigate to the 'Settings' section. You can change your password, address, and payment methods there.",
    "technical_support": "For technical issues, please visit our FAQ page or contact our support team via email or phone. Our technicians are available Monday to Friday, 9 AM to 5 PM EST.",
    "warranty_information": "All our electronic products come with a one-year manufacturer's warranty covering defects in materials and workmanship. Accidental damage is not covered."
}
kb_articles = list(knowledge_base.values())

# Pre-compute embeddings for the knowledge base articles
kb_embeddings = embedder.encode(kb_articles, convert_to_tensor=True)

# --- 3. Helper Functions ---
def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "en" # Default to English if detection fails

def perform_translation(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=512)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def retrieve_knowledge(query_embedding, kb_embeddings, kb_articles, top_k=2):
    # Calculate cosine similarity between query embedding and KB embeddings
    cos_scores = util.cos_sim(query_embedding, kb_embeddings)[0]
    # Get the top-k most similar articles
    top_results = torch.topk(cos_scores, k=top_k)
    retrieved_snippets = [kb_articles[idx] for idx in top_results.indices]
    return retrieved_snippets

def create_inclt_prompt(user_query, target_lang, retrieved_english_snippets, translated_snippets,
                        en_es_tokenizer_obj, en_es_model_obj):
    prompt_parts = []
    prompt_parts.append(f"You are a helpful customer support assistant. Answer the following question in {target_lang}. \n")
    prompt_parts.append("Here are some examples of how to answer questions, leveraging both English and "
                        f"{target_lang} information to facilitate cross-lingual transfer:\n\n")

    # Generate in-context examples with both source (English) and target language content
    for i, (en_snippet, target_snippet_for_example) in enumerate(zip(retrieved_english_snippets, translated_snippets)):
        # Simulate a simple question and answer based on the snippet for the in-context example
        simulated_en_question = f"What is the core message of this: {en_snippet[:min(50, len(en_snippet))]}..."
        simulated_en_answer = f"The primary information is: {en_snippet}"

        # Translate the simulated English Q&A to the target language for the example
        if target_lang == "es":
            simulated_target_question = perform_translation(simulated_en_question, en_es_tokenizer_obj, en_es_model_obj)
            simulated_target_answer = perform_translation(simulated_en_answer, en_es_tokenizer_obj, en_es_model_obj)
        else:
            # Fallback for languages other than Spanish (for demo simplicity)
            simulated_target_question = simulated_en_question
            simulated_target_answer = simulated_en_answer

        prompt_parts.append(f"Example {i+1} (English):\nContext: {en_snippet}\nQuestion: {simulated_en_question}\nAnswer: {simulated_en_answer}\n\n")
        prompt_parts.append(f"Example {i+1} ({target_lang}):\nContext: {target_snippet_for_example}\nQuestion: {simulated_target_question}\nAnswer: {simulated_target_answer}\n\n")

    prompt_parts.append(f"Now, based on the provided context, answer the following question in {target_lang}:\n")
    # Provide the most relevant translated context for the actual user query
    final_context_for_query = translated_snippets[0] if translated_snippets else 'No relevant context found.'
    prompt_parts.append(f"Context: {final_context_for_query}\n")
    prompt_parts.append(f"Question: {user_query}\nAnswer:")

    return "".join(prompt_parts)

# --- Streamlit App ---
st.title("🌍 Multilingual Customer Support Chatbot (InCLT Prompting)")
st.write("Ask a question in English or Spanish, and the chatbot will use cross-lingual in-context learning to respond.")

user_query = st.text_area("Your Question:")

if st.button("Ask"):
    if user_query:
        with st.spinner("Thinking..."):
            # 1. Detect Language of the user's query
            detected_lang = detect_language(user_query)
            st.info(f"Detected Language: {detected_lang}")

            # 2. Translate query to English for better retrieval if not English
            # This step helps align the query with the English-based knowledge base.
            query_for_embedding = user_query
            if detected_lang != "en":
                st.info(f"Translating query from {detected_lang} to English for embedding...")
                if detected_lang == "es": # Use Spanish to English translator
                    query_for_embedding = perform_translation(user_query, es_en_tokenizer, es_en_model)
                else:
                    st.warning(f"No specific translator for {detected_lang} to English. Using original query for embedding.")
                st.info(f"Query used for embedding: {query_for_embedding}")

            # 3. Embed Query and Retrieve Knowledge from the English KB
            query_embedding = embedder.encode(query_for_embedding, convert_to_tensor=True)
            retrieved_english_snippets = retrieve_knowledge(query_embedding, kb_embeddings, kb_articles)
            st.subheader("Retrieved English Knowledge Snippets:")
            for i, snippet in enumerate(retrieved_english_snippets):
                st.write(f"- {snippet}")

            # 4. Translate Retrieved English Snippets to the target language (user's query language)
            translated_snippets = []
            if detected_lang != "en":
                st.info(f"Translating retrieved English snippets to {detected_lang}...")
                for snippet in retrieved_english_snippets:
                    if detected_lang == "es": # Use English to Spanish translator
                        translated_snippets.append(perform_translation(snippet, en_es_tokenizer, en_es_model))
                    else:
                        translated_snippets.append(snippet) # Fallback to English if no specific model for target lang
                        st.warning(f"No specific translator for English to {detected_lang}. Using English snippets directly for examples.")
                st.subheader(f"Translated Knowledge Snippets ({detected_lang}):")
                for i, snippet in enumerate(translated_snippets):
                    st.write(f"- {snippet}")
            else:
                translated_snippets = retrieved_english_snippets # If target is English, no translation needed

            # 5. Create InCLT Prompt for the Multilingual LLM
            final_prompt = create_inclt_prompt(user_query, detected_lang, retrieved_english_snippets,
                                               translated_snippets, en_es_tokenizer, en_es_model)
            st.subheader("Generated LLM Prompt (excerpt):")
            st.text(final_prompt[:700] + "... [truncated]" if len(final_prompt) > 700 else final_prompt) # Show an excerpt

            # 6. Call Multilingual LLM to generate the response
            inputs = llm_tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=1024)
            output = llm_model.generate(**inputs, max_new_tokens=200)
            llm_response = llm_tokenizer.decode(output[0], skip_special_tokens=True)

            st.subheader("Chatbot Response:")
            st.success(llm_response)
    else:
        st.warning("Please enter a question.")