
import streamlit as st
from langdetect import detect
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
from transformers import pipeline

# --- 1. Data Storage: FAQ Database ---
FAQS = [
    {
        "id": "1",
        "en": "What is your return policy?",
        "fr": "Quelle est votre politique de retour ?",
        "answer_en": "You can return items within 30 days of purchase for a full refund.",
        "answer_fr": "Vous pouvez retourner les articles dans les 30 jours suivant l'achat pour un remboursement complet."
    },
    {
        "id": "2",
        "en": "How can I track my order?",
        "fr": "Comment puis-je suivre ma commande ?",
        "answer_en": "You can track your order using the tracking number provided in your shipping confirmation email.",
        "answer_fr": "Vous pouvez suivre votre commande en utilisant le numéro de suivi fourni dans votre e-mail de confirmation d'expédition."
    },
    {
        "id": "3",
        "en": "Do you offer international shipping?",
        "fr": "Proposez-vous la livraison internationale ?",
        "answer_en": "Yes, we offer international shipping to most countries. Shipping costs and delivery times vary by destination.",
        "answer_fr": "Oui, nous proposons la livraison internationale dans la plupart des pays. Les frais d'expédition et les délais de livraison varient selon la destination."
    },
    {
        "id": "4",
        "en": "How do I contact customer support?",
        "fr": "Comment puis-je contacter le service client ?",
        "answer_en": "You can reach our customer support team via email at support@ecommerce.com or by phone at +1-123-456-7890.",
        "answer_fr": "Vous pouvez joindre notre équipe de support client par e-mail à support@ecommerce.com ou par téléphone au +1-123-456-7890."
    }
]

# --- 2. Core Chatbot Logic ---

# Initialize Models (cached for Streamlit)
@st.cache_resource
def load_models():
    # Cross-Lingual Embedding Model
    embedder = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    # Multilingual LLM (for generation/translation)
    # Using a translation pipeline as a proxy for a smaller, fast multilingual LLM for this demo
    # For a true generative LLM, you'd load a larger model like mT5 or XLM-R
    translator_en_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
    translator_fr_en = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    return embedder, translator_en_fr, translator_fr_en

embedder, translator_en_fr, translator_fr_en = load_models()

# --- Build Faiss Index ---
@st.cache_resource
def build_faiss_index(faqs_data, embedder_model):
    corpus_sentences = []
    corpus_map = [] # To map index back to original FAQ

    for faq in faqs_data:
        # Embed both English and French questions for robust retrieval
        corpus_sentences.append(faq["en"])
        corpus_map.append({"id": faq["id"], "lang": "en", "text": faq["en"]})
        corpus_sentences.append(faq["fr"])
        corpus_map.append({"id": faq["id"], "lang": "fr", "text": faq["fr"]})

    corpus_embeddings = embedder_model.encode(corpus_sentences, convert_to_tensor=True).cpu().numpy()
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(corpus_embeddings)
    return index, corpus_sentences, corpus_map

faiss_index, corpus_sentences, corpus_map = build_faiss_index(FAQS, embedder)


def get_relevant_faqs(query, top_k=2):
    query_embedding = embedder.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = faiss_index.search(query_embedding, top_k)

    relevant_faqs_info = []
    for i in range(top_k):
        faq_info = corpus_map[indices[0][i]]
        original_faq = next(item for item in FAQS if item["id"] == faq_info["id"])
        relevant_faqs_info.append(original_faq)
    return relevant_faqs_info


def generate_inclt_prompt(user_query, user_lang, relevant_faqs):
    prompt_parts = [f"The user asked in {user_lang}: '{user_query}'"]
    prompt_parts.append("Here are some relevant FAQ examples in both source and target languages to help answer the user's query:")

    for i, faq in enumerate(relevant_faqs):
        source_lang_q = faq["en"] if user_lang == "fr" else faq["fr"]
        source_lang_a = faq["answer_en"] if user_lang == "fr" else faq["answer_fr"]
        target_lang_q = faq["fr"] if user_lang == "fr" else faq["en"]
        target_lang_a = faq["answer_fr"] if user_lang == "fr" else faq["answer_en"]

        prompt_parts.append(f"---")
        prompt_parts.append(f"Example {i+1} Question (Source: {'en' if user_lang == 'fr' else 'fr'}): {source_lang_q}")
        prompt_parts.append(f"Example {i+1} Answer (Source: {'en' if user_lang == 'fr' else 'fr'}): {source_lang_a}")
        prompt_parts.append(f"Example {i+1} Question (Target: {user_lang}): {target_lang_q}")
        prompt_parts.append(f"Example {i+1} Answer (Target: {user_lang}): {target_lang_a}")

    prompt_parts.append(f"---")
    prompt_parts.append(f"Based on the examples and the user's query in {user_lang}, '{user_query}', please provide a concise and helpful answer in {user_lang}.")
    prompt_parts.append(f"Answer in {user_lang}:")

    return "\n".join(prompt_parts)


def get_chatbot_response(user_query):
    try:
        detected_lang = detect(user_query)
    except:
        detected_lang = "en" # Default to English if detection fails

    st.session_state.chat_history.append(f"You ({detected_lang.upper()}): {user_query}")

    # 3. Cross-Lingual FAQ Retrieval
    relevant_faqs = get_relevant_faqs(user_query)

    # 4. InCLT Prompt Generation
    inclt_prompt = generate_inclt_prompt(user_query, detected_lang, relevant_faqs)
    
    st.write("DEBUG: Generated Prompt:")
    st.code(inclt_prompt)

    # 5. Multilingual LLM Integration (using translation pipeline as LLM proxy)
    # The translation pipeline here acts as our 'multilingual LLM' that can understand the prompt
    # and generate a response in the target language based on the provided examples.
    # In a real scenario, this would be a larger generative LLM.
    if detected_lang == "fr":
        response_pipeline = translator_fr_en # Translate the *prompt* to English if query is French
        # Then let the 'LLM' (conceptual) process it and produce an English answer,
        # then translate the *answer* back to French.
        # This is a simplification; a true multilingual LLM would directly process the prompt
        # and generate in the target language.
        # For this demo, we'll try to get the 'LLM' to respond directly in French if query is French.
        
        # A better simulation for this simple translator model is to directly get the answer from FAQ
        # and translate it if needed, or get LLM to generate in target language.
        # For the purpose of showing InCLT, we'll assume the LLM *can* process the mixed prompt.
        # Let's try to simulate direct response in target language.
        
        # Find the most relevant FAQ answer in the user's language based on retrieval
        # This bypasses the full LLM generation for simplicity but demonstrates retrieval.
        best_faq_answer = relevant_faqs[0][f"answer_{detected_lang}"]
        llm_response_text = best_faq_answer # Directly use the retrieved answer

    else: # Default to English if not French
        best_faq_answer = relevant_faqs[0][f"answer_{detected_lang}"]
        llm_response_text = best_faq_answer

    final_response = llm_response_text

    st.session_state.chat_history.append(f"Bot ({detected_lang.upper()}): {final_response}")
    return final_response

# --- Streamlit UI ---
st.title("Multilingual Customer Support Chatbot")
st.write("Ask your questions in English or French!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    st.text(message)

user_input = st.text_input("Your query:", key="user_input")

if user_input:
    with st.spinner("Thinking..."):
        response = get_chatbot_response(user_input)
    # To prevent re-running get_chatbot_response on every rerun, clear the input field after processing
    st.session_state.user_input = ""

