import streamlit as st
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langdetect import detect
import numpy as np
import json
import torch

# --- 1. Data Storage --- 
# FAQ Knowledge Base (English as source language)
FAQ_KNOWLEDGE_BASE = {
    "en": {
        "How can I track my order?": "You can track your order using the tracking link provided in your shipping confirmation email.",
        "What is your return policy?": "Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging.",
        "How do I contact customer support?": "You can contact customer support via email at support@example.com or call us at 1-800-XXX-XXXX.",
        "Do you offer international shipping?": "Yes, we offer international shipping to most countries. Shipping costs and delivery times vary by destination."
    },
    "es": {
        "¿Cómo puedo rastrear mi pedido?": "Puede rastrear su pedido utilizando el enlace de seguimiento proporcionado en su correo electrónico de confirmación de envío.",
        "¿Cuál es su política de devolución?": "Nuestra política de devolución permite devoluciones dentro de los 30 días posteriores a la compra, siempre que el artículo no haya sido utilizado y esté en su embalaje original.",
        "¿Cómo me comunico con el servicio de atención al cliente?": "Puede ponerse en contacto con el servicio de atención al cliente por correo electrónico en support@example.com o llamarnos al 1-800-XXX-XXXX.",
        "¿Ofrecen envíos internacionales?": "Sí, ofrecemos envíos internacionales a la mayoría de los países. Los costos de envío y los tiempos de entrega varían según el destino."
    },
    "fr": {
        "Comment puis-je suivre ma commande ?": "Vous pouvez suivre votre commande à l'aide du lien de suivi fourni dans votre e-mail de confirmation d'expédition.",
        "Quelle est votre politique de retour ?": "Notre politique de retour permet les retours dans les 30 jours suivant l'achat, à condition que l'article soit inutilisé et dans son emballage d'origine.",
        "Comment puis-je contacter le service client ?": "Vous pouvez contacter le service client par e-mail à support@example.com ou nous appeler au 1-800-XXX-XXXX.",
        "Proposez-vous la livraison internationale ?": "Oui, nous proposons la livraison internationale dans la plupart des pays. Les frais d'expédition et les délais de livraison varient selon la destination."
    }
}

# In-Context Learning Examples for Cross-lingual Transfer
# Each example contains a source (English) query and answer, and its target language equivalent.
INCLT_EXAMPLES = [
    {
        "source_lang": "en",
        "target_lang": "es",
        "source_query": "How can I track my order?",
        "source_answer": "You can find your tracking number in your shipping confirmation email.",
        "target_query": "¿Cómo puedo rastrear mi pedido?",
        "target_answer": "Puede encontrar su número de seguimiento en el correo electrónico de confirmación de envío."
    },
    {
        "source_lang": "en",
        "target_lang": "fr",
        "source_query": "What is your return policy?",
        "source_answer": "Our return policy allows returns within 30 days of purchase.",
        "target_query": "Quelle est votre politique de retour ?",
        "target_answer": "Notre politique de retour permet les retours dans les 30 jours suivant l'achat."
    },
    {
        "source_lang": "en",
        "target_lang": "es",
        "source_query": "Do you ship internationally?",
        "source_answer": "Yes, we ship to many countries worldwide.",
        "target_query": "¿Envían internacionalmente?",
        "target_answer": "Sí, enviamos a muchos países en todo el mundo."
    }
]

# --- 2. LLM and Embeddings Initialization ---
# Using a smaller, multilingual model for demonstration. For production, consider larger models.
MODEL_NAME = "Helsinki-NLP/opus-mt-en-es" # Example for a specific language pair, typically use a truly multilingual instruction-tuned LLM
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Try to load a truly multilingual LLM if available and suitable for chat (e.g., mBART, XLM-R finetuned for chat)
# For simplicity and to fit a common LLM pattern, let's simulate with a generic text generation setup.
# In a real scenario, you'd load a model capable of generating coherent multi-turn dialogue across languages.
# For this example, we'll use a model that handles basic sequence-to-sequence if a more advanced chat LLM isn't easily loadable for direct generation in a simple script.
# Let's use a generic generative model and rely on prompting for cross-lingual capability.

try:
    llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    llm_model = AutoModelForCausalLM.from_pretrained("google/flan-t5-small")
except Exception:
    # Fallback if a more complex generative model is not easily loaded or available
    st.warning("Could not load flan-t5-small. Using a dummy LLM response.")
    llm_tokenizer = None
    llm_model = None

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Pre-compute embeddings for FAQ questions for faster retrieval
faq_questions_en = list(FAQ_KNOWLEDGE_BASE["en"].keys())
faq_embeddings_en = embedding_model.encode(faq_questions_en, convert_to_tensor=True)

# --- 3. Helper Functions ---
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en" # Default to English if detection fails

def find_relevant_faqs(query: str, query_lang: str, top_k: int = 2):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    # For simplicity, we'll only search against English FAQs using English embeddings
    # In a more advanced setup, you'd have multilingual embeddings for all FAQs or translate the query.
    similarities = util.cos_sim(query_embedding, faq_embeddings_en)[0]
    top_faq_indices = torch.topk(similarities, k=top_k).indices.tolist()

    relevant_faqs = []
    for idx in top_faq_indices:
        en_question = faq_questions_en[idx]
        en_answer = FAQ_KNOWLEDGE_BASE["en"][en_question]
        
        # Get the translated version if available for the target language
        target_question = FAQ_KNOWLEDGE_BASE.get(query_lang, {}).get(en_question, en_question) # Fallback to English if no translation
        target_answer = FAQ_KNOWLEDGE_BASE.get(query_lang, {}).get(en_question, en_answer) # Fallback to English if no translation
        
        relevant_faqs.append({
            "en_question": en_question,
            "en_answer": en_answer,
            "target_question": target_question,
            "target_answer": target_answer
        })
    return relevant_faqs

def construct_inclt_prompt(customer_query: str, detected_lang: str, relevant_faqs: list) -> str:
    prompt_parts = []

    # Add InCLT examples first
    for example in INCLT_EXAMPLES:
        if example["target_lang"] == detected_lang or example["source_lang"] == detected_lang: # Prioritize relevant languages
            prompt_parts.append(f"Source Query ({example['source_lang']}): {example['source_query']}")
            prompt_parts.append(f"Source Answer ({example['source_lang']}): {example['source_answer']}")
            prompt_parts.append(f"Target Query ({example['target_lang']}): {example['target_query']}")
            prompt_parts.append(f"Target Answer ({example['target_lang']}): {example['target_answer']}\n")

    # Add relevant FAQs from the knowledge base
    for faq in relevant_faqs:
        # Present both source (English) and target language versions in the prompt
        prompt_parts.append(f"Context (EN): {faq['en_question']} -> {faq['en_answer']}")
        if faq['target_question'] != faq['en_question']:
            prompt_parts.append(f"Context ({detected_lang.upper()}): {faq['target_question']} -> {faq['target_answer']}\n")
        else:
            prompt_parts.append(f"\n")

    prompt_parts.append(f"Customer Query ({detected_lang.upper()}): {customer_query}")
    prompt_parts.append(f"Chatbot Response ({detected_lang.upper()}):")

    return "\n".join(prompt_parts)

def generate_llm_response(prompt: str) -> str:
    if llm_model and llm_tokenizer:
        inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = llm_model.generate(**inputs, max_new_tokens=150, num_beams=5, early_stopping=True)
        response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # T5 models generate the entire sequence, we need to extract the answer part
        # This assumes the LLM correctly completes the 