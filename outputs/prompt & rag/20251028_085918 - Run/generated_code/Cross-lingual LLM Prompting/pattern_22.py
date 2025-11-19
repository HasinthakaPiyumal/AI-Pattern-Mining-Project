import streamlit as st
from langdetect import detect
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Load Models and Data

# Sentence Transformer for embedding queries and knowledge base
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embedding_model = load_embedding_model()

# Multilingual LLM (e.g., for translation and generation)
@st.cache_resource
def load_llm():
    # Using a simpler model for demonstration, ideally a larger multilingual model would be used
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    return tokenizer, model

llm_tokenizer, llm_model = load_llm()

# 2. Knowledge Base and FAISS Index

# Dummy Knowledge Base: (Question in English, Question in Spanish, Answer in English, Answer in Spanish)
knowledge_base = [
    {"en_q": "What is your return policy?", "es_q": "¿Cuál es su política de devolución?", "en_a": "Our return policy allows returns within 30 days of purchase with a valid receipt.", "es_a": "Nuestra política de devolución permite devoluciones dentro de los 30 días posteriores a la compra con un recibo válido.", "id": 0},
    {"en_q": "How do I track my order?", "es_q": "¿Cómo rastreo mi pedido?", "en_a": "You can track your order using the tracking number provided in your shipping confirmation email.", "es_a": "Puede rastrear su pedido utilizando el número de seguimiento que se proporciona en el correo electrónico de confirmación de envío.", "id": 1},
    {"en_q": "Can I change my shipping address?", "es_q": "¿Puedo cambiar mi dirección de envío?", "en_a": "Please contact customer support immediately to change your shipping address. Changes may not be possible if the order has already shipped.", "es_a": "Póngase en contacto con el servicio de atención al cliente de inmediato para cambiar su dirección de envío. Es posible que los cambios no sean posibles si el pedido ya ha sido enviado.", "id": 2},
    {"en_q": "What payment methods do you accept?", "es_q": "¿Qué métodos de pago aceptan?", "en_a": "We accept major credit cards, PayPal, and Google Pay.", "es_a": "Aceptamos las principales tarjetas de crédito, PayPal y Google Pay.", "id": 3},
    {"en_q": "How do I reset my password?", "es_q": "¿Cómo restablezco mi contraseña?", "en_a": "You can reset your password by clicking on 'Forgot Password' on the login page.", "es_a": "Puede restablecer su contraseña haciendo clic en 'Olvidé mi contraseña' en la página de inicio de sesión.", "id": 4},
]

# Create embeddings for all English questions in the knowledge base
kb_questions_en = [item["en_q"] for item in knowledge_base]
kb_embeddings_en = embedding_model.encode(kb_questions_en)

# Create FAISS index
dimension = kb_embeddings_en.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.asarray(kb_embeddings_en).astype('float32'))

# 3. InCLT Prompting Logic
def get_in_context_examples(user_query, source_lang, top_k=2):
    query_embedding = embedding_model.encode([user_query])
    D, I = faiss_index.search(np.asarray(query_embedding).astype('float32'), top_k)

    examples_text = []
    for idx in I[0]:
        example = knowledge_base[idx]
        # Construct example in both source and target language (English assumed as target for LLM processing)
        example_prompt = f"Question ({source_lang}): {getattr(example, f'{source_lang}_q', example['en_q'])} (English): {example['en_q']}\nAnswer (English): {example['en_a']}"
        examples_text.append(example_prompt)
    return "\n".join(examples_text)

def generate_llm_response(prompt):
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = llm_model.generate(**inputs, max_new_tokens=200, num_beams=5, early_stopping=True)
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 4. Streamlit UI
st.title("Multilingual Customer Support Chatbot")
st.write("Ask your question in any supported language and get a response!")

user_input = st.text_area("Your Question:", "")

if st.button("Get Response") and user_input:
    with st.spinner("Detecting language and generating response..."):
        try:
            source_lang = detect(user_input)
            st.info(f"Detected language: {source_lang.upper()}")

            # Step 1: Get in-context examples
            in_context_examples = get_in_context_examples(user_input, source_lang)

            # Step 2: Construct the final prompt for the LLM
            # We'll translate the user's question to English for better LLM performance if not English
            if source_lang != 'en':
                translation_inputs = llm_tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
                translated_user_input_tokens = llm_model.generate(**translation_inputs, max_new_tokens=100)
                translated_user_input = llm_tokenizer.decode(translated_user_input_tokens[0], skip_special_tokens=True)
            else:
                translated_user_input = user_input
            
            # The prompt includes the original source language query, the translated query, and cross-lingual examples.
            # The LLM is then asked to respond in English.
            prompt = f"""Here are some example customer support interactions:
{in_context_examples}

User Question ({source_lang}): {user_input}
User Question (English): {translated_user_input}
Assistant (English):"""

            # Step 3: Generate response from the LLM
            chatbot_response = generate_llm_response(prompt)

            st.subheader("Chatbot Response:")
            st.write(chatbot_response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Could not detect language or generate response. Please try again.")

