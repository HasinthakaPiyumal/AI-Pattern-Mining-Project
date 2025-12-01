import streamlit as st
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

DetectorFactory.seed = 0

client = chromadb.Client()
collection_name = "ecommerce_products_kb"

try:
    kb_collection = client.get_or_create_collection(name=collection_name)
except Exception as e:
    st.error(f"Error accessing ChromaDB: {e}")
    st.stop()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

kb_data = [
    {"id": "p1", "text": "The 'EcoBlend' coffee maker brews delicious coffee with sustainable pods. Features a 1.2L capacity and programmable timer.", "metadata": {"product_name": "EcoBlend Coffee Maker"}},
    {"id": "p2", "text": "Our 'ComfyCloud' wireless headphones offer superior sound quality and 20 hours of battery life. Ergonomic design for maximum comfort.", "metadata": {"product_name": "ComfyCloud Headphones"}},
    {"id": "p3", "text": "The 'GourmetPro' stand mixer is perfect for baking enthusiasts. Comes with multiple attachments for kneading, whisking, and mixing.", "metadata": {"product_name": "GourmetPro Stand Mixer"}},
    {"id": "p4", "text": "The 'BrightVision' smart lamp features adjustable brightness and color temperature, controlled via a mobile app. Energy-efficient LED.", "metadata": {"product_name": "BrightVision Smart Lamp"}},
    {"id": "p5", "text": "Need help with your order? Our customer support is available 24/7.", "metadata": {"type": "support"}},
    {"id": "p6", "text": "Shipping usually takes 3-5 business days within the country.", "metadata": {"type": "shipping"}},
]

if kb_collection.count() == 0:
    st.info("Initializing knowledge base...")
    documents = [item["text"] for item in kb_data]
    metadatas = [item["metadata"] for item in kb_data]
    ids = [item["id"] for item in kb_data]
    embeddings = embedding_model.encode(documents).tolist()

    kb_collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    st.success("Knowledge base initialized with sample data.")
else:
    st.info("Knowledge base already populated.")

@st.cache_resource
def load_llm_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
text_generator = load_llm_pipeline()

@st.cache_resource
def load_translator(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    return pipeline("translation", model=model_name, tokenizer=model_name)

supported_langs = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
}

in_context_examples = [
    {
        "query_fr": "Quel café est le meilleur pour la durabilité ?",
        "kb_info_en": "The 'EcoBlend' coffee maker brews delicious coffee with sustainable pods. Features a 1.2L capacity and programmable timer.",
        "response_fr": "Le 'EcoBlend' est excellent pour la durabilité car il utilise des dosettes durables."
    },
    {
        "query_fr": "Je veux des écouteurs avec une bonne autonomie de batterie.",
        "kb_info_en": "Our 'ComfyCloud' wireless headphones offer superior sound quality and 20 hours of battery life. Ergonomic design for maximum comfort.",
        "response_fr": "Les écouteurs 'ComfyCloud' ont une excellente autonomie de batterie de 20 heures."
    },
    {
        "query_es": "¿Qué batidora recomiendan para repostería?",
        "kb_info_en": "The 'GourmetPro' stand mixer is perfect for baking enthusiasts. Comes with multiple attachments for kneading, whisking, and mixing.",
        "response_es": "Para repostería, recomendamos la batidora de pie 'GourmetPro' con sus múltiples accesorios."
    }
]

def get_chatbot_response(user_query: str):
    try:
        detected_lang = detect(user_query)
        st.write(f"Detected Language: {supported_langs.get(detected_lang, detected_lang)}")
    except:
        detected_lang = "en"
        st.write("Could not detect language, defaulting to English.")

    if detected_lang not in supported_langs:
        return f"Désolé, je ne supporte pas encore le langage '{detected_lang}'. Je peux répondre en Anglais, Français ou Espagnol."

    query_in_english = user_query
    if detected_lang != "en":
        try:
            translator_l1_en = load_translator(detected_lang, "en")
            query_in_english = translator_l1_en(user_query)[0]["translation_text"]
            st.write(f"Translated query to English for KB: {query_in_english}")
        except Exception as e:
            st.warning(f"Could not translate query to English for KB lookup ({e}). Using original query.")
            query_in_english = user_query

    query_embedding = embedding_model.encode(query_in_english).tolist()
    results = kb_collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["documents", "metadatas"]
    )

    retrieved_info_en = "No relevant information found."
    if results["documents"] and results["documents"][0]:
        retrieved_info_en = results["documents"][0][0]
        st.write(f"Retrieved KB info (EN): {retrieved_info_en}")

    prompt_parts = [
        "You are a helpful multilingual customer support assistant.",
        "Answer the user's question concisely in their original language, using the provided product information.",
        "If the information is not relevant, state that you cannot help with that specific query.",
        "\n--- Examples ---"
    ]

    for example in in_context_examples:
        if f"query_{detected_lang}" in example:
            prompt_parts.append(
                f"User ({detected_lang}): {example[f'query_{detected_lang}']}\n"
                f"Product Info (EN): {example['kb_info_en']}\n"
                f"Assistant ({detected_lang}): {example[f'response_{detected_lang}']}\n"
            )
        elif detected_lang == "fr" and "query_fr" in example:
             prompt_parts.append(
                f"User (fr): {example['query_fr']}\n"
                f"Product Info (EN): {example['kb_info_en']}\n"
                f"Assistant (fr): {example['response_fr']}\n"
            )
        elif detected_lang == "es" and "query_es" in example:
             prompt_parts.append(
                f"User (es): {example['query_es']}\n"
                f"Product Info (EN): {example['kb_info_en']}\n"
                f"Assistant (es): {example['response_es']}\n"
            )

    prompt_parts.append("\n--- Current Query ---")
    prompt_parts.append(f"User ({detected_lang}): {user_query}")
    prompt_parts.append(f"Product Info (EN): {retrieved_info_en}")
    prompt_parts.append(f"Assistant ({detected_lang}):")

    final_prompt = "\n".join(prompt_parts)
    st.write("--- Generated Prompt ---")
    st.text(final_prompt)

    try:
        llm_output = text_generator(final_prompt, max_new_tokens=100, num_return_sequences=1)
        response = llm_output[0]["generated_text"].strip()

        if detected_lang != "en" and detect(response) == "en" and "Product Info (EN):" in final_prompt:
             st.warning("LLM generated response in English, attempting translation back to user's language.")
             try:
                 translator_en_l1 = load_translator("en", detected_lang)
                 response = translator_en_l1(response)[0]["translation_text"]
             except Exception as e:
                 st.error(f"Failed to translate LLM response back to {detected_lang}: {e}")
                 response = response

        return response
    except Exception as e:
        st.error(f"Error during LLM response generation: {e}")
        return "Désolé, je rencontre un problème technique. Veuillez réessayer plus tard."

st.title("Multilingual E-commerce Chatbot (InCLT Prompting Demo)")
st.subheader("Boosting cross-lingual capabilities with In-Context Learning Transfer")

st.write("Ask questions about products in French, Spanish, or English.")
st.write("Example: 'Quel café est le meilleur pour la durabilité ?' (French)")
st.write("Example: 'Quiero unos auriculares con buena batería.' (Spanish)")
st.write("Example: 'What is the EcoBlend coffee maker?' (English)")

user_input = st.text_input("Your question:")

if user_input:
    with st.spinner("Thinking..."):
        chatbot_response = get_chatbot_response(user_input)
    st.markdown(f"**Chatbot:** {chatbot_response}")