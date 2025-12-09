import streamlit as st
from langdetect import detect, DetectorFactory
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import chromadb
import random

# Ensure consistent language detection
DetectorFactory.seed = 0

# --- Configuration --- 
# For demonstration, using smaller models. For production, consider larger models.
TRANSLATION_MODEL_EN_ES = "Helsinki-NLP/opus-mt-en-es"
TRANSLATION_MODEL_ES_EN = "Helsinki-NLP/opus-mt-es-en"
TRANSLATION_MODEL_EN_FR = "Helsinki-NLP/opus-mt-en-fr"
TRANSLATION_MODEL_FR_EN = "Helsinki-NLP/opus-mt-fr-en"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Placeholder for LLM - In a real scenario, this would be an API call or a loaded model
# For simplicity, we'll mock the LLM response. 
# If using a local LLM, you would load it here, e.g., from transformers library.
# LLM_MODEL_NAME = "path/to/your/multilingual-llm"

# --- Initialize Models and DB (cached for Streamlit) ---
@st.cache_resource
def load_translation_pipelines():
    return {
        "en-es": pipeline("translation", model=TRANSLATION_MODEL_EN_ES, tokenizer=TRANSLATION_MODEL_EN_ES),
        "es-en": pipeline("translation", model=TRANSLATION_MODEL_ES_EN, tokenizer=TRANSLATION_MODEL_ES_EN),
        "en-fr": pipeline("translation", model=TRANSLATION_MODEL_EN_FR, tokenizer=TRANSLATION_MODEL_EN_FR),
        "fr-en": pipeline("translation", model=TRANSLATION_MODEL_FR_EN, tokenizer=TRANSLATION_MODEL_FR_EN),
    }

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def setup_chromadb():
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="customer_support_kb")
    
    # Dummy Knowledge Base Articles (primarily English)
    kb_articles = [
        {"id": "doc1", "content": "Our shipping policy states that orders are processed within 2-3 business days. Delivery times vary by region, typically 5-7 days for international shipping.", "metadata": {"lang": "en", "topic": "shipping"}},
        {"id": "doc2", "content": "Returns are accepted within 30 days of purchase, provided the item is in its original condition with proof of purchase. Refunds are issued to the original payment method.", "metadata": {"lang": "en", "topic": "returns"}},
        {"id": "doc3", "content": "Para devolver un artículo, debe estar en su estado original y con el comprobante de compra. Los reembolsos se realizan al método de pago original.", "metadata": {"lang": "es", "topic": "returns"}},
        {"id": "doc4", "content": "You can track your order using the tracking number provided in your shipping confirmation email.", "metadata": {"lang": "en", "topic": "tracking"}},
        {"id": "doc5", "content": "Pour suivre votre commande, utilisez le numéro de suivi fourni dans l'e-mail de confirmation d'expédition.", "metadata": {"lang": "fr", "topic": "tracking"}},
        {"id": "doc6", "content": "We offer 24/7 customer support via chat and email. Our agents are available to assist you with any inquiries.", "metadata": {"lang": "en", "topic": "support"}},
    ]
    
    # Add articles to ChromaDB if not already present
    existing_ids = set(collection.get(ids=[article["id"] for article in kb_articles])["ids"])
    articles_to_add = [article for article in kb_articles if article["id"] not in existing_ids]

    if articles_to_add:
        contents = [a["content"] for a in articles_to_add]
        metadatas = [a["metadata"] for a in articles_to_add]
        ids = [a["id"] for a in articles_to_add]
        embeddings = embedding_model.encode(contents).tolist()
        collection.add(embeddings=embeddings, documents=contents, metadatas=metadatas, ids=ids)
    
    return collection

translation_pipelines = load_translation_pipelines()
embedding_model = load_embedding_model()
kb_collection = setup_chromadb()

# --- Language Detection Module ---
def detect_query_language(text):
    try:
        return detect(text)
    except:
        return "en" # Default to English if detection fails

# --- Translation Service ---
def translate_text(text, source_lang, target_lang):
    key = f"{source_lang}-{target_lang}"
    if key in translation_pipelines:
        # The pipeline returns a list of dictionaries, we need the 'translation_text' from the first element
        translated = translation_pipelines[key](text)[0]['translation_text']
        return translated
    return text # Return original text if no translator found

# --- RAG Module ---
def retrieve_kb_articles(query, top_k=3):
    query_embedding = embedding_model.encode([query]).tolist()
    results = kb_collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=['documents', 'metadatas']
    )
    return results['documents'][0] if results['documents'] else []

# --- InCLT Prompting Engine ---
def create_inclt_examples(retrieved_articles, user_query, user_query_lang, llm_target_lang="en"):
    inclt_examples = []
    for i, article_content in enumerate(retrieved_articles):
        # For simplicity, we'll assume retrieved articles are mostly in English for this example
        # In a real system, you might detect article language or store it in metadata
        article_lang = detect_query_language(article_content) # Assuming we might retrieve non-English too
        
        # If the article is not in the user's query language, and not in the LLM's target language (e.g., English),
        # we want to create cross-lingual examples.

        # Example 1: Original Article (EN) -> Translated Query (ES) -> LLM's expected response if it were to answer in EN based on EN article
        # This helps the LLM understand the cross-lingual intent related to the EN article.
        if user_query_lang != article_lang and article_lang == llm_target_lang:
            translated_query = translate_text(user_query, user_query_lang, llm_target_lang)
            inclt_examples.append(
                f"CONTEXT (English): {article_content}\n"
                f"QUERY (Original {user_query_lang.upper()}): {user_query}\n"
                f"QUERY (Translated {llm_target_lang.upper()} for LLM): {translated_query}\n"
                f"EXPECTED TASK (Understand {user_query_lang.upper()} based on {llm_target_lang.upper()} context and respond in {llm_target_lang.upper()}):"
            )
        # Example 2: Article in target language for LLM (EN) -> LLM's expected response in target lang (EN)
        # This provides a direct example of relevant info.
        elif article_lang == llm_target_lang:
            inclt_examples.append(
                f"CONTEXT (English): {article_content}\n"
                f"QUERY (English): How does this article address \"{user_query}\"?\n"
                f"EXPECTED ANSWER (English): The article discusses..."
            )

    # Add a final instruction for the LLM based on the user's query language
    final_instruction_translated_query = translate_text(user_query, user_query_lang, llm_target_lang)
    inclt_examples.append(
        f"Given the above contexts and examples, please answer the following question.\n"
        f"Original Query ({user_query_lang.upper()}): {user_query}\n"
        f"Question for me ({llm_target_lang.upper()}): {final_instruction_translated_query}\n"
        f"Answer in {llm_target_lang.upper()}:"
    )

    return "\n\n".join(inclt_examples)

# --- Multilingual LLM (Mock) ---
def call_llm(prompt):
    # This is a mock LLM. In a real application, you'd integrate with an actual LLM.
    # For instance, using Hugging Face's transformers for a local model, or OpenAI API.
    
    # Simple keyword-based response for demonstration
    prompt_lower = prompt.lower()
    if "shipping" in prompt_lower or "delivery" in prompt_lower:
        response = "Based on our shipping policy, orders are processed within 2-3 business days. International delivery typically takes 5-7 days."
    elif "return" in prompt_lower or "refund" in prompt_lower:
        response = "Our return policy allows returns within 30 days if the item is in its original condition with proof of purchase. Refunds are issued to the original payment method."
    elif "track" in prompt_lower:
        response = "You can track your order using the tracking number sent in your shipping confirmation email."
    elif "support" in prompt_lower:
        response = "We offer 24/7 customer support via chat and email."
    else:
        response = "I'm sorry, I couldn't find a direct answer to your question in the knowledge base. Can you please rephrase or provide more details?"

    return response

# --- Orchestration (Main Chatbot Logic) ---
def chatbot_response(user_query):
    user_query_lang = detect_query_language(user_query)
    st.sidebar.write(f"Detected input language: {user_query_lang.upper()}")

    retrieved_articles = retrieve_kb_articles(user_query)
    st.sidebar.write("Retrieved KB Articles:")
    for i, doc in enumerate(retrieved_articles):
        st.sidebar.write(f"- {doc[:100]}...")

    # LLM will primarily process in English (common for many multilingual LLMs for RAG)
    llm_processing_lang = "en"
    
    inclt_prompt = create_inclt_examples(
        retrieved_articles=retrieved_articles,
        user_query=user_query,
        user_query_lang=user_query_lang,
        llm_target_lang=llm_processing_lang
    )
    st.sidebar.subheader("InCLT Prompt to LLM:")
    st.sidebar.code(inclt_prompt)

    llm_raw_response = call_llm(inclt_prompt) # Pass the crafted prompt to the LLM
    st.sidebar.subheader("LLM Raw Response (English):")
    st.sidebar.write(llm_raw_response)

    final_response = llm_raw_response
    # Translate the LLM's response back to the user's original language if needed
    if user_query_lang != llm_processing_lang:
        final_response = translate_text(llm_raw_response, llm_processing_lang, user_query_lang)
    
    return final_response

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Multilingual Chatbot with InCLT")
st.title("🌍 Multilingual Customer Support Chatbot (with InCLT Prompting)")
st.markdown("This chatbot demonstrates `InCLT Crosslingual Transfer Prompting` to enhance cross-lingual understanding in customer support.")

user_input = st.text_input("Ask a question in your preferred language:", "¿Cómo puedo rastrear mi pedido?")

if user_input:
    with st.spinner("Processing your request..."):
        response = chatbot_response(user_input)
        st.subheader("Chatbot Response:")
        st.write(response)
else:
    st.info("Type a question above to get started!")

st.sidebar.header("Chatbot Internals")
st.sidebar.info("Details about language detection, retrieved articles, and the generated InCLT prompt will appear here.")