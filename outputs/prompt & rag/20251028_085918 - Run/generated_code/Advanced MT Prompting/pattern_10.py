import streamlit as st
from langdetect import detect, DetectorFactory
from transformers import pipeline, MarianMTModel, MarianTokenizer
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Ensure reproducibility for langdetect
DetectorFactory.seed = 0

# --- Configuration and Model Loading ---

# High-resource language for internal processing
HIGH_RESOURCE_LANG = "en"

# Translation Models (loaded once)
@st.cache_resource
def load_translation_model(src_lang, dest_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{dest_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Load English to other languages (for response translation back)
translation_pipelines = {}

@st.cache_resource
def get_translator_pipeline(src_lang, dest_lang):
    if (src_lang, dest_lang) not in translation_pipelines:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{dest_lang}"
        translation_pipelines[(src_lang, dest_lang)] = pipeline("translation", model=model_name, tokenizer=model_name)
    return translation_pipelines[(src_lang, dest_lang)]

# Embedding Model (for ChromaDB retrieval)
@st.cache_resource
def load_embedding_model():
    # Using a multilingual embedding model
    model_name = "intfloat/multilingual-e5-large"
    # Wrap in HuggingFaceBgeEmbeddings for LangChain compatibility
    return HuggingFaceBgeEmbeddings(model_name=model_name)

embeddings = load_embedding_model()

# LLM for response generation (using a small, local-friendly model for demonstration)
@st.cache_resource
def load_llm_pipeline():
    # For demonstration, using a small, instruction-tuned model. 
    # In production, consider a larger Llama/Mistral, or API-based LLMs.
    try:
        llm_pipeline = pipeline(
            "text-generation",
            model="distilgpt2", # A very small model for fast local demo. Replace with larger for better quality.
            torch_dtype=torch.bfloat16, # Use bfloat16 if your GPU supports it, otherwise float16 or float32
            device=0 if torch.cuda.is_available() else -1, # Use GPU if available
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        return HuggingFacePipeline(pipeline=llm_pipeline)
    except Exception as e:
        st.error(f"Error loading LLM: {e}. Please ensure `distilgpt2` model is available or choose another.")
        st.info("If running on CPU, `torch_dtype=torch.float32` might be necessary.")
        return None

llm = load_llm_pipeline()

# --- Knowledge Base and Glossary ---

# Sample Knowledge Base Content
KNOWLEDGE_BASE_DOCS = [
    {"id": "prod_001", "content": "Our latest smartphone, the 'Zenith X', features a 108MP camera, 5G connectivity, and a 5000mAh battery. It comes in Midnight Black and Aurora Silver.", "metadata": {"product": "Zenith X"}},
    {"id": "faq_warranty", "content": "All electronic products come with a 1-year limited warranty covering manufacturing defects. Physical damage or water damage is not covered.", "metadata": {"topic": "Warranty"}},
    {"id": "faq_shipping", "content": "Standard shipping takes 5-7 business days. Express shipping is available for an additional charge and takes 2-3 business days. We ship internationally to most countries.", "metadata": {"topic": "Shipping"}},
    {"id": "return_policy", "content": "You can return items within 30 days of purchase, provided they are in their original packaging and condition. Refunds are processed within 7-10 business days.", "metadata": {"topic": "Returns"}},
    {"id": "prod_002", "content": "The 'Aero Buds Pro' offer active noise cancellation, 24-hour battery life with the charging case, and IPX4 water resistance. Perfect for workouts and daily commutes.", "metadata": {"product": "Aero Buds Pro"}}
]

# Domain-specific Glossary (English to various languages)
GLOSSARY = {
    "smartphone": {"es": "teléfono inteligente", "fr": "téléphone intelligent", "de": "Smartphone"},
    "warranty": {"es": "garantía", "fr": "garantie", "de": "Garantie"},
    "shipping": {"es": "envío", "fr": "expédition", "de": "Versand"},
    "refund": {"es": "reembolso", "fr": "remboursement", "de": "Rückerstattung"},
    "camera": {"es": "cámara", "fr": "appareil photo", "de": "Kamera"},
    "battery": {"es": "batería", "fr": "batterie", "de": "Batterie"},
    "noise cancellation": {"es": "cancelación de ruido", "fr": "annulation du bruit", "de": "Geräuschunterdrückung"}
}

# ChromaDB Client and Collection
CHROMA_PERSIST_DIR = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

@st.cache_resource
def setup_vector_store(documents, embeddings):
    try:
        # Try to get existing collection or create a new one
        collection_name = "customer_support_kb"
        # Ensure the collection is reset if we want to re-index during development
        # client.delete_collection(name=collection_name) # Uncomment for full reset

        vectordb = Chroma.from_documents(
            documents=[
                LangChainDocument(page_content=doc['content'], metadata=doc['metadata'])
                for doc in documents
            ],
            embedding=embeddings,
            client=client,
            collection_name=collection_name,
            persist_directory=CHROMA_PERSIST_DIR
        )
        vectordb.persist()
        return vectordb
    except Exception as e:
        st.error(f"Error setting up vector store: {e}")
        return None

from langchain_core.documents import Document as LangChainDocument # Renamed to avoid conflict
vectorstore = setup_vector_store(KNOWLEDGE_BASE_DOCS, embeddings)
if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Chatbot Functions ---

def translate_to_high_resource(text, src_lang, dest_lang=HIGH_RESOURCE_LANG):
    if src_lang == dest_lang:
        return text
    try:
        translator = get_translator_pipeline(src_lang, dest_lang)
        translated = translator(text, max_length=512)[0]['translation_text']
        return translated
    except Exception as e:
        st.warning(f"Could not translate from {src_lang} to {dest_lang}: {e}. Using original text.")
        return text

def translate_from_high_resource(text, dest_lang, src_lang=HIGH_RESOURCE_LANG):
    if src_lang == dest_lang:
        return text
    try:
        translator = get_translator_pipeline(src_lang, dest_lang)
        translated = translator(text, max_length=512)[0]['translation_text']
        return translated
    except Exception as e:
        st.warning(f"Could not translate from {src_lang} to {dest_lang}: {e}. Using original text.")
        return text

def get_glossary_terms_for_lang(lang):
    relevant_terms = []
    for english_term, translations in GLOSSARY.items():
        if lang in translations:
            relevant_terms.append(f"{english_term} ({translations[lang]})")
    return ", ".join(relevant_terms) if relevant_terms else ""

def analyze_query_complexity(query: str) -> bool:
    # Simple heuristic: check for common conjunctions or length
    complex_keywords = ["and", "or", "but", "how to", "what about"]
    if any(keyword in query.lower() for keyword in complex_keywords) or len(query.split()) > 15:
        return True
    return False

def self_assess_response(response: str, query: str) -> bool:
    # Very basic self-assessment: check if the response is not empty and seems to address the query
    if not response or len(response.strip()) < 10:
        return False
    if "I cannot answer" in response or "I don't know" in response: # Placeholder for LLM refusal
        return False
    # More sophisticated assessment would involve another LLM call or semantic similarity check
    return True

# LangChain RAG Setup
if llm and retriever:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful customer support assistant for an e-commerce platform. Answer the user's question based on the provided context and glossary. If you don't know the answer, politely state that you cannot provide it. \n\nContext: {context}\n\nGlossary: {glossary_terms}"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- Streamlit UI ---

st.set_page_config(page_title="Multilingual E-commerce Chatbot", layout="wide")
st.title("🌍 Multilingual Contextual Customer Support Chatbot")
st.markdown("This chatbot uses context augmentation, strategic planning (simplified), and iterative feedback (simulated) to assist users in multiple languages.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question in any language...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # 1. Language Detection
            detected_lang = detect(user_query)
            st.info(f"Detected Language: {detected_lang}")

            # 2. Translate to High-Resource Language for Processing
            translated_query_for_llm = translate_to_high_resource(user_query, detected_lang, HIGH_RESOURCE_LANG)
            st.info(f"Translated query for processing ({HIGH_RESOURCE_LANG}): {translated_query_for_llm}")

            # 3. Context Augmentation: Glossary
            glossary_for_context = get_glossary_terms_for_lang(detected_lang)
            if glossary_for_context:
                st.info(f"Relevant Glossary terms for {detected_lang}: {glossary_for_context}")

            # 4. Strategic Planning & Decomposition (Simplified)
            if analyze_query_complexity(translated_query_for_llm):
                st.warning("Query identified as potentially complex. Attempting decomposition (simplified).")
                # In a real scenario, this would involve breaking down the query into sub-queries
                # For this demo, we'll just emphasize the strategic approach in the prompt.
                strategic_prompt_addition = "Break down complex parts if necessary and provide a comprehensive answer."
            else:
                strategic_prompt_addition = ""
            
            if llm and rag_chain:
                # Generate response using RAG chain with augmented context
                response_obj = rag_chain.invoke({
                    "input": translated_query_for_llm + "\n\n" + strategic_prompt_addition,
                    "glossary_terms": glossary_for_context # Pass glossary directly
                })
                llm_response_english = response_obj["answer"]
                st.info(f"LLM Raw Response (English): {llm_response_english}")

                # 5. Automated Self-Assessment
                if not self_assess_response(llm_response_english, translated_query_for_llm):
                    st.warning("Automated self-assessment failed: Response seems inadequate. Attempting re-generation or flagging.")
                    llm_response_english = "I apologize, I'm having difficulty providing a comprehensive answer to that query at the moment. Would you like to rephrase it or provide more details?"
                
                # 6. Translate Response back to User's Language
                final_response = translate_from_high_resource(llm_response_english, detected_lang, HIGH_RESOURCE_LANG)
                full_response = final_response
            else:
                full_response = "Error: Chatbot core components (LLM or RAG chain) are not initialized. Please check logs."

        except Exception as e:
            full_response = f"An error occurred: {e}. Please try again."
            st.error(f"Chatbot error: {e}")

        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # 7. Human-in-the-Loop (HITL) Placeholder
        st.sidebar.subheader("Feedback (Human-in-the-Loop Simulation)")
        feedback = st.sidebar.radio("Was this answer helpful?", ("Yes", "No", "N/A"), key=f"feedback_{len(st.session_state.messages)}")
        if feedback != "N/A":
            st.sidebar.info(f"Feedback recorded: {feedback}. This data would be used for future model improvement.")
            # In a real system, log this feedback to a database for model fine-tuning or prompt engineering

else:
    st.info("Welcome! Ask me anything about our products or policies.")

