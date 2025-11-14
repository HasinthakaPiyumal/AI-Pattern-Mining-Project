import streamlit as st
import os
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# Ensure reproducibility for langdetect
DetectorFactory.seed = 0

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 1. Load NLTK data (if not already downloaded) ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- 2. Initialize Models and Components ---

# a. Initial Machine Translation (MT) Model
@st.cache_resource
def load_mt_model():
    # Using a smaller model for demonstration; for production, consider larger models or specific language pairs
    model_name = "Helsinki-NLP/opus-mt-en-es" # English to Spanish, for example
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("translation", model=model, tokenizer=tokenizer)

mt_translator = load_mt_model()

# b. Embedding Model for RAG
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# c. ChromaDB for Medical Exemplars (RAG)
@st.cache_resource
def init_chromadb():
    client = chromadb.Client()
    try:
        collection = client.get_or_create_collection(name="medical_exemplars")
    except Exception as e:
        st.error(f"Error initializing ChromaDB collection: {e}")
        st.info("Attempting to reset and recreate collection...")
        client.delete_collection(name="medical_exemplars")
        collection = client.get_or_create_collection(name="medical_exemplars")

    # Add dummy medical exemplars if the collection is empty
    if collection.count() == 0:
        exemplars = [
            {"text": "Chest pain radiating to the left arm is a common symptom of myocardial infarction.", "translation": "El dolor en el pecho que se irradia al brazo izquierdo es un síntoma común de infarto de miocardio.", "id": "ex1"},
            {"text": "Take two tablets orally three times a day after meals.", "translation": "Tome dos tabletas por vía oral tres veces al día después de las comidas.", "id": "ex2"},
            {"text": "The patient presents with dyspnea, fever, and a persistent cough.", "translation": "El paciente presenta disnea, fiebre y tos persistente.", "id": "ex3"},
            {"text": "Regular monitoring of blood pressure is crucial for hypertension management.", "translation": "El control regular de la presión arterial es crucial para el manejo de la hipertensión.", "id": "ex4"},
            {"text": "Diagnosed with Type 2 Diabetes Mellitus, requiring insulin therapy.", "translation": "Diagnosticado con Diabetes Mellitus tipo 2, que requiere terapia con insulina.", "id": "ex5"}
        ]
        texts = [ex["text"] for ex in exemplars]
        metadatas = [{
            "original_text": ex["text"],
            "translated_text": ex["translation"]
        } for ex in exemplars]
        ids = [ex["id"] for ex in exemplars]
        embeddings = embedding_model.encode(texts).tolist()
        collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
        st.success("Medical exemplars added to ChromaDB.")
    return collection

chroma_collection = init_chromadb()

# d. Medical Dictionary (dummy for demo)
medical_dictionary = {
    "myocardial infarction": "infarto de miocardio",
    "dyspnea": "disnea",
    "hypertension": "hipertensión",
    "diabetes mellitus": "diabetes mellitus",
    "diagnosis": "diagnóstico",
    "therapy": "terapia",
    "oral": "oral",
    "tablets": "tabletas",
    "fever": "fiebre",
    "cough": "tos"
}

# e. OpenAI LLM
@st.cache_resource
def load_openai_llm():
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found in environment variables. Please set it.")
        return None
    return ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4-turbo-preview", temperature=0.2)

llm = load_openai_llm()

# --- 3. LangChain Prompt Templates (Simplified Agents) ---

analysis_prompt = ChatPromptTemplate.from_messages([
    SystemMessage("You are an expert medical text analyzer. Your task is to break down complex medical texts, identify key medical terms, concepts, and potential ambiguities relevant for translation."),
    HumanMessage(content="Analyze the following medical text for key terms and concepts for translation: {text}")
])

translation_prompt = ChatPromptTemplate.from_messages([
    SystemMessage("You are a highly accurate medical translator, specializing in translating medical documents from {source_lang} to {target_lang}. Use all provided context to ensure precision and nuance. \n\nMedical Dictionary Terms: {medical_terms}\n\nRelevant Translation Examples: {exemplars}"),
    HumanMessage(content="Translate the following medical text. Maintain medical accuracy and professional tone.\n\nOriginal Text: {original_text}\n\nInitial Machine Translation (if available): {initial_mt}")
])

verification_prompt = ChatPromptTemplate.from_messages([
    SystemMessage("You are a medical translation verifier. Your role is to critically review a translated medical text against its original, checking for accuracy, consistency, and potential misinterpretations. Suggest improvements if necessary."),
    HumanMessage(content="Original Text: {original_text}\n\nTranslated Text: {translated_text}\n\nReview the translated text for medical accuracy and suggest any necessary corrections or refinements.")
])

# --- 4. Helper Functions ---
def get_medical_term_context(text):
    found_terms = []
    for term in medical_dictionary:
        if term in text.lower():
            found_terms.append(f"{term} (translation: {medical_dictionary[term]})")
    return "; ".join(found_terms) if found_terms else "No specific medical terms found in dictionary."

def retrieve_medical_exemplars(query_text, num_results=2):
    if not chroma_collection:
        return ""
    try:
        query_embedding = embedding_model.encode(query_text).tolist()
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results,
            include=['metadatas']
        )
        exemplars = []
        if results and results['metadatas'] and results['metadatas'][0]:
            for metadata in results['metadatas'][0]:
                exemplars.append(f"Original: '{metadata.get('original_text', '')}' -> Translated: '{metadata.get('translated_text', '')}'")
        return "\n".join(exemplars) if exemplars else "No relevant exemplars found."
    except Exception as e:
        st.error(f"Error retrieving exemplars from ChromaDB: {e}")
        return "No relevant exemplars found due to an error."


def translate_text_pipeline(original_text, source_lang, target_lang):
    if llm is None:
        st.warning("LLM not initialized. Please check your OpenAI API key.")
        return "LLM not available", "", "", ""

    # LangDetect for source language if not provided
    if source_lang == "auto":
        try:
            source_lang = detect(original_text)
            st.info(f"Detected source language: {source_lang}")
        except:
            source_lang = "en" # Default to English if detection fails
            st.warning("Could not detect source language, defaulting to English.")

    # 1. Augmented Prompting & Preprocessing
    st.subheader("Step 1: Augmented Prompting & Preprocessing")

    # Initial MT (if target_lang is different from MT model's target)
    initial_mt_output = ""
    try:
        # The `mt_translator` is loaded for a specific pair, e.g., en-es. Adjust logic if needed.
        if source_lang == "en" and target_lang == "es": # Example specific to Helsinki-NLP opus-mt-en-es
            initial_mt_output = mt_translator(original_text)[0]['translation_text']
            st.write(f"Initial MT ({source_lang} -> {target_lang}):", initial_mt_output)
        else:
             st.info(f"Skipping direct initial MT for {source_lang} -> {target_lang} as model is not configured for this specific pair. Relying on LLM.")

    except Exception as e:
        st.warning(f"Could not perform initial MT: {e}. Relying solely on LLM.")

    # Medical Terminology Lookup
    medical_term_context = get_medical_term_context(original_text)
    st.write("Identified Medical Terms:", medical_term_context)

    # Exemplar Retrieval (RAG)
    relevant_exemplars = retrieve_medical_exemplars(original_text)
    st.write("Relevant Medical Exemplars (RAG):")
    st.text(relevant_exemplars)

    # 2. Strategic Planning & Decomposition (via Analysis Agent concept)
    st.subheader("Step 2: Strategic Planning & Decomposition (Analysis)")
    analysis_output = analysis_prompt.format_messages(text=original_text)
    analysis_response = llm(analysis_output).content
    st.write("Analysis Agent Output:", analysis_response)

    # Construct Augmented Prompt for Translation
    augmented_prompt_content = (
        f"Original Source Language: {source_lang}\n"
        f"Target Language: {target_lang}\n"
        f"Analyzed Concepts: {analysis_response}\n"
        f"Medical Dictionary Context: {medical_term_context}\n"
        f"Retrieved Exemplars: {relevant_exemplars}\n"
        f"Original Text to Translate: {original_text}\n"
        f"Initial Machine Translation (if available): {initial_mt_output}\n"
    )
    st.subheader("Augmented Prompt Sent to Translation LLM")
    st.text(augmented_prompt_content)

    # Translation Agent (LLM Call)
    st.subheader("Step 3: Generating Draft Translation")
    translation_messages = translation_prompt.format_messages(
        source_lang=source_lang,
        target_lang=target_lang,
        medical_terms=medical_term_context,
        exemplars=relevant_exemplars,
        original_text=original_text,
        initial_mt=initial_mt_output
    )
    draft_translation = llm(translation_messages).content
    st.write("Draft Translation:", draft_translation)

    # Verification Agent (LLM Call for self-correction)
    st.subheader("Step 4: Verification and Refinement (LLM Self-Correction)")
    verification_messages = verification_prompt.format_messages(
        original_text=original_text,
        translated_text=draft_translation
    )
    verification_response = llm(verification_messages).content
    st.write("Verification Agent Feedback:", verification_response)

    # Incorporate verification feedback into a final draft suggestion
    final_llm_suggested_translation = draft_translation + "\n\n(LLM Review Feedback: " + verification_response + ")" if "suggest improvement" in verification_response.lower() else draft_translation

    return initial_mt_output, augmented_prompt_content, draft_translation, final_llm_suggested_translation


# --- 5. Streamlit UI ---
st.set_page_config(layout="wide", page_title="Multilingual Medical Translator")
st.title("🩺 Multilingual Medical Information Translator")
st.markdown("Enhance medical translation quality using augmented prompting, strategic planning, and human-in-the-loop refinement.")

# Input Section
st.header("Input Medical Text")
input_text = st.text_area("Enter medical text here:", height=200, value="The patient was admitted with severe chest pain and shortness of breath. Diagnosis: Myocardial infarction.")

col1, col2 = st.columns(2)
with col1:
    source_language = st.selectbox("Source Language", ["auto", "en", "es", "fr", "de"], index=0)
with col2:
    target_language = st.selectbox("Target Language", ["es", "en", "fr", "de"], index=0)

if st.button("Translate Medical Text"): 
    if not input_text:
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating and refining..."):
            initial_mt, augmented_prompt, draft_translation, llm_final_suggestion = translate_text_pipeline(input_text, source_language, target_language)

        st.markdown("--- ")
        st.header("Human-in-the-Loop Refinement")
        st.info("Review the LLM's suggested translation below. Make any necessary edits for final accuracy.")

        final_translation_output = st.text_area(
            "Final Translation (Edit if needed):",
            value=llm_final_suggestion, # Pre-fill with LLM's best suggestion
            height=300
        )

        if st.button("Finalize Translation"):
            st.success("Translation finalized!")
            st.download_button(
                label="Download Final Translation",
                data=final_translation_output.encode("utf-8"),
                file_name="final_medical_translation.txt",
                mime="text/plain"
            )
            st.session_state["final_output"] = final_translation_output

        if "final_output" in st.session_state:
            st.subheader("Your Final Translated Output:")
            st.markdown(f"```\n{st.session_state['final_output']}\n```")



