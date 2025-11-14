
import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import threading
import os
import sqlite3
import json
import uuid
from datetime import datetime
import requests # For Streamlit to call FastAPI

# NLP/ML Libraries
# For demonstration purposes, some heavy models are replaced with dummy fallbacks
# or require manual download/installation as specified in comments.
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
# import fasttext # Uncomment and download model if using robust language detection
# import chromadb # Uncomment if using a persistent ChromaDB
# from sentence_transformers import SentenceTransformer # Uncomment if using sentence-transformers for embeddings

# --- Configuration ---+
DATABASE_FILE = "medical_translator_feedback.db"
CHROMA_PATH = "medical_knowledge_db" # Path for persistent ChromaDB (if enabled)
PIVOT_LANGUAGE_CODE = "eng_Latn" # NLLB-200 code for English, used as a robust intermediate if needed

# Main NMT model. This is a large model. For quick testing, consider a smaller one or a dummy fallback.
# Download model if not cached: `from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M"); AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")`
NMT_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Sentence Transformer model for embedding medical terms
# Download model: `from sentence_transformers import SentenceTransformer; SentenceTransformer("all-MiniLM-L6-v2")`
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# SpaCy model for text decomposition and sentence segmentation
# Download model: `python -m spacy download en_core_web_sm`
SPACY_MODEL_NAME = "en_core_web_sm"

# --- Database Setup (SQLite) ---
class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self._create_tables()

    def _create_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS translations (
                id TEXT PRIMARY KEY,
                original_text TEXT NOT NULL,
                source_language TEXT NOT NULL,
                initial_translation TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                translation_id TEXT,
                corrected_translation TEXT NOT NULL,
                feedback_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (translation_id) REFERENCES translations(id)
            )
        """)
        conn.commit()
        conn.close()

    def save_translation(self, original_text, source_language, initial_translation):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        translation_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO translations (id, original_text, source_language, initial_translation) VALUES (?, ?, ?, ?)",
            (translation_id, original_text, source_language, initial_translation)
        )
        conn.commit()
        conn.close()
        return translation_id

    def update_feedback(self, translation_id, corrected_translation):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        feedback_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO feedback (feedback_id, translation_id, corrected_translation) VALUES (?, ?, ?)",
            (feedback_id, translation_id, corrected_translation)
        )
        conn.commit()
        conn.close()

db_manager = DatabaseManager(DATABASE_FILE)

# --- AI Model Initialization ---
# Language Detection Model (FastText placeholder)
LID_MODEL = None # Set to None if fasttext is not installed/model not downloaded
# try:
#     import fasttext
#     # Download 'lid.176.bin' from https://fasttext.cc/docs/en/language-identification.html
#     LID_MODEL = fasttext.load_model('lid.176.bin')
#     print("FastText language identification model loaded.")
# except (ImportError, ValueError):
#     print("FastText language identification model not found. Using dummy detection.")

# NMT Model (Hugging Face Transformers - NLLB-200)
NMT_TOKENIZER = None
NMT_MODEL = None
NMT_PIPELINE = None
print(f"Loading NMT model: {NMT_MODEL_NAME}...")
try:
    NMT_TOKENIZER = AutoTokenizer.from_pretrained(NMT_MODEL_NAME)
    NMT_MODEL = AutoModelForSeq2SeqLM.from_pretrained(NMT_MODEL_NAME)
    # NMT_PIPELINE can be used for simpler calls, but for NLLB with explicit src/tgt, direct model.generate is better
    print("NMT model loaded successfully.")
except Exception as e:
    print(f"Error loading NMT model {NMT_MODEL_NAME}: {e}. Translations will be dummy.")

# SpaCy Model for text processing
NLP_SPACY = None
print(f"Loading spaCy model: {SPACY_MODEL_NAME}...")
try:
    NLP_SPACY = spacy.load(SPACY_MODEL_NAME)
    print("spaCy model loaded successfully.")
except Exception as e:
    print(f"Error loading spaCy model {SPACY_MODEL_NAME}: {e}. Please run `python -m spacy download {SPACY_MODEL_NAME}`. Text decomposition will be simplified.")

# Embedding Model for ChromaDB (Sentence Transformers)
EMBEDDING_MODEL = None
# print(f"Loading Sentence Transformer model: {SENTENCE_TRANSFORMER_MODEL}...")
# try:
#     from sentence_transformers import SentenceTransformer
#     EMBEDDING_MODEL = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
#     print("Sentence Transformer model loaded successfully.")
# except ImportError:
#     print("Sentence Transformers not installed. Please run `pip install sentence-transformers`. Using dummy embedding.")
# except Exception as e:
#     print(f"Error loading Sentence Transformer model {SENTENCE_TRANSFORMER_MODEL}: {e}. Using dummy embedding.")

# ChromaDB for Medical Knowledge Base
CHROMA_CLIENT = None
MEDICAL_KB_COLLECTION = None
# print("Initializing ChromaDB...")
# try:
#     import chromadb
#     # For persistent ChromaDB: CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_PATH)
#     # For in-memory (good for demo): CHROMA_CLIENT = chromadb.Client()
#     CHROMA_CLIENT = chromadb.Client()
#     MEDICAL_KB_COLLECTION = CHROMA_CLIENT.get_or_create_collection(name="medical_knowledge_base")

#     # Add some dummy medical knowledge if the collection is empty
#     if MEDICAL_KB_COLLECTION.count() == 0 and EMBEDDING_MODEL:
#         medical_docs = [
#             "Hypertension: abnormally high blood pressure.",
#             "Diabetes Mellitus: a group of diseases that result in too much sugar in the blood (high blood glucose).",
#             "Myocardial Infarction: also known as a heart attack, occurs when blood flow to the heart muscle is blocked.",
#             "Fever: an abnormally high body temperature, usually accompanied by shivering, headache, and in severe instances, delirium.",
#             "Pain: an unpleasant sensory and emotional experience associated with actual or potential tissue damage."
#         ]
#         doc_ids = [f"med_doc_{i}" for i in range(len(medical_docs))]
#         embeddings = EMBEDDING_MODEL.encode(medical_docs).tolist()
#         MEDICAL_KB_COLLECTION.add(
#             documents=medical_docs,
#             embeddings=embeddings,
#             metadatas=[{"term": d.split(':')[0].strip(), "language": "en"} for d in medical_docs],
#             ids=doc_ids
#         )
#         print("ChromaDB initialized with dummy medical knowledge.")
#     elif MEDICAL_KB_COLLECTION.count() == 0 and not EMBEDDING_MODEL:
#         print("ChromaDB initialized but no embedding model available to add dummy data.")
#     else:
#         print("ChromaDB already contains data.")
# except ImportError:
#     print("ChromaDB not installed. Please run `pip install chromadb`. Using dummy retriever.")
# except Exception as e:
#     print(f"Error initializing ChromaDB: {e}. Using dummy retriever.")

# Medical NER Model (Hugging Face Transformers pipeline for NER)
NER_PIPELINE = None
# print("Loading NER pipeline...")
# try:
#     # Using a generic NER model for demonstration. For production, fine-tune on medical data.
#     NER_PIPELINE = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
#     print("Generic NER pipeline loaded successfully.")
# except Exception as e:
#     print(f"Error loading NER model: {e}. Using dummy NER.")


# --- Helper Functions / Modules ---

class LanguageDetectionModule:
    @staticmethod
    def detect_language(text: str) -> str:
        if LID_MODEL:
            predictions = LID_MODEL.predict(text, k=1)
            lang_code = predictions[0][0].replace('__label__', '')
            # Map fasttext codes to NLLB codes if necessary
            # Example: 'en' -> 'eng_Latn', 'es' -> 'spa_Latn'
            lang_map = {"en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "zh": "zho_Hans"}
            return lang_map.get(lang_code, "und_Latn") # 'und_Latn' for undetermined/unknown
        else:
            # Dummy detection for demo if fasttext not loaded
            if any(char in text for char in "你好世界"): return "zho_Hans" # Chinese simplified
            if any(char in text for char in "Hola mundo"): return "spa_Latn" # Spanish
            if any(char in text for char in "Bonjour le monde"): return "fra_Latn" # French
            return "eng_Latn" # Default to English

class OCRModule:
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        # Placeholder for PyPDF2 or other OCR library integration
        # Requires `pip install PyPDF2`
        # import PyPDF2
        # reader = PyPDF2.PdfReader(pdf_file)
        # text = ""
        # for page in reader.pages:
        #     text += page.extract_text()
        # return text
        return "DUMMY PDF TEXT: This is placeholder text from a PDF. Patient has high fever and requires medication. Diagnosis: Influenza. Treatment: Rest and fluids."

class ContextualAugmentationModule:
    @staticmethod
    def retrieve_context(segment_text: str, top_k: int = 3) -> List[str]:
        if CHROMA_CLIENT and MEDICAL_KB_COLLECTION and EMBEDDING_MODEL:
            try:
                query_embeddings = EMBEDDING_MODEL.encode([segment_text]).tolist()
                results = MEDICAL_KB_COLLECTION.query(
                    query_embeddings=query_embeddings,
                    n_results=top_k,
                    include=['documents', 'metadatas']
                )
                context = [f"Medical context: {doc} ({meta.get('term', '')})" for doc, meta in zip(results['documents'][0], results['metadatas'][0])] 
                return context
            except Exception as e:
                print(f"Error querying ChromaDB: {e}")
                return []
        else:
            # Dummy context for demo if ChromaDB/Embedding model not loaded/initialized
            dummy_contexts = []
            if "heart" in segment_text.lower() or "cardiac" in segment_text.lower():
                dummy_contexts.append("Medical context: Myocardial Infarction: also known as a heart attack, occurs when blood flow to the heart muscle is blocked.")
            if "blood pressure" in segment_text.lower() or "hypertension" in segment_text.lower():
                dummy_contexts.append("Medical context: Hypertension: abnormally high blood pressure.")
            if "diabetes" in segment_text.lower() or "glucose" in segment_text.lower():
                dummy_contexts.append("Medical context: Diabetes Mellitus: a group of diseases that result in too much sugar in the blood (high blood glucose).")
            if "fever" in segment_text.lower() or "temperature" in segment_text.lower():
                dummy_contexts.append("Medical context: Fever: an abnormally high body temperature, usually accompanied by shivering, headache, and in severe instances, delirium.")
            return dummy_contexts

class TextDecompositionModule:
    @staticmethod
    def decompose_text(text: str) -> List[str]:
        if NLP_SPACY:
            doc = NLP_SPACY(text)
            # Simple sentence segmentation. Can be extended with rule-based decomposition for specific medical sections.
            segments = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            return segments
        else:
            # Fallback to simple paragraph splitting if spaCy is not available
            return [s.strip() for s in text.split('\n\n') if s.strip()]

class MedicalNERModule:
    @staticmethod
    def perform_ner(text: str) -> List[Dict]:
        if NER_PIPELINE:
            entities = NER_PIPELINE(text)
            # Filter for relevant medical entities if the model provides types (e.g., 'DIS', 'DRUG')
            return entities
        else:
            # Dummy NER for demo if pipeline not loaded
            dummy_entities = []
            lower_text = text.lower()
            if "hypertension" in lower_text:
                dummy_entities.append({"entity_group": "DISEASE", "word": "Hypertension", "start": lower_text.find("hypertension"), "end": lower_text.find("hypertension") + len("hypertension")})
            if "diabetes" in lower_text:
                dummy_entities.append({"entity_group": "DISEASE", "word": "Diabetes", "start": lower_text.find("diabetes"), "end": lower_text.find("diabetes") + len("diabetes")})
            if "fever" in lower_text:
                dummy_entities.append({"entity_group": "SYMPTOM", "word": "Fever", "start": lower_text.find("fever"), "end": lower_text.find("fever") + len("fever")})
            if "ibuprofen" in lower_text:
                dummy_entities.append({"entity_group": "DRUG", "word": "Ibuprofen", "start": lower_text.find("ibuprofen"), "end": lower_text.find("ibuprofen") + len("ibuprofen")})
            return dummy_entities

class CoreTranslationModule:
    @staticmethod
    def translate_segment(
        segment: str,
        source_lang: str,
        target_lang: str,
        medical_entities: List[Dict],
        contextual_info: List[str]
    ) -> str:
        if not (NMT_TOKENIZER and NMT_MODEL):
            return f"[DUMMY TRANSLATION of '{segment}' from {source_lang} to {target_lang} with entities {medical_entities} and context {contextual_info}]"

        # Construct a LangChain-like sophisticated prompt
        # This aims to enrich the input for the NMT model.
        prompt_parts = []
        prompt_parts.append(f"Translate the following medical text from {source_lang} to {target_lang}. ")
        if contextual_info:
            prompt_parts.append("Consider the following relevant medical information:")
            for info in contextual_info:
                prompt_parts.append(f"- {info}")
        if medical_entities:
            entity_words = [ent.get('word', '') for ent in medical_entities if ent.get('word')]
            if entity_words:
                prompt_parts.append(f"Pay close attention to these medical terms: {', '.join(entity_words)}.")
        
        prompt_parts.append("\nMedical Report Segment: ")
        prompt_parts.append(segment)

        full_prompt = " ".join(prompt_parts) # Join with space for a more natural flow

        try:
            # For NLLB-200, explicitly set source and target language tokens
            NMT_TOKENIZER.src_lang = source_lang
            inputs = NMT_TOKENIZER(full_prompt, return_tensors="pt")
            # Ensure target_lang is a valid NLLB language code, e.g., 'eng_Latn'
            if target_lang not in NMT_TOKENIZER.lang_code_to_id:
                print(f"Warning: Target language code '{target_lang}' not recognized by NMT model. Falling back to {PIVOT_LANGUAGE_CODE}.")
                target_lang = PIVOT_LANGUAGE_CODE # Fallback to a known high-resource language

            generated_tokens = NMT_MODEL.generate(
                **inputs,
                forced_bos_token_id=NMT_TOKENIZER.lang_code_to_id[target_lang],
                max_new_tokens=200 # Limit output length to prevent very long generations
            )
            translated_text = NMT_TOKENIZER.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translated_text
        except Exception as e:
            print(f"Error during core translation for segment '{segment}': {e}. Returning fallback.")
            return f"[TRANSLATION ERROR for '{segment}'. Source: {source_lang}, Target: {target_lang}. Entities: {', '.join([e['word'] for e in medical_entities]) if medical_entities else 'None'}, Context: {contextual_info}]"


# --- FastAPI Backend ---
app = FastAPI()

class TranslateRequest(BaseModel):
    text: str
    target_language: str # e.g., 'eng_Latn', 'spa_Latn'

class FeedbackRequest(BaseModel):
    translation_id: str
    corrected_translation: str

@app.post("/translate")
async def translate_medical_report(request: TranslateRequest):
    original_text = request.text
    target_language = request.target_language

    # 1. Language Detection
    source_language = LanguageDetectionModule.detect_language(original_text)
    print(f"Detected source language: {source_language}")

    # 2. Text Decomposition
    segments = TextDecompositionModule.decompose_text(original_text)
    print(f"Decomposed into {len(segments)} segments.")

    translated_segments = []
    full_initial_translation = ""
    for i, segment in enumerate(segments):
        if not segment.strip():
            continue
        
        print(f"Processing segment {i+1}/{len(segments)}: {segment[:50]}...")

        # 3. Medical NER
        medical_entities = MedicalNERModule.perform_ner(segment)
        # print(f"  Identified entities: {medical_entities}") # Too verbose

        # 4. Contextual Augmentation (Retrieval from Medical Knowledge Base)
        contextual_info = ContextualAugmentationModule.retrieve_context(segment)
        # print(f"  Retrieved context: {contextual_info}") # Too verbose

        # 5. Core Translation
        translated_segment = CoreTranslationModule.translate_segment(
            segment=segment,
            source_lang=source_language,
            target_lang=target_language,
            medical_entities=medical_entities,
            contextual_info=contextual_info
        )
        translated_segments.append(translated_segment)
        full_initial_translation += translated_segment + "\n\n"

    # Store initial translation for feedback loop
    translation_id = db_manager.save_translation(original_text, source_language, full_initial_translation)
    print(f"Translation saved with ID: {translation_id}")

    return {"translation_id": translation_id, "initial_translation": full_initial_translation}

@app.post("/submit_feedback")
async def submit_feedback(request: FeedbackRequest):
    db_manager.update_feedback(request.translation_id, request.corrected_translation)
    print(f"Feedback submitted for translation ID: {request.translation_id}")
    return {"message": "Feedback submitted successfully."}


# --- Streamlit Frontend ---

def run_streamlit_app():
    st.set_page_config(layout="wide")
    st.title("Cross-Lingual Medical Report Translator")
    st.subheader("Context-Augmented and Iterative Translation with Doctor Feedback")

    st.markdown("""
    This application translates medical reports, leveraging contextual information,
    text decomposition, and an iterative feedback loop to improve accuracy.
    """)

    st.header("1. Upload Medical Report")
    uploaded_file = st.file_uploader("Choose a PDF or text file", type=["txt", "pdf"])
    report_text_input = st.text_area("Or paste your medical report here:", height=300)

    actual_report_text = report_text_input

    if uploaded_file is not None:
        st.info(f"Processing uploaded file: {uploaded_file.name}")
        if uploaded_file.type == "application/pdf":
            # This is a placeholder; full PDF text extraction requires PyPDF2 or other tools
            # For a real application, you would implement OCRModule.extract_text_from_pdf(uploaded_file)
            st.warning("PDF extraction is a placeholder in this demo. For full functionality, use text files or paste text.")
            actual_report_text = OCRModule.extract_text_from_pdf(uploaded_file) # Use dummy text for demo
        else:
            actual_report_text = uploaded_file.read().decode("utf-8")
        st.text_area("Report Content (from file/PDF placeholder):", value=actual_report_text, height=200, disabled=True)

    target_language_options = [
        ("English", "eng_Latn"),
        ("Spanish", "spa_Latn"),
        ("French", "fra_Latn"),
        ("Chinese (Simplified)", "zho_Hans"),
        ("German", "deu_Latn"),
        ("Arabic", "arb_Arab")
    ]
    target_language_display = st.selectbox(
        "Select Target Language",
        options=target_language_options,
        format_func=lambda x: x[0]
    )
    selected_target_language_code = target_language_display[1]

    if st.button("Translate Report"):
        if actual_report_text:
            with st.spinner("Translating... This may take a moment due to AI model processing."):
                try:
                    response = requests.post(
                        "http://localhost:8000/translate", # FastAPI backend URL
                        json={
                            "text": actual_report_text,
                            "target_language": selected_target_language_code
                        }
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state["translation_id"] = result["translation_id"]
                        st.session_state["initial_translation"] = result["initial_translation"]
                        st.success("Translation complete!")
                    else:
                        st.error(f"Error from backend: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Please ensure the backend is running (e.g., `uvicorn medical_translator_app:app --reload`).")
                except Exception as e:
                    st.error(f"An unexpected error occurred during translation: {e}")
        else:
            st.warning("Please upload a file or paste text to translate.")

    if "initial_translation" in st.session_state and st.session_state["initial_translation"]:
        st.header("2. Initial Machine Translation")
        st.text_area("Machine Translation Output:", value=st.session_state["initial_translation"], height=400, disabled=True)

        st.header("3. Doctor Feedback and Correction")
        corrected_translation = st.text_area(
            "Review and Correct Translation (editable):",
            value=st.session_state["initial_translation"],
            height=400,
            key="corrected_translation_input" # Add a key for consistency
        )

        if st.button("Submit Corrected Translation"): 
            if "translation_id" in st.session_state and st.session_state["translation_id"]:
                try:
                    feedback_response = requests.post(
                        "http://localhost:8000/submit_feedback", # FastAPI backend URL
                        json={
                            "translation_id": st.session_state["translation_id"],
                            "corrected_translation": corrected_translation
                        }
                    )
                    if feedback_response.status_code == 200:
                        st.success("Feedback submitted successfully! Thank you for improving our model.")
                    else:
                        st.error(f"Error submitting feedback: {feedback_response.status_code} - {feedback_response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Please ensure the backend is running.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during feedback submission: {e}")
            else:
                st.warning("Please translate a report first to obtain a translation ID before submitting feedback.")

    st.sidebar.markdown("### How to Run This Application")
    st.sidebar.markdown("1.  **Save this code** as `medical_translator_app.py`.")
    st.sidebar.markdown("2.  **Install dependencies** (`pip install`):")
    st.sidebar.markdown("    *   `fastapi uvicorn streamlit pydantic transformers spacy requests`")
    st.sidebar.markdown("    *   **Optional for full functionality** (uncomment imports in code and install):")
    st.sidebar.markdown("        *   `fasttext`: for robust language detection (download `lid.176.bin` from `fasttext.cc/docs/en/language-identification.html`)")
    st.sidebar.markdown("        *   `PyPDF2`: for basic PDF text extraction (replace dummy in `OCRModule`)")
    st.sidebar.markdown("        *   `chromadb sentence-transformers`: for a persistent medical knowledge base and embeddings (download `all-MiniLM-L6-v2`)")
    st.sidebar.markdown("    *   **For spaCy model**: `python -m spacy download en_core_web_sm`")
    st.sidebar.markdown("    *   **For Hugging Face NMT model**: The model `facebook/nllb-200-distilled-600M` is large and will be downloaded on first use by `transformers`. Ensure you have sufficient disk space and memory.")
    st.sidebar.markdown("3.  **Run FastAPI Backend**: Open your terminal/command prompt and execute: `uvicorn medical_translator_app:app --reload`")
    st.sidebar.markdown("    *(This starts the API server, usually on `http://localhost:8000`)*")
    st.sidebar.markdown("4.  **Run Streamlit Frontend**: Open *another* terminal/command prompt and execute: `streamlit run medical_translator_app.py`")
    st.sidebar.markdown("    *(This starts the UI, usually on `http://localhost:8501`)*")
    st.sidebar.markdown("5.  Access the Streamlit app in your web browser (check the URL provided by Streamlit). Ensure both are running.")

# --- Main Execution Block --- 
# This block ensures that if the script is run directly by Streamlit, it calls the Streamlit app function.
# If run by uvicorn, uvicorn directly imports `app` and runs it.
if __name__ == "__main__":
    # Check if the script is being run by Streamlit
    # This is a heuristic as __name__ will be '__main__' in both direct run and streamlit run
    # We check environment variables or command-line arguments typical for streamlit
    if os.getenv("STREAMLIT_SERVER_URL") or "streamlit" in " ".join(os.sys.argv):
        run_streamlit_app()
    else:
        # If not detected as Streamlit, assume it's for FastAPI or just providing instructions.
        # Uvicorn will handle starting the FastAPI app directly via `uvicorn medical_translator_app:app`
        print("To run this application, please follow the instructions below:")
        print("1. Run FastAPI backend: `uvicorn medical_translator_app:app --reload`")
        print("2. In a separate terminal, run Streamlit frontend: `streamlit run medical_translator_app.py`")
        print("Refer to the Streamlit app's sidebar for detailed setup and running instructions.")
