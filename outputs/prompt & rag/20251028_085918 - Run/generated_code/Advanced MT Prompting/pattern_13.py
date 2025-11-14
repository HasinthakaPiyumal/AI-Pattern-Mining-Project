"""
Global Health Bridge - An AI-powered medical translation platform for low-resource languages.

This single file demonstrates the architecture components using placeholder implementations
for a conceptual understanding. In a real-world scenario, this would be split into
multiple services and files.

To run:
1.  Install dependencies: `pip install fastapi uvicorn[standard] streamlit transformers spacy nltk sentence-transformers chromadb python-dotenv pydantic`
    *   For spaCy model: `python -m spacy download en_core_web_sm`
    *   For NLTK data: `python -c "import nltk; nltk.download('punkt')"`
2.  Create a `.env` file with `OPENAI_API_KEY=your_openai_key_here` (if using OpenAI LLM).
3.  To run the FastAPI backend: `RUN_MODE=api uvicorn global_health_bridge:app --host 0.0.0.0 --port 8000`
4.  To run the Streamlit frontend: `RUN_MODE=streamlit streamlit run global_health_bridge.py`
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

# External Libraries (placeholders, ensure these are installed)
from dotenv import load_dotenv
from pydantic import BaseModel

# AI/ML Libraries
import spacy
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from chromadb import Client, Settings
from chromadb.utils import embedding_functions

# Web Frameworks
from fastapi import FastAPI, HTTPException
import uvicorn

# UI Framework
import streamlit as st

# --- Configuration and Environment Variables ---
load_dotenv() # Load environment variables from .env file

# Placeholder for OpenAI API Key if using OpenAI models for candidate generation
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ChromaDB Configuration
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "medical_documents"

# NLTK data download (run once)
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")

# --- Pydantic Models for Data Validation ---
class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str
    user_id: Optional[str] = None

class TranslationSegment(BaseModel):
    id: str
    original_text: str
    translated_text: str
    confidence_score: float
    requires_review: bool = False

class TranslationResult(BaseModel):
    request_id: str
    original_text: str
    source_language: str
    target_language: str
    translated_segments: List[TranslationSegment]
    final_translated_text: str
    timestamp: datetime = datetime.now()

class FeedbackRequest(BaseModel):
    translation_id: str
    segment_id: Optional[str] = None # For segment-specific feedback
    feedback_text: str
    is_correction: bool = False
    user_id: Optional[str] = None

# --- Placeholder Database Interface ---
# In a real application, this would interact with PostgreSQL or another persistent DB.
class MockDatabase:
    def __init__(self):
        self.translations = {}
        self.feedback = {}

    def save_translation(self, result: TranslationResult):
        self.translations[result.request_id] = result.model_dump()
        print(f"[DB] Saved translation: {result.request_id}")

    def get_translation(self, translation_id: str) -> Optional[TranslationResult]:
        data = self.translations.get(translation_id)
        return TranslationResult(**data) if data else None

    def save_feedback(self, feedback: FeedbackRequest):
        feedback_id = str(uuid.uuid4())
        self.feedback[feedback_id] = feedback.model_dump()
        print(f"[DB] Saved feedback: {feedback_id} for translation {feedback.translation_id}")

db = MockDatabase()

# --- RAG System (ChromaDB Integration) ---
class RAGSystem:
    def __init__(self, db_path: str, collection_name: str):
        self.client = Client(Settings(persist_directory=db_path))
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_function)
        self._initialize_medical_knowledge_base()
        print("[RAG] ChromaDB initialized.")

    def _initialize_medical_knowledge_base(self):
        # Add some mock medical documents to the vector store if it's empty
        if self.collection.count() == 0:
            print("[RAG] Initializing medical knowledge base with mock data...")
            documents = [
                "Diabetes mellitus is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored for energy.",
                "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
                "Aspirin is used to reduce fever and relieve mild to moderate pain from conditions such as muscle aches, toothaches, common cold, and headaches. It may also be used to reduce pain and swelling in conditions such as arthritis.",
                "Symptoms of pneumonia include cough, fever, chills, and difficulty breathing. It is an infection that inflames air sacs in one or both lungs, which may fill with fluid.",
                "Magnetic Resonance Imaging (MRI) is a medical imaging technique used in radiology to form pictures of the anatomy and the physiological processes of the body in both health and disease."
            ]
            metadatas = [
                {"source": "medical_wiki", "topic": "endocrinology"},
                {"source": "medical_wiki", "topic": "cardiology"},
                {"source": "drug_info", "topic": "pharmacology"},
                {"source": "medical_wiki", "topic": "respiratory"},
                {"source": "medical_imaging", "topic": "radiology"}
            ]
            ids = [f"doc{i}" for i in range(len(documents))]
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"[RAG] Added {len(documents)} mock medical documents.")

    def retrieve_medical_exemplars(self, query_text: str, n_results: int = 2) -> List[str]:
        try:
            results = self.collection.query(query_texts=[query_text], n_results=n_results)
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            print(f"[RAG Error] Failed to retrieve exemplars: {e}")
            return []

rag_system = RAGSystem(db_path=CHROMA_DB_PATH, collection_name=CHROMA_COLLECTION_NAME)

# --- Translation Service ---
class TranslationService:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm") # For NER and segmentation
        self.mt_tokenizer_en_xx = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.mt_model_en_xx = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M") # Placeholder for MT to/from English
        self.llm_pipeline = pipeline("text2text-generation", model="t5-small") # Smaller LLM for candidate generation
        # self.llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base") # Alternative for better quality
        print("[Service] Models loaded: spaCy, NLLB, T5-small.")

    def _ner_medical_terms(self, text: str, lang: str) -> List[Dict[str, str]]:
        # Simplified NER: For a real app, integrate with medical ontologies/vocabularies.
        doc = self.nlp(text)
        medical_entities = []
        for ent in doc.ents:
            # Placeholder: Assume certain entity types are medical
            if ent.label_ in ["ORG", "PRODUCT", "PERSON", "GPE"] or any(term in ent.text.lower() for term in ["syndrome", "disease", "medication", "diagnosis", "treatment"]):
                 medical_entities.append({"text": ent.text, "label": ent.label_, "lang": lang})
        return medical_entities

    def _segment_document(self, text: str) -> List[str]:
        return nltk.sent_tokenize(text)

    def _pre_translate_to_english(self, text: str, source_lang: str) -> str:
        # Placeholder: NLLB supports many languages, but we're simulating pre-translation if source is low-resource.
        # In a real app, check a list of low-resource languages.
        if source_lang.lower() not in ["en", "eng", "english"] and source_lang.lower() not in ["fr", "de", "es"] : # Example: Treat these as potentially low-resource/requiring pivot
            print(f"[Pre-translate] Translating from {source_lang} to English...")
            # NLLB uses language codes like 'eng_Latn', 'fra_Latn'
            src_lang_code = f"{source_lang.lower()}_Latn" if source_lang.lower() != "en" else "eng_Latn"
            tgt_lang_code = "eng_Latn"
            
            # Set the source language for the tokenizer
            self.mt_tokenizer_en_xx.src_lang = src_lang_code
            
            encoded_text = self.mt_tokenizer_en_xx(text, return_tensors="pt")
            generated_tokens = self.mt_model_en_xx.generate(**encoded_text, forced_bos_token_id=self.mt_tokenizer_en_xx.lang_code_to_id[tgt_lang_code])
            return self.mt_tokenizer_en_xx.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return text # If English or high-resource, no pre-translation needed

    def _augment_prompt(self, original_text: str, ner_results: List[Dict[str, str]], exemplars: List[str]) -> str:
        context_parts = []
        if ner_results:
            context_parts.append("Medical terms identified: " + ", ".join([e["text"] for e in ner_results]) + ".")
        if exemplars:
            context_parts.append("Relevant medical context: " + " ".join(exemplars) + ".")
        
        if context_parts:
            return f"Context: {' '.join(context_parts)}\nTranslate the following medical text: {original_text}"
        return original_text

    def _generate_candidate_translations(self, text_to_translate: str, target_lang: str) -> List[str]:
        # Use a smaller LLM to generate multiple translation candidates
        # In a real scenario, this would be more sophisticated, potentially using few-shot prompting.
        
        # T5 uses prefixes like "translate English to German:"
        if target_lang.lower() == "english":
            prompt = f"translate to English: {text_to_translate}"
        elif target_lang.lower() == "french":
            prompt = f"translate to French: {text_to_translate}"
        elif target_lang.lower() == "spanish":
            prompt = f"translate to Spanish: {text_to_translate}"
        else:
            prompt = f"translate to {target_lang}: {text_to_translate}"

        # Generate a few diverse candidates
        candidates = self.llm_pipeline(prompt, num_return_sequences=3, do_sample=True, top_k=50, top_p=0.95, max_new_tokens=50)
        return [c["generated_text"] for c in candidates]

    def _score_and_select(self, candidates: List[str]) -> (str, float):
        # Simplified scoring: just pick the first one, or add more complex logic.
        # In a real app: use metrics like BLEU, COMET, or specialized medical translation quality estimators.
        # Could also use another LLM to score candidates based on medical accuracy and fluency.
        if not candidates: return "", 0.0
        
        # Placeholder confidence based on candidate count or mock value
        confidence = 0.8 + (0.05 * len(candidates)) # Higher confidence if more candidates were generated (mock)
        return candidates[0], min(confidence, 0.99) # Cap confidence

    def _cohere_segments(self, translated_segments: List[TranslationSegment]) -> str:
        # Simple joining. Advanced cohesion would involve checking pronoun consistency, flow, etc.
        return " ".join([s.translated_text for s in translated_segments])

    def translate(self, request: TranslationRequest) -> TranslationResult:
        print(f"[Service] Processing translation request for text: {request.text[:50]}...")
        
        # 1. Augmented Prompting & Preprocessing
        ner_results = self._ner_medical_terms(request.text, request.source_language)
        query_for_rag = request.text
        if ner_results:
            query_for_rag = " ".join([e["text"] for e in ner_results]) + " " + request.text # Prioritize medical terms for RAG
        exemplars = rag_system.retrieve_medical_exemplars(query_for_rag)
        augmented_input = self._augment_prompt(request.text, ner_results, exemplars)
        
        pre_translated_to_english = augmented_input # Start with augmented input
        if request.source_language.lower() not in ["en", "english"]:
            pre_translated_to_english = self._pre_translate_to_english(augmented_input, request.source_language)
        
        # 2. Strategic Planning & Decomposition
        segments = self._segment_document(pre_translated_to_english)
        translated_segments_data = []

        for i, segment_text in enumerate(segments):
            segment_id = str(uuid.uuid4())
            # In a real app, knowledge extraction per segment would be more granular.
            # For demo, _generate_candidate_translations will use the segment directly.
            
            candidate_translations = self._generate_candidate_translations(segment_text, request.target_language)
            best_translation, confidence = self._score_and_select(candidate_translations)
            
            # 3. Human-in-the-Loop (initial flagging)
            requires_review = confidence < 0.7 # Example threshold
            
            translated_segments_data.append(TranslationSegment(
                id=segment_id,
                original_text=segment_text,
                translated_text=best_translation,
                confidence_score=confidence,
                requires_review=requires_review
            ))
        
        final_translated_text = self._cohere_segments(translated_segments_data)

        translation_id = str(uuid.uuid4())
        result = TranslationResult(
            request_id=translation_id,
            original_text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            translated_segments=translated_segments_data,
            final_translated_text=final_translated_text
        )
        db.save_translation(result)
        return result

translation_service = TranslationService()

# --- FastAPI Backend ---
app = FastAPI(
    title="Global Health Bridge API",
    description="API for AI-powered medical translation, leveraging augmented prompting, strategic decomposition, and human-in-the-loop refinement.",
    version="1.0.0",
)

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok", "message": "Global Health Bridge API is running!"}

@app.post("/translate", response_model=TranslationResult, tags=["Translation"])
async def translate_text(request: TranslationRequest):
    """Perform a medical translation."""
    try:
        result = translation_service.translate(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(feedback_request: FeedbackRequest):
    """Submit feedback or corrections for a translation."""
    # In a real system, this would trigger model retraining or a human review workflow.
    db.save_feedback(feedback_request)
    return {"message": "Feedback submitted successfully.", "feedback_id": str(uuid.uuid4())}

# --- Streamlit Frontend ---
def streamlit_app():
    st.set_page_config(layout="wide", page_title="Global Health Bridge")

    st.title("🌍 Global Health Bridge")
    st.subheader("AI-powered Medical Translation for Low-Resource Languages")

    st.markdown("""
    This platform facilitates accurate medical translations using advanced AI patterns:
    *   **Augmented Prompting:** Integrates medical terminology and relevant context.
    *   **Strategic Planning:** Decomposes complex texts and generates robust translations.
    *   **Human-in-the-Loop:** Incorporates feedback for continuous refinement.
    """)

    # --- Translation Input Section ---
    st.header("Translate Medical Text")
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox("Source Language", ["English", "French", "Spanish", "Swahili", "Haitian Creole"], index=0)
    with col2:
        target_lang = st.selectbox("Target Language", ["French", "English", "Spanish", "Swahili", "Haitian Creole"], index=0)

    input_text = st.text_area("Enter Medical Text Here", height=200, 
                               placeholder="e.g., The patient presents with symptoms of severe abdominal pain, nausea, and persistent fever. Diagnosis indicates acute appendicitis, requiring immediate surgical intervention.")

    if st.button("Translate", type="primary"):
        if not input_text:
            st.warning("Please enter text to translate.")
        else:
            with st.spinner("Translating..."):
                try:
                    # In a real app, call the FastAPI backend:
                    # import requests
                    # response = requests.post("http://localhost:8000/translate", json={
                    #     "text": input_text,
                    #     "source_language": source_lang,
                    #     "target_language": target_lang
                    # })
                    # response.raise_for_status()
                    # translation_result = TranslationResult(**response.json())

                    # For this single-file demo, call the service directly
                    request_obj = TranslationRequest(
                        text=input_text,
                        source_language=source_lang,
                        target_language=target_lang,
                        user_id="streamlit_user"
                    )
                    translation_result = translation_service.translate(request_obj)
                    st.session_state.current_translation_result = translation_result

                    st.success("Translation Complete!")
                    st.subheader("Translated Text:")
                    st.write(translation_result.final_translated_text)

                    st.subheader("Detailed Segments (for Review):")
                    for segment in translation_result.translated_segments:
                        expander_title = f"Segment {segment.id[:8]}... (Confidence: {segment.confidence_score:.2f})"
                        if segment.requires_review:
                            expander_title += " ⚠️ **Requires Review**"
                        with st.expander(expander_title):
                            st.markdown(f"**Original:** {segment.original_text}")
                            st.markdown(f"**Translated:** {segment.translated_text}")
                            st.markdown(f"**Confidence Score:** {segment.confidence_score:.2f}")
                            if segment.requires_review:
                                st.error("This segment's translation confidence is low. Please review.")
                            
                            # --- Feedback Section for Each Segment ---
                            st.markdown("#### Provide Feedback for this Segment")
                            feedback_key = f"feedback_input_{segment.id}"
                            segment_feedback_text = st.text_area("Your feedback/correction for this segment:", key=feedback_key)
                            if st.button("Submit Segment Feedback", key=f"submit_segment_feedback_{segment.id}"):
                                if segment_feedback_text:
                                    feedback_req = FeedbackRequest(
                                        translation_id=translation_result.request_id,
                                        segment_id=segment.id,
                                        feedback_text=segment_feedback_text,
                                        is_correction=True, # Assuming specific feedback is a correction
                                        user_id="streamlit_user"
                                    )
                                    # In a real app, send to FastAPI: requests.post("http://localhost:8000/feedback", json=feedback_req.model_dump())
                                    db.save_feedback(feedback_req)
                                    st.success("Segment feedback submitted!")
                                else:
                                    st.warning("Please enter feedback text.")

                except HTTPException as e:
                    st.error(f"API Error: {e.detail}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    # --- General Translation Feedback Section ---
    if 'current_translation_result' in st.session_state:
        st.markdown("--- ")
        st.header("Overall Translation Feedback")
        general_feedback_text = st.text_area("Provide general feedback for the entire translation:", height=100)
        if st.button("Submit General Feedback"):
            if general_feedback_text:
                feedback_req = FeedbackRequest(
                    translation_id=st.session_state.current_translation_result.request_id,
                    feedback_text=general_feedback_text,
                    is_correction=False,
                    user_id="streamlit_user"
                )
                # In a real app, send to FastAPI: requests.post("http://localhost:8000/feedback", json=feedback_req.model_dump())
                db.save_feedback(feedback_req)
                st.success("General feedback submitted!")
            else:
                st.warning("Please enter general feedback text.")

# --- Main Execution Logic ---
if __name__ == "__main__":
    run_mode = os.getenv("RUN_MODE", "streamlit").lower()

    if run_mode == "api":
        print("Starting FastAPI application...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif run_mode == "streamlit":
        print("Starting Streamlit application...")
        streamlit_app()
    else:
        print("Invalid RUN_MODE. Please set to 'api' or 'streamlit'. Defaulting to Streamlit.")
        streamlit_app()