"""
Global HealthBridge Translator

This application provides a platform for healthcare professionals and patients to accurately
translate medical documents and texts across various languages, with a focus on enhancing
quality for non-English and low-resource languages by incorporating AI design patterns:
Augmented Prompting & Preprocessing, Strategic Planning & Decomposition, and
Human-in-the-Loop & Iterative Refinement.
"""

import os
from typing import List, Dict, Any, Tuple
import sqlite3
import json

# --- External Libraries (Mocked for single-file execution without actual installs) ---
# In a real scenario, you would install these:
# pip install fastapi uvicorn "python-multipart" gradio transformers sentence-transformers chromadb nltk spacy SQLAlchemy pyspellchecker language-tool-python python-dotenv

# Mock imports for demonstration. Remove these and uncomment actual imports for a runnable app.
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import uvicorn
# import gradio as gr
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import SentenceTransformer
# from chromadb import Client, Settings
# from chromadb.utils import embedding_functions
# import nltk
# import spacy
# from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, func
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from spellchecker import SpellChecker
# import language_tool_python
# from dotenv import load_dotenv

# --- Mock Implementations for external libraries and models ---
# This section allows the code structure to be present without requiring full environment setup.
# For actual execution, replace these mocks with real imports and model loading.

# Mock FastAPI components
class FastAPI:
    def __init__(self, *args, **kwargs): pass
    def get(self, *args, **kwargs): return lambda f: f
    def post(self, *args, **kwargs): return lambda f: f
class HTTPException(Exception): pass
class BaseModel: pass # Simplified for mock

# Mock uvicorn
class UvicornMock:
    @staticmethod
    def run(app, host="0.0.0.0", port=8000):
        print(f"Mock Uvicorn: Running app on http://{host}:{port}")

uvicorn = UvicornMock()

# Mock Gradio
class GradioMock:
    def Interface(self, fn, inputs, outputs, *args, **kwargs):
        print("Mock Gradio: Interface created")
        return self
    def launch(self, *args, **kwargs):
        print("Mock Gradio: Launching interface...")
gr = GradioMock()

# Mock transformers
class AutoTokenizerMock:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, text, return_tensors=None, truncation=True, padding=True): 
        print(f"Mock Tokenizer: Tokenizing '{text[:30]}...' ")
        return {"input_ids": [list(map(ord, text))], "attention_mask": [[1]*len(text)]}
class AutoModelForSeq2SeqLMMock:
    def __init__(self, *args, **kwargs): pass
    def generate(self, input_ids, max_length=150, num_beams=5, early_stopping=True): 
        print("Mock Model: Generating translation...")
        # Simple mock translation: reverse the input characters
        mock_translation = input_ids[0][0][::-1] # Assuming input_ids[0][0] is a list of ASCII values
        return [[mock_translation]]
    
AutoTokenizer = AutoTokenizerMock
AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLMMock

# Mock sentence_transformers
class SentenceTransformerMock:
    def __init__(self, *args, **kwargs): pass
    def encode(self, sentences, convert_to_tensor=False):
        print(f"Mock SentenceTransformer: Encoding '{sentences[0][:30]}...' ")
        return [[hash(s) % 1000 for _ in range(384)] for s in sentences] # Mock embeddings
SentenceTransformer = SentenceTransformerMock

# Mock chromadb
class ChromaDBMock:
    def __init__(self, *args, **kwargs):
        self.collections = {}
    def get_or_create_collection(self, name, *args, **kwargs):
        if name not in self.collections: self.collections[name] = CollectionMock(name)
        return self.collections[name]
class CollectionMock:
    def __init__(self, name):
        self.name = name
        self.documents = []
        self.metadatas = []
        self.embeddings = []
        self.ids = []
    def add(self, documents, metadatas, embeddings, ids):
        print(f"Mock ChromaDB Collection '{self.name}': Adding {len(documents)} documents.")
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.embeddings.extend(embeddings)
        self.ids.extend(ids)
    def query(self, query_embeddings, n_results=1, *args, **kwargs):
        print(f"Mock ChromaDB Collection '{self.name}': Querying for {n_results} results.")
        if not self.embeddings or not query_embeddings: return {"documents": [[]], "metadatas": [[]], "distances": [[0]]}
        # Simple mock query: return the first document as a closest match
        return {"documents": [[self.documents[0] if self.documents else ""]], 
                "metadatas": [[self.metadatas[0] if self.metadatas else {}]], 
                "distances": [[0.1]]}
Client = ChromaDBMock
Settings = object # Mock class
embedding_functions = object # Mock module

# Mock nltk
class NLTKMock:
    def sent_tokenize(self, text, language="english"): 
        print(f"Mock NLTK: Sentence tokenizing '{text[:30]}...' ")
        return [s.strip() for s in text.replace(". ", ".\n").split("\n")] # Basic sentence split
    def word_tokenize(self, text): 
        print(f"Mock NLTK: Word tokenizing '{text[:30]}...' ")
        return text.split()
    def download(self, *args, **kwargs): pass
nltk = NLTKMock()
nltk.download("punkt", quiet=True)

# Mock spacy
class SpacyMock:
    def load(self, model_name):
        print(f"Mock Spacy: Loading model '{model_name}'")
        return SpacyModelMock()
class SpacyModelMock:
    def __call__(self, text):
        print(f"Mock Spacy Model: Processing '{text[:30]}...' ")
        return SpacyDocMock(text)
class SpacyDocMock:
    def __init__(self, text):
        self.text = text
        self.ents = [] # Mock entities
    def sents(self):
        return [SpacySpanMock(s) for s in nltk.sent_tokenize(self.text)]
class SpacySpanMock:
    def __init__(self, text): self.text = text
    def __str__(self): return self.text
spacy = SpacyMock()

# Mock SQLAlchemy
Base = declarative_base() # type: ignore
class Column: pass
class Integer: pass
class String: pass
class Text: pass
class Boolean: pass
class DateTime: pass
class func: pass
def create_engine(*args, **kwargs): 
    print("Mock SQLAlchemy: Engine created.")
    class MockEngine: 
        def connect(self): return self
        def close(self): pass
        def execute(self, query): 
            print(f"Mock SQLAlchemy: Executing query: {query}")
            # Simulate table creation for sqlite
            if "CREATE TABLE" in query:
                try:
                    self.cursor.execute(query)
                    self.conn.commit()
                except sqlite3.OperationalError as e:
                    print(f"Mock DB Error (ignored for mock setup): {e}")
            return MockResult()
        def begin(self): return self
        def __enter__(self): 
            self.conn = sqlite3.connect(":memory:") # Use in-memory for mock
            self.cursor = self.conn.cursor()
            return self
        def __exit__(self, exc_type, exc_val, exc_tb): self.conn.close()
        def commit(self): self.conn.commit()
        def rollback(self): self.conn.rollback()
    return MockEngine()
class MockResult:
    def first(self): return None
    def scalar(self): return None
    def all(self): return []
def sessionmaker(bind):
    print("Mock SQLAlchemy: Sessionmaker created.")
    class MockSession:
        def __init__(self):
            self.conn = sqlite3.connect(":memory:")
            self.cursor = self.conn.cursor()
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): self.conn.close()
        def add(self, *args): pass
        def commit(self0): self.conn.commit()
        def rollback(self): self.conn.rollback()
        def query(self, *args): return self # Simplified mock query
        def filter_by(self, *args, **kwargs): return self
        def first(self): return None
        def all(self): return []
    return MockSession

# Mock pyspellchecker
class SpellCheckerMock:
    def __init__(self, *args, **kwargs): pass
    def correction(self, word): 
        print(f"Mock SpellChecker: Correcting '{word}' ")
        return word if len(word) > 2 and word[0].islower() else "corrected_word" # Simple heuristic
    def unknown(self, words): return []
SpellChecker = SpellCheckerMock

# Mock language_tool_python
class LanguageToolMock:
    def __init__(self, *args, **kwargs): pass
    def check(self, text):
        print(f"Mock LanguageTool: Checking '{text[:30]}...' ")
        if "grammar error" in text.lower():
            return [{"message": "Grammar error found", "replacements": ["fixed phrase"]}]
        return []
language_tool_python = LanguageToolMock()

# Mock dotenv
def load_dotenv(*args, **kwargs): print("Mock Dotenv: Loading environment variables.")
load_dotenv()

# --- Database Models (SQLite for simplicity) ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./healthbridge.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class MedicalGlossary(Base): # type: ignore
    __tablename__ = "medical_glossary"
    id = Column(Integer, primary_key=True, index=True)
    source_term = Column(String, index=True, nullable=False)
    target_term = Column(String, nullable=False)
    language = Column(String, nullable=False) # e.g., "en", "fr"
    domain = Column(String, default="general_medical")

class TranslationFeedback(Base): # type: ignore
    __tablename__ = "translation_feedback"
    id = Column(Integer, primary_key=True, index=True)
    original_text = Column(Text, nullable=False)
    machine_translation = Column(Text, nullable=False)
    human_correction = Column(Text)
    source_lang = Column(String, nullable=False)
    target_lang = Column(String, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    is_accurate = Column(Boolean)
    feedback_notes = Column(Text)

# Create tables (will only work with real SQLAlchemy setup)
# Base.metadata.create_all(bind=engine)

# Mock Base.metadata.create_all for sqlite
with engine.begin() as conn:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS medical_glossary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_term VARCHAR NOT NULL,
        target_term VARCHAR NOT NULL,
        language VARCHAR NOT NULL,
        domain VARCHAR DEFAULT 'general_medical'
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS translation_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_text TEXT NOT NULL,
        machine_translation TEXT NOT NULL,
        human_correction TEXT,
        source_lang VARCHAR NOT NULL,
        target_lang VARCHAR NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        is_accurate BOOLEAN,
        feedback_notes TEXT
    );
    """)

# --- Helper Classes and Components ---

class MedicalTerminology:
    def __init__(self):
        self.db_session = SessionLocal()
        # Initialize with some mock terms
        self._add_mock_glossary_terms()

    def _add_mock_glossary_terms(self):
        # Check if table is empty before adding mocks
        count = self.db_session.query(MedicalGlossary).count() # type: ignore
        if count == 0:
            mock_terms = [
                {"source_term": "hypertension", "target_term": "alta presión", "language": "es", "domain": "cardiology"},
                {"source_term": "diabetes", "target_term": "diabetes", "language": "es", "domain": "endocrinology"},
                {"source_term": "diagnosis", "target_term": "diagnóstico", "language": "es"},
                {"source_term": "treatment", "target_term": "tratamiento", "language": "es"},
                {"source_term": "fever", "target_term": "fiebre", "language": "es"},
                {"source_term": "headache", "target_term": "dolor de cabeza", "language": "es"},
                {"source_term": "blood pressure", "target_term": "presión arterial", "language": "es"},
            ]
            for term_data in mock_terms:
                term = MedicalGlossary(**term_data)
                self.db_session.add(term)
            self.db_session.commit()
            print("Mock medical glossary terms added.")
        self.db_session.close()

    def get_translation(self, term: str, target_lang: str) -> str | None:
        # Simple exact match for demonstration
        session = SessionLocal()
        entry = session.query(MedicalGlossary).filter_by(
            source_term=term.lower(), language=target_lang
        ).first()
        session.close()
        return entry.target_term if entry else None

    def identify_medical_terms(self, text: str, source_lang: str = "en") -> List[str]:
        # In a real app, this would use more sophisticated NLP/NER or ontology lookup.
        # For now, it's a simple keyword check against our mock glossary.
        session = SessionLocal()
        all_source_terms = [t.source_term for t in session.query(MedicalGlossary).filter_by(language=source_lang).all()] # type: ignore
        session.close()
        found_terms = [term for term in all_source_terms if term in text.lower()]
        return found_terms

class ExemplarRetriever:
    def __init__(self, collection_name="medical_exemplars"):
        # Use an in-memory client for simplicity in this single file example
        self.client = Client(Settings(allow_reset=True))
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        self._add_mock_exemplars()

    def _add_mock_exemplars(self):
        if self.collection.count() == 0:
            mock_docs = [
                {"text": "Patient diagnosed with severe hypertension, requiring immediate medication adjustment.", "translated": "Paciente diagnosticado con hipertensión severa, requiriendo ajuste inmediato de medicación.", "lang": "es"},
                {"text": "The patient presents with symptoms of Type 2 Diabetes Mellitus, including elevated blood glucose levels.", "translated": "El paciente presenta síntomas de Diabetes Mellitus tipo 2, incluyendo niveles elevados de glucosa en sangre.", "lang": "es"},
                {"text": "Follow-up appointment scheduled for next week to review treatment plan.", "translated": "Cita de seguimiento programada para la próxima semana para revisar el plan de tratamiento.", "lang": "es"}
            ]
            for i, doc in enumerate(mock_docs):
                self.collection.add(
                    documents=[doc["text"]],
                    metadatas=[{"source_lang": "en", "target_lang": doc["lang"], "translated_text": doc["translated"]}],
                    ids=[f"doc_{i}"]
                )
            print("Mock exemplars added to ChromaDB.")

    def retrieve_exemplar(self, query_text: str, target_lang: str, n_results: int = 1) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"$and": [
                {"source_lang": "en"}, # Assuming English source for now
                {"target_lang": target_lang}
            ]}
        )
        
        if results and results["documents"] and results["documents"][0]:
            return [
                {"original": doc, "translated": meta["translated_text"], "distance": dist}
                for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
            ]
        return []

class TranslatorModels:
    def __init__(self):
        # General purpose multilingual model for initial pass
        self.nmt_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en") # Multi-to-English
        self.nmt_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
        
        # English-to-Multi for output
        self.en_mul_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
        self.en_mul_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
        
        # Load spacy model for entity recognition, if needed (placeholder)
        # self.spacy_nlp = spacy.load("en_core_web_sm") # For actual entity recognition
        self.spacy_nlp = spacy.load("en_core_web_sm") # Mock load
        print("Translator models and Spacy NLP loaded.")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        # A simplified translation flow. In reality, handling many-to-many is complex.
        # This mock assumes if source is not English, it first goes to English,
        # then from English to target_lang.

        if source_lang == target_lang: return text

        pivot_text = text
        if source_lang != "en":
            print(f"Translating from {source_lang} to English (pivot)...")
            inputs = self.nmt_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            translated_tokens = self.nmt_model.generate(**inputs, max_length=150)
            pivot_text = self.nmt_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            print(f"Pivot (EN): {pivot_text}")
        
        print(f"Translating from English (or source if EN) to {target_lang}...")
        inputs = self.en_mul_tokenizer(pivot_text, return_tensors="pt", truncation=True, padding=True)
        translated_tokens = self.en_mul_model.generate(**inputs, max_length=150)
        final_translation = self.en_mul_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        return final_translation

class TextProcessor:
    def __init__(self):
        self.spacy_nlp = spacy.load("en_core_web_sm") # Mock load

    def segment_text(self, text: str) -> List[str]:
        doc = self.spacy_nlp(text)
        return [str(sent) for sent in doc.sents]

    def extract_key_phrases(self, text: str) -> List[str]:
        # Placeholder for more advanced key phrase extraction
        doc = self.spacy_nlp(text)
        return [ent.text for ent in doc.ents] # Mock entity extraction

class PostEditor:
    def __init__(self):
        self.spell = SpellChecker()
        # self.grammar_tool = language_tool_python.LanguageTool("en-US") # For actual usage
        self.grammar_tool = language_tool_python # Mock tool

    def spell_check(self, text: str) -> str:
        words = nltk.word_tokenize(text)
        corrected_words = [self.spell.correction(word) for word in words]
        return " ".join(corrected_words)

    def grammar_check(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        matches = self.grammar_tool.check(text)
        corrections = []
        corrected_text = text
        # Apply corrections from grammar tool (simplified)
        for match in matches:
            if match.get("replacements"): # type: ignore
                # This part would need careful implementation to apply replacements correctly
                # For now, just collect messages.
                corrections.append({"message": match["message"], "replacements": match["replacements"]})
        return corrected_text, corrections

    def ensure_terminology_consistency(self, original_text: str, translated_text: str, source_lang: str, target_lang: str, glossary: MedicalTerminology) -> str:
        # Simple consistency check: if a glossary term was used, ensure its translation appears
        # This is a highly simplified example.
        consistent_translation = translated_text
        identified_terms = glossary.identify_medical_terms(original_text, source_lang)
        for term in identified_terms:
            glossary_translation = glossary.get_translation(term, target_lang)
            if glossary_translation and glossary_translation not in consistent_translation:
                # If a term's translation is missing, try to insert it or flag
                print(f"Warning: Glossary term '{term}' (-> '{glossary_translation}') not found in translated text.")
                # For demo, just append. In real app, more intelligent insertion.
                consistent_translation += f" (Term missing: {glossary_translation})"
        return consistent_translation

# --- Core Translation Logic Orchestrator ---

class HealthBridgeTranslator:
    def __init__(self):
        self.medical_terminology = MedicalTerminology()
        self.exemplar_retriever = ExemplarRetriever()
        self.translator_models = TranslatorModels()
        self.text_processor = TextProcessor()
        self.post_editor = PostEditor()
        self.db_session = SessionLocal()

    def _get_db(self):
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def translate_medical_text(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        # 1. Augmented Prompting & Preprocessing
        print("Step 1: Augmented Prompting & Preprocessing")
        identified_terms = self.medical_terminology.identify_medical_terms(text, source_lang)
        print(f"Identified medical terms: {identified_terms}")

        exemplars = self.exemplar_retriever.retrieve_exemplar(text, target_lang)
        exemplar_context = ""
        if exemplars: 
            exemplar_context = f"\n\nRelevant medical exemplar (for context):\nOriginal: {exemplars[0]['original']}\nTranslated: {exemplars[0]['translated']}"
            print(f"Retrieved exemplar: {exemplars[0]['original'][:50]}...")

        # Prepend identified terms and exemplar context to the input for the NMT model
        # (This is a simplified way of 'augmenting' the prompt)
        augmented_text = f"Translate the following medical text. Key terms: {', '.join(identified_terms)}.{exemplar_context}\n\nText: {text}"
        print(f"Augmented text for translation: {augmented_text[:200]}...")

        # 2. Strategic Planning & Decomposition
        print("Step 2: Strategic Planning & Decomposition")
        segments = self.text_processor.segment_text(augmented_text) # Segment the *augmented* text
        print(f"Text segmented into {len(segments)} parts.")

        draft_translations = []
        for i, segment in enumerate(segments):
            print(f"Translating segment {i+1}/{len(segments)}: {segment[:50]}...")
            # Apply glossary translations first for known terms within the segment
            processed_segment = segment
            for term in identified_terms:
                glossary_translation = self.medical_terminology.get_translation(term, target_lang)
                if glossary_translation and term in processed_segment:
                    processed_segment = processed_segment.replace(term, glossary_translation)
            
            draft_translation = self.translator_models.translate(processed_segment, source_lang, target_lang)
            draft_translations.append(draft_translation)

        full_draft_translation = " ".join(draft_translations)
        print(f"Full draft translation: {full_draft_translation[:200]}...")

        # Cohesion and Consistency Check (simplified)
        final_translation = self.post_editor.ensure_terminology_consistency(
            text, full_draft_translation, source_lang, target_lang, self.medical_terminology
        )
        print(f"Translation after consistency check: {final_translation[:200]}...")

        # Risk-based Prioritization (mock)
        is_critical_content = any(term in text.lower() for term in ["diagnosis", "dosage", "allergy"])
        confidence_score = 0.8 # Mock confidence
        if is_critical_content: confidence_score -= 0.2 # Lower confidence for critical content, indicating need for review
        if not exemplars: confidence_score -= 0.1 # Lower if no exemplars found
        
        # 3. Human-in-the-Loop & Iterative Refinement (integrated via UI and feedback storage)
        # The actual human feedback loop happens in Gradio and saves via FastAPI.
        
        return {
            "original_text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "translated_text": final_translation,
            "identified_terms": identified_terms,
            "exemplar_used": bool(exemplars),
            "confidence_score": max(0.1, round(confidence_score, 2)), # Ensure > 0
            "needs_human_review": is_critical_content or (confidence_score < 0.7)
        }
    
    def save_feedback(self, feedback_data: Dict[str, Any]):
        session = SessionLocal()
        feedback_entry = TranslationFeedback(
            original_text=feedback_data["original_text"],
            machine_translation=feedback_data["machine_translation"],
            human_correction=feedback_data.get("human_correction"),
            source_lang=feedback_data["source_lang"],
            target_lang=feedback_data["target_lang"],
            is_accurate=feedback_data.get("is_accurate"),
            feedback_notes=feedback_data.get("feedback_notes")
        )
        session.add(feedback_entry)
        session.commit()
        session.close()
        print("Feedback saved successfully.")

# --- FastAPI Backend ---

app = FastAPI(title="Global HealthBridge Translator API")
healthbridge_translator = HealthBridgeTranslator()

# Pydantic models for request/response bodies
class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "auto" # Not fully implemented 'auto' in mock
    target_lang: str

class TranslationResponse(BaseModel):
    original_text: str
    source_lang: str
    target_lang: str
    translated_text: str
    identified_terms: List[str]
    exemplar_used: bool
    confidence_score: float
    needs_human_review: bool

class FeedbackRequest(BaseModel):
    original_text: str
    machine_translation: str
    human_correction: str | None = None
    source_lang: str
    target_lang: str
    is_accurate: bool | None = None
    feedback_notes: str | None = None

@app.post("/translate", response_model=TranslationResponse)
async def translate_text_api(request: TranslationRequest):
    try:
        result = healthbridge_translator.translate_medical_text(
            request.text, request.source_lang, request.target_lang
        )
        return TranslationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback_api(request: FeedbackRequest):
    try:
        healthbridge_translator.save_feedback(request.dict())
        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Global HealthBridge Translator"}

# --- Gradio UI ---

def gradio_translate(text: str, source_lang: str, target_lang: str) -> Tuple[str, str, str, str]:
    # This function will call the FastAPI backend
    # In a real app, you'd use requests.post to call your FastAPI endpoint
    # For this single file mock, we'll directly call the HealthBridgeTranslator instance.
    
    if not text or not target_lang:
        return "", "", "", "Please provide text and target language."
    
    try:
        print(f"Gradio: Calling translate_medical_text with '{text[:50]}...', {source_lang}, {target_lang}")
        result = healthbridge_translator.translate_medical_text(text, source_lang, target_lang)
        
        translated_text = result["translated_text"]
        confidence = f"Confidence: {result['confidence_score']:.2f}"
        review_needed = "" 
        if result["needs_human_review"]:
            review_needed = "*** HUMAN REVIEW RECOMMENDED ***"

        # Prepare output for Gradio
        info_message = f"Identified Terms: {', '.join(result['identified_terms'])}\nExemplar Used: {result['exemplar_used']}\nConfidence: {result['confidence_score']:.2f}\nNeeds Review: {result['needs_human_review']}"

        return translated_text, confidence, review_needed, info_message
    except Exception as e:
        return "", "", "", f"Error during translation: {str(e)}"

def gradio_submit_feedback(original_text: str, machine_translation: str, human_correction: str, source_lang: str, target_lang: str, is_accurate: bool, feedback_notes: str) -> str:
    feedback_data = {
        "original_text": original_text,
        "machine_translation": machine_translation,
        "human_correction": human_correction,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "is_accurate": is_accurate,
        "feedback_notes": feedback_notes
    }
    try:
        # Directly call save_feedback for this mock
        healthbridge_translator.save_feedback(feedback_data)
        return "Feedback submitted successfully! Thank you."
    except Exception as e:
        return f"Error submitting feedback: {str(e)}"

# Define Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Global HealthBridge Translator
        Translate medical documents with enhanced accuracy, leveraging specialized AI patterns.
        """
    )

    with gr.Row():
        with gr.Column():
            original_text_input = gr.Textbox(label="Original Medical Text", lines=10, placeholder="Enter medical text to translate...")
            source_lang_dropdown = gr.Dropdown(label="Source Language", choices=["en", "es", "fr", "de"], value="en") # Mock common languages
            target_lang_dropdown = gr.Dropdown(label="Target Language", choices=["es", "fr", "de", "en"], value="es")
            translate_btn = gr.Button("Translate")

        with gr.Column():
            translated_text_output = gr.Textbox(label="Translated Text", lines=10, interactive=True)
            confidence_output = gr.Textbox(label="Confidence Score", interactive=False)
            review_needed_output = gr.Textbox(label="Human Review Status", interactive=False, visible=True, elem_classes="warning_text")
            info_output = gr.Textbox(label="Translation Details", interactive=False, lines=4)

    translate_btn.click(
        gradio_translate,
        inputs=[original_text_input, source_lang_dropdown, target_lang_dropdown],
        outputs=[translated_text_output, confidence_output, review_needed_output, info_output]
    )

    gr.Markdown(
        """
        ### Human-in-the-Loop Feedback
        Help us improve by providing feedback on the translation quality.
        """
    )
    with gr.Row():
        with gr.Column():
            feedback_human_correction = gr.Textbox(label="Your Correction (if any)", lines=5, placeholder="Enter corrections here...")
            feedback_is_accurate = gr.Radio(label="Is the Machine Translation Accurate?", choices=[True, False], value=True)
            feedback_notes = gr.Textbox(label="Additional Notes", lines=3, placeholder="e.g., specific terms were mistranslated")
            submit_feedback_btn = gr.Button("Submit Feedback")
        with gr.Column():
            feedback_status_output = gr.Textbox(label="Feedback Status", interactive=False)
    
    submit_feedback_btn.click(
        gradio_submit_feedback,
        inputs=[
            original_text_input,
            translated_text_output, # Machine translation output
            feedback_human_correction,
            source_lang_dropdown,
            target_lang_dropdown,
            feedback_is_accurate,
            feedback_notes
        ],
        outputs=feedback_status_output
    )

# --- Main Application Entry Point ---
if __name__ == "__main__":
    print("Starting Global HealthBridge Translator...")
    print("Note: This is a mocked version for demonstration. Full functionality requires installing external libraries and potentially specialized models.")
    
    # In a real application, you'd run FastAPI and Gradio separately or use a tool like 'gunicorn'
    # to serve FastAPI and then run Gradio on top, or embed Gradio within FastAPI.
    # For this single file, we run Gradio and indicate FastAPI is also 'running' conceptually.

    print("FastAPI backend is conceptually running on http://127.0.0.1:8000 (mocked endpoints available).")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)

    # To run FastAPI independently (uncomment if real FastAPI is installed and mocks are removed):
    # uvicorn.run(app, host="127.0.0.1", port=8000)

