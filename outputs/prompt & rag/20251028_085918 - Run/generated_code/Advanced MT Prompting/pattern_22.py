import os
from dotenv import load_dotenv
import PyPDF2
from docx import Document
from langdetect import detect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2, IndexFlatIP, IndexFlatLS, read_index, write_index
import numpy as np
from openai import OpenAI
from google.cloud import translate_v2 as translate
import gradio as gr
import re

load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") # Path to your service account key file

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
translate_client = translate.Client()

# --- 1. Input & Document Processing Layer ---
class DocumentLoader:
    def load_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() or ""
        return text

    def load_docx(self, file_path):
        doc = Document(file_path)
        text = [paragraph.text for paragraph in doc.paragraphs]
        return "\n".join(text)

    def load_txt(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def load_document(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".pdf":
            return self.load_pdf(file_path)
        elif file_extension == ".docx":
            return self.load_docx(file_path)
        elif file_extension == ".txt":
            return self.load_txt(file_path)
        else:
            raise ValueError("Unsupported file type. Only PDF, DOCX, and TXT are supported.")

class TextProcessor:
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
        text = text.strip()
        return text

    def detect_language(self, text):
        try:
            return detect(text)
        except Exception:
            return "unknown"

# --- 2. Pre-processing & Chunking Layer ---
class Preprocessor:
    def __init__(self, target_pivot_lang="en"):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            add_start_index=True,
        )
        self.target_pivot_lang = target_pivot_lang

    def chunk_text(self, text):
        return self.text_splitter.create_documents([text])

    def pre_translate_if_low_resource(self, text, source_lang):
        if source_lang not in ["en", "es", "fr", "de", "zh", "ar"] and source_lang != self.target_pivot_lang: # Simplified check for low-resource
            print(f"Pre-translating from {source_lang} to {self.target_pivot_lang}")
            result = translate_client.translate(text, target_language=self.target_pivot_lang)
            return result['translatedText']
        return text # No pre-translation needed

# --- 3. Context Augmentation & Retrieval Layer ---
class MedicalKnowledgeBase:
    def __init__(self, faiss_index_path="medical_kb.faiss", model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.faiss_index_path = faiss_index_path
        self.index = None
        self.documents = []
        self._load_or_build_index()
        # Simplified Medical Dictionary - for explicit lookup
        self.medical_dictionary = {
            "hypertension": "A condition in which the force of the blood against the artery walls is too high.",
            "diabetes": "A chronic condition that affects how your body turns food into energy.",
            "myocardial infarction": "A heart attack; permanent damage to the heart muscle, usually caused by a blocked artery.",
            "oncology": "The study and treatment of tumors.",
            "pediatrics": "The branch of medicine dealing with children and their diseases."
        }

    def _load_or_build_index(self):
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.faiss_index_path + ".docs"):
            print("Loading FAISS index and documents...")
            self.index = read_index(self.faiss_index_path)
            with open(self.faiss_index_path + ".docs", "r", encoding="utf-8") as f:
                self.documents = f.read().splitlines()
            print(f"Loaded {len(self.documents)} documents into FAISS index.")
        else:
            print("Building new FAISS index (using dummy data)...")
            # Dummy medical exemplars for demonstration
            dummy_exemplars = [
                "Hypertension management often involves lifestyle changes and medication, such as ACE inhibitors or diuretics.",
                "Diabetic retinopathy is a complication of diabetes that affects the eyes, potentially leading to blindness.",
                "Early diagnosis of myocardial infarction is crucial for effective treatment, often involving angioplasty.",
                "Oncology departments specialize in various cancer treatments including chemotherapy, radiation, and surgery.",
                "Pediatric care focuses on the health and medical care of infants, children, and adolescents.",
                "The cardiovascular system comprises the heart and blood vessels.",
                "Neurological disorders affect the brain, spinal cord, and all nerves."
            ]
            self.documents = dummy_exemplars
            embeddings = self.embedding_model.encode(self.documents)
            dimension = embeddings.shape[1]
            self.index = IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype('float32'))
            write_index(self.index, self.faiss_index_path)
            with open(self.faiss_index_path + ".docs", "w", encoding="utf-8") as f:
                f.write("\n".join(self.documents))
            print(f"Built FAISS index with {len(self.documents)} documents.")

    def retrieve_exemplars(self, query_text, k=3):
        if not self.index:
            return []
        query_embedding = self.embedding_model.encode([query_text]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        retrieved_docs = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        return retrieved_docs

    def lookup_definition(self, term):
        return self.medical_dictionary.get(term.lower(), None)

# --- 4. Core Translation & Refinement Layer ---
class MultiStrategyLLMOrchestrator:
    def __init__(self, knowledge_base, llm_model="gpt-4o-mini"):
        self.knowledge_base = knowledge_base
        self.llm_model = llm_model

    def _build_prompt(self, original_chunk, pre_translated_chunk, retrieved_exemplars, target_language, source_language):
        prompt_parts = []
        prompt_parts.append(f"You are an expert medical translator. Translate the following medical text from {source_language} to {target_language}.\n")
        prompt_parts.append("Maintain clinical accuracy, consistency in terminology, and adhere to the nuances of medical language.\n")

        if pre_translated_chunk and pre_translated_chunk != original_chunk:
            prompt_parts.append(f"An initial translation to a high-resource pivot language was performed: '{pre_translated_chunk}'\n")
            prompt_parts.append("Use this as a reference but prioritize accurate translation of the original source text.")

        if retrieved_exemplars:
            prompt_parts.append("\n--- Retrieved Medical Exemplars (for context and terminology) ---\n")
            for i, exemplar in enumerate(retrieved_exemplars):
                prompt_parts.append(f"Exemplar {i+1}: {exemplar}\n")
            prompt_parts.append("--- End of Exemplars ---\n")
            prompt_parts.append("Leverage these exemplars for consistent terminology and phrasing.")

        # Try to find medical terms in the original chunk and look them up
        found_terms = []
        for term in self.knowledge_base.medical_dictionary:
            if re.search(r'\b' + re.escape(term) + r'\b', original_chunk.lower()):
                definition = self.knowledge_base.lookup_definition(term)
                if definition: # Ensure a definition was found before adding
                    found_terms.append(f"'{term}': {definition}")
        
        if found_terms:
            prompt_parts.append("\n--- Relevant Medical Terminology Definitions ---\n")
            prompt_parts.append("\n".join(found_terms))
            prompt_parts.append("--- End of Definitions ---\n")
            prompt_parts.append("Ensure these definitions guide your translation for accuracy.")

        prompt_parts.append(f"\nOriginal Medical Text ({source_language}):\n```\n{original_chunk}\n```\n")
        prompt_parts.append(f"Translated Medical Text ({target_language}):\n")

        return " ".join(prompt_parts)

    def translate_chunk(self, original_chunk, pre_translated_chunk, target_language, source_language):
        retrieved_exemplars = self.knowledge_base.retrieve_exemplars(original_chunk)
        
        prompt = self._build_prompt(original_chunk, pre_translated_chunk, retrieved_exemplars, target_language, source_language)
        
        try:
            response = openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a highly accurate medical translator assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during LLM translation: {e}")
            return f"[Translation Error: {e}] Original: {original_chunk}"

class IterativeRefinementModule:
    def __init__(self, medical_dictionary):
        self.medical_dictionary = medical_dictionary
        self.terminology_cache = {}

    def check_consistency(self, original_text_chunks, translated_text_chunks, target_language):
        issues = []
        # Simplified consistency check: Look for specific terms and their translations
        # In a real system, this would involve NLP for entity recognition and cross-referencing
        for i, (orig_chunk, trans_chunk) in enumerate(zip(original_text_chunks, translated_text_chunks)):
            for term_en, def_en in self.medical_dictionary.items():
                # Placeholder: if 'hypertension' is in original, expect a specific translation in target
                # This requires a more sophisticated mapping or neural alignment
                if term_en in orig_chunk.lower():
                    # Example: Check if a basic form of the term appears in translation
                    # This is very basic and needs improvement for real-world use
                    if target_language == "es" and "hipertensión" not in trans_chunk.lower():
                        issues.append(f"Chunk {i+1}: Possible inconsistency for '{term_en}'. Expected 'hipertensión' in Spanish, but not found.")
                    elif target_language == "fr" and "hypertension" not in trans_chunk.lower(): # French uses similar term
                         issues.append(f"Chunk {i+1}: Possible inconsistency for '{term_en}'. Expected 'hypertension' in French, but not found.")

        return issues

    def human_feedback_mechanism(self, translation_id, suggested_correction):
        print(f"Received human feedback for translation {translation_id}: {suggested_correction}")
        # In a real system, this would update a database, retrain models, or refine RAG data
        pass

# --- 5. Post-processing & Output Layer ---
class Postprocessor:
    def reassemble_chunks(self, translated_chunks):
        return " ".join(translated_chunks)

    def format_output(self, text, original_format=None):
        # Basic formatting, more complex formatting would require analyzing original document structure
        # and regenerating based on that. For now, just ensures readability.
        formatted_text = text.replace(". ", ".\n\n") # Add line breaks after sentences for better readability
        return formatted_text

# --- Main Application Logic --- 
class MedicalTranslationAssistant:
    def __init__(self):
        self.loader = DocumentLoader()
        self.text_processor = TextProcessor()
        self.preprocessor = Preprocessor()
        self.knowledge_base = MedicalKnowledgeBase()
        self.orchestrator = MultiStrategyLLMOrchestrator(self.knowledge_base)
        self.refiner = IterativeRefinementModule(self.knowledge_base.medical_dictionary)
        self.postprocessor = Postprocessor()

    def translate_document(self, file_path, target_language):
        original_raw_text = self.loader.load_document(file_path)
        cleaned_text = self.text_processor.clean_text(original_raw_text)
        source_lang = self.text_processor.detect_language(cleaned_text)
        
        if source_lang == "unknown":
            return "Error: Could not detect source language. Please ensure the document contains sufficient text."
        
        print(f"Detected source language: {source_lang}")

        document_chunks = self.preprocessor.chunk_text(cleaned_text)
        original_chunks_content = [chunk.page_content for chunk in document_chunks]
        
        translated_chunks = []
        consistency_issues = []

        for i, chunk_obj in enumerate(document_chunks):
            original_chunk_text = chunk_obj.page_content
            
            pre_translated_chunk = self.preprocessor.pre_translate_if_low_resource(
                original_chunk_text, source_lang
            )
            
            translated_chunk = self.orchestrator.translate_chunk(
                original_chunk_text, pre_translated_chunk, target_language, source_lang
            )
            translated_chunks.append(translated_chunk)
            print(f"Translated chunk {i+1}/{len(document_chunks)}")

        # Run consistency checks (simplified)
        consistency_issues = self.refiner.check_consistency(original_chunks_content, translated_chunks, target_language)
        if consistency_issues:
            print("\n--- Consistency Issues Detected ---")
            for issue in consistency_issues:
                print(issue)
            print("-----------------------------------\n")

        final_translated_text = self.postprocessor.reassemble_chunks(translated_chunks)
        formatted_output = self.postprocessor.format_output(final_translated_text)

        return formatted_output

# --- Gradio Interface ---
def translate_ui(file, target_lang):
    if file is None:
        return "Please upload a document."
    
    assistant = MedicalTranslationAssistant()
    try:
        translated_content = assistant.translate_document(file.name, target_lang)
        return translated_content
    except Exception as e:
        return f"An error occurred during translation: {str(e)}"

if __name__ == "__main__":
    # Example of how to use the assistant directly (without Gradio)
    # assistant = MedicalTranslationAssistant()
    # test_file = "path/to/your/medical_document.pdf" # Replace with a real path
    # target_language = "es"
    # translated_text = assistant.translate_document(test_file, target_language)
    # print(translated_text)

    # Gradio Interface
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY is not set. LLM translation will fail.")
    if not GOOGLE_APPLICATION_CREDENTIALS or not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set or file not found. Google Translate (for pivot lang) will fail.")

    print("Starting Gradio interface...")
    interface = gr.Interface(
        fn=translate_ui,
        inputs=[
            gr.File(label="Upload Medical Document (PDF, DOCX, TXT)"),
            gr.Dropdown(
                ["es", "fr", "de", "zh", "ar"],  # Example target languages
                label="Target Language",
                value="es"
            )
        ],
        outputs=gr.Textbox(label="Translated Medical Document", lines=20),
        title="Medical Document Translation Assistant",
        description="Upload a medical document and get an enhanced translation using multi-strategy AI."
    )

    interface.launch(share=True)
