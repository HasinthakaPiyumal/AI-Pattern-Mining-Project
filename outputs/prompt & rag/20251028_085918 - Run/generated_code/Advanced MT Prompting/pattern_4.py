
import nltk
from nltk.tokenize import sent_tokenize
from langdetect import detect, DetectorFactory
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb
import os

# Ensure reproducibility for langdetect
DetectorFactory.seed = 0

# Download necessary NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")

# --- Configuration --- #
TARGET_LANGUAGE = "en"  # Target language for summaries
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-es-en"  # Example for Spanish to English
# You might need multiple models or a language-agnostic one depending on input languages
EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"
CHROMA_DB_PATH = "./chroma_db"

# --- 1. Input & Pre-processing Module ---

def extract_text_from_document(filepath: str) -> str:
    """
    Placeholder function to extract text from various document formats.
    In a real application, this would use PyPDF2, python-docx, etc.
    """
    print(f"[INFO] Extracting text from {filepath}...")
    # Simulate text extraction for a dummy file
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return ""

def detect_source_language(text: str) -> str:
    """
    Detects the language of the input text.
    """
    if not text.strip():
        return "unknown"
    try:
        lang = detect(text)
        print(f"[INFO] Detected language: {lang}")
        return lang
    except Exception as e:
        print(f"[ERROR] Language detection failed: {e}")
        return "unknown"

def decompose_text(text: str) -> list[str]:
    """
    Decomposes the text into a list of sentences/chunks.
    """
    if not text.strip():
        return []
    sentences = sent_tokenize(text)
    print(f"[INFO] Decomposed text into {len(sentences)} sentences.")
    return sentences

# --- 2. Context Augmentation Layer ---

class ContextAugmentor:
    def __init__(self):
        print("[INFO] Initializing Context Augmentor...")
        self.translator = pipeline("translation", model=TRANSLATION_MODEL_NAME)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.medical_exemplars_collection = self._init_exemplar_db()
        self.medical_dictionary = {
            "en": {
                "hypertension": "High blood pressure.",
                "diabetes": "A condition in which the body does not properly process food for use as energy.",
                "myocardial infarction": "Heart attack."
            },
            "es": {
                "hipertensión": "Presión arterial alta.",
                "diabetes": "Una afección en la que el cuerpo no procesa adecuadamente los alimentos para usarlos como energía.",
                "infarto de miocardio": "Ataque al corazón."
            }
            # Add more languages and terms as needed
        }

    def _init_exemplar_db(self):
        """
        Initializes or loads the ChromaDB collection for medical exemplars.
        Populates with dummy data if empty.
        """
        collection_name = "medical_exemplars"
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            print(f"[INFO] Loaded existing ChromaDB collection: {collection_name}")
        except:
            print(f"[INFO] Creating new ChromaDB collection: {collection_name}")
            collection = self.chroma_client.create_collection(name=collection_name)
            # Add some dummy medical exemplars for demonstration
            exemplar_texts = [
                "Clinical trials for a new drug targeting Type 2 Diabetes showed significant reduction in HbA1c levels.",
                "A comprehensive review of hypertension management guidelines emphasizes lifestyle modifications and combination therapy.",
                "Recent advancements in oncology research focus on immunotherapy for metastatic melanoma with promising outcomes."
            ]
            exemplar_ids = [f"med_exp_{i}" for i in range(len(exemplar_texts))]
            exemplar_embeddings = self.embedder.encode(exemplar_texts).tolist()
            collection.add(documents=exemplar_texts, embeddings=exemplar_embeddings, ids=exemplar_ids)
            print(f"[INFO] Populated ChromaDB with {len(exemplar_texts)} dummy exemplars.")
        return collection

    def translate_to_pivot(self, text: str, source_lang: str) -> str:
        """
        Translates text to a high-resource pivot language (e.g., English).
        """
        if source_lang == TARGET_LANGUAGE or source_lang == "unknown":
            return text
        print(f"[INFO] Translating text from {source_lang} to {TARGET_LANGUAGE}...")
        # Note: The chosen model 'Helsinki-NLP/opus-mt-es-en' only translates from Spanish to English.
        # For a truly multi-lingual platform, you'd need a more robust translation strategy (e.g., Google Translate API, or multiple models).
        # For this example, we'll assume the input is Spanish if not English.
        if source_lang == "es" and TARGET_LANGUAGE == "en":
             translated_text = self.translator(text, max_length=512)[0]["translation_text"]
             return translated_text
        else:
            print(f"[WARNING] No specific translation model for {source_lang} to {TARGET_LANGUAGE} in this example. Returning original text.")
            return text # Fallback


    def retrieve_cross_lingual_exemplars(self, query_text: str, top_k: int = 2) -> list[str]:
        """
        Retrieves semantically similar medical exemplars from the vector database.
        """
        if not query_text.strip():
            return []
        print(f"[INFO] Retrieving medical exemplars for query: '{query_text[:50]}...' ")
        query_embedding = self.embedder.encode([query_text]).tolist()
        results = self.medical_exemplars_collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents"]
        )
        return results["documents"][0] if results["documents"] else []

    def medical_terminology_lookup(self, term: str, lang: str) -> str:
        """
        Looks up a medical term in the multilingual dictionary.
        """
        term_lower = term.lower()
        if lang in self.medical_dictionary and term_lower in self.medical_dictionary[lang]:
            print(f"[INFO] Found dictionary definition for '{term}' ({lang}).")
            return self.medical_dictionary[lang][term_lower]
        print(f"[INFO] No dictionary definition found for '{term}' ({lang}).")
        return ""

# --- 3. Strategic Planning & Generation Module ---

def generate_summary_and_analysis(text_chunks: list[str], augmented_context: dict) -> str:
    """
    Simulates the generation of a medical summary and analysis using an LLM.
    In a real scenario, this would involve prompting a powerful LLM (e.g., OpenAI GPT-4, Llama-2).
    """
    print("[INFO] Generating summary and analysis...")
    combined_input = "\n".join(text_chunks)

    # Incorporate augmented context
    context_str = ""
    if augmented_context.get("translated_text") and augmented_context["translated_text"] != combined_input:
        context_str += f"Translated Input: {augmented_context['translated_text']}\n"
    if augmented_context.get("exemplars"):
        context_str += "Relevant Medical Exemplars:\n" + "\n".join([f"- {e}" for e in augmented_context["exemplars"]]) + "\n"
    if augmented_context.get("terminology_definitions"):
        context_str += "Medical Terminology Definitions:\n" + "\n".join([f"- {t}: {d}" for t, d in augmented_context["terminology_definitions"].items()]) + "\n"

    prompt = f"Given the following medical research text and supplementary context, provide a concise summary and a brief analysis of its key findings in English. Focus on clinical relevance and potential implications.\n\nMedical Text:\n{combined_input}\n\n{context_str}\nSummary and Analysis:"

    # Placeholder for LLM call
    # In a real application:
    # from openai import OpenAI
    # client = OpenAI()
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content

    # Dummy generation for demonstration
    dummy_summary = f"[DUMMY LLM OUTPUT] This document discusses a medical topic. Key information from the original text has been processed. The augmented context provided details such as: \n- Translation: {augmented_context.get('translated_text', 'N/A')}\n- Exemplars: {augmented_context.get('exemplars', 'N/A')}\n- Terminology: {augmented_context.get('terminology_definitions', 'N/A')}\nFurther analysis would delve into specific findings and their implications based on the original content and retrieved context."
    return dummy_summary

# --- 4. Iterative Feedback & Refinement Module ---

def detect_ambiguities(generated_text: str) -> list[str]:
    """
    Detects potential ambiguities or inconsistencies in the generated text.
    This is a simplified example using keyword matching.
    In a real system, this would involve more sophisticated NLP, potentially another LLM call,
    or domain-specific rule sets.
    """
    print("[INFO] Detecting ambiguities...")
    ambiguity_flags = []
    common_uncertainty_phrases = ["potentially", "may indicate", "could suggest", "unclear"]
    for phrase in common_uncertainty_phrases:
        if phrase in generated_text.lower():
            ambiguity_flags.append(f"Phrase '{phrase}' detected, suggesting potential ambiguity.")
    
    # Example: Check for missing crucial information (very simplified)
    if "key findings" not in generated_text.lower() and "implications" not in generated_text.lower():
        ambiguity_flags.append("Summary might be missing explicit mention of key findings or implications.")

    if ambiguity_flags:
        print(f"[WARNING] Detected {len(ambiguity_flags)} potential ambiguities.")
    return ambiguity_flags

def refine_generation(original_text: str, augmented_context: dict, feedback: str) -> str:
    """
    Simulates re-generation based on feedback.
    In a real system, feedback would be used to adjust prompts or provide additional context
    to the LLM for a more accurate re-generation.
    """
    print(f"[INFO] Refining generation based on feedback: '{feedback}'")
    # For this example, we'll just append the feedback to a new dummy generation.
    refined_output = generate_summary_and_analysis(
        text_chunks=decompose_text(original_text), # Re-decompose original text
        augmented_context=augmented_context # Re-use augmented context
    ) + f"\n\n[REFINED based on feedback: {feedback}]"
    return refined_output

# --- Main Orchestration Function ---

def process_medical_document(filepath: str) -> dict:
    """
    Orchestrates the entire process of medical research synthesis.
    """
    print(f"\n=== Starting processing for: {filepath} ===")
    
    # 1. Input & Pre-processing
    original_text = extract_text_from_document(filepath)
    if not original_text:
        return {"error": "Could not read or empty document.", "filepath": filepath}

    source_lang = detect_source_language(original_text)
    text_chunks = decompose_text(original_text)

    context_augmentor = ContextAugmentor()

    # 2. Context Augmentation Layer
    augmented_context = {
        "translated_text": "",
        "exemplars": [],
        "terminology_definitions": {}
    }

    if source_lang != TARGET_LANGUAGE and source_lang != "unknown":
        # Translate a representative part or the whole text for context
        # For simplicity, translating the first chunk here
        if text_chunks:
            augmented_context["translated_text"] = context_augmentor.translate_to_pivot(text_chunks[0], source_lang)
        
    # Retrieve exemplars based on the original (or translated) text
    query_for_exemplars = augmented_context["translated_text"] if augmented_context["translated_text"] else original_text
    augmented_context["exemplars"] = context_augmentor.retrieve_cross_lingual_exemplars(query_for_exemplars)

    # Look up some dummy terms for demonstration
    # In a real app, you'd extract key terms using NLP from the text
    dummy_terms = ["hypertension", "diabetes"]
    for term in dummy_terms:
        definition = context_augmentor.medical_terminology_lookup(term, TARGET_LANGUAGE)
        if definition:
            augmented_context["terminology_definitions"][term] = definition
        if source_lang != TARGET_LANGUAGE:
            translated_term_def = context_augmentor.medical_terminology_lookup(term, source_lang) # If you want to show original lang definition
            if translated_term_def and translated_term_def != definition:
                 augmented_context["terminology_definitions"][f"{term} ({source_lang})"] = translated_term_def

    print("[INFO] Augmented Context:", augmented_context)

    # 3. Strategic Planning & Generation
    initial_summary_analysis = generate_summary_and_analysis(text_chunks, augmented_context)
    print("\n--- Initial Generated Output ---")
    print(initial_summary_analysis)

    # 4. Iterative Feedback & Refinement
    feedback_loop_count = 0
    max_feedback_loops = 2
    current_output = initial_summary_analysis

    while feedback_loop_count < max_feedback_loops:
        ambiguities = detect_ambiguities(current_output)
        if not ambiguities:
            print("\n[INFO] No significant ambiguities detected. Exiting feedback loop.")
            break

        feedback_loop_count += 1
        print(f"\n--- Entering Feedback Loop {feedback_loop_count} ---")
        human_feedback = f"Please clarify the following: {', '.join(ambiguities)}. Make sure to explicitly state the main research question."
        print(f"[SIMULATED HUMAN FEEDBACK] {human_feedback}")

        current_output = refine_generation(original_text, augmented_context, human_feedback)
        print("\n--- Refined Generated Output ---")
        print(current_output)

    print(f"\n=== Finished processing for: {filepath} ===")
    return {
        "filepath": filepath,
        "final_output": current_output,
        "feedback_loops": feedback_loop_count
    }

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy medical research paper file in Spanish
    dummy_spanish_medical_paper = """
    Estudio sobre la eficacia de un nuevo fármaco para la hipertensión.
    Los resultados preliminares muestran una reducción significativa de la presión arterial en pacientes con hipertensión esencial. Se observaron efectos secundarios leves como mareos. Este estudio sugiere un potencial prometedor para el tratamiento de la hipertensión, pero se necesitan ensayos clínicos más grandes para confirmar estos hallazgos y evaluar la seguridad a largo plazo. La diabetes mellitus no fue un foco principal de este estudio, aunque algunos pacientes presentaban comorbilidades.
    """
    dummy_filepath = "dummy_medical_paper_es.txt"
    with open(dummy_filepath, "w", encoding="utf-8") as f:
        f.write(dummy_spanish_medical_paper)
    print(f"Created dummy file: {dummy_filepath}")

    # Process the dummy document
    result = process_medical_document(dummy_filepath)
    print("\n--- Final Result ---")
    print(result)

    # Clean up dummy file
    os.remove(dummy_filepath)
    print(f"Cleaned up dummy file: {dummy_filepath}")

    # Clean up chromadb directory
    if os.path.exists(CHROMA_DB_PATH):
        import shutil
        shutil.rmtree(CHROMA_DB_PATH)
        print(f"Cleaned up ChromaDB directory: {CHROMA_DB_PATH}")

