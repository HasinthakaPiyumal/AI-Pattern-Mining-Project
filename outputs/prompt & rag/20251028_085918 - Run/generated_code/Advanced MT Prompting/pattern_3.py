import re
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Ensure NLTK punkt tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

class MedicalReportProcessor:
    """Processes raw medical report text."""

    def load_report(self, text: str) -> str:
        """Loads a raw text string representing a medical report."""
        return text

    def preprocess_text(self, text: str) -> str:
        """Cleans the text by removing extra whitespace and special characters."""
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        text = re.sub(r'[^a-zA-Z0-9.,;:\s]', '', text) # Remove most special chars, keep punctuation
        return text

class ContextAugmenter:
    """Enriches input text with external contextual information."""

    def __init__(self, medical_glossary: dict):
        """Initializes with a pre-loaded medical glossary."""
        self.medical_glossary = medical_glossary

    def identify_medical_terms(self, text: str) -> list:
        """Identifies potential medical terms from the text using the glossary."""
        identified_terms = []
        words = re.findall(r'\b\w+\b', text.lower())
        for word in set(words):
            if word in self.medical_glossary:
                identified_terms.append(word)
        return identified_terms

    def get_contextual_info(self, term: str, source_lang: str, target_lang: str) -> dict:
        """Retrieves definitions, synonyms, or common translations from the glossary.
           Simulates retrieving cross-lingual exemplars or dictionary definitions.
        """
        info = self.medical_glossary.get(term, {})
        # For simulation, we'll just return the available info.
        # In a real system, this would involve fetching specific language-pair info.
        return {"term": term, "context": info.get("definition", "No definition available."), "translation": info.get(f"translation_{target_lang}", info.get("translation_en", "No direct translation found."))}

    def translate_terms_to_high_resource(self, terms: list, source_lang: str, target_lang: str, llm_service) -> dict:
        """Translates identified terms to a high-resource language using LLMService if needed.
           For simplicity, this uses the glossary or a direct LLM call.
        """
        translated_terms_info = {}
        for term in terms:
            glossary_translation = self.medical_glossary.get(term, {}).get(f"translation_{target_lang}")
            if glossary_translation:
                translated_terms_info[term] = glossary_translation
            else:
                # Simulate translation if not in glossary
                # In a real scenario, this would be a specific term translation call to LLMService
                translated_terms_info[term] = llm_service.translate_text(term, source_lang, target_lang)
        return translated_terms_info


class ReportDecomposer:
    """Breaks down complex or lengthy medical reports into manageable segments."""

    def segment_report(self, report_text: str) -> list:
        """Divides the report into logical sections using NLTK sentence tokenization."""
        return nltk.sent_tokenize(report_text)

    def identify_key_sections(self, segments: list) -> dict:
        """Simulated: Identifies and labels key sections for prioritized processing.
           For this example, it just returns segments grouped as a 'Main Report'."""
        return {"Main Report": segments}

class LLMService:
    """Performs cross-lingual translation and summarization using transformer models."""

    def __init__(self, translation_model_name: str, summarization_model_name: str):
        """Loads appropriate models and tokenizers from `transformers`."""
        # Translation Model
        self.translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)

        # Summarization Model
        self.summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
        self.summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)

    def translate_text(self, text: str, source_lang: str, target_lang: str, augmented_context: dict = None) -> str:
        """Translates a given text segment."""
        # Prepend augmented context if available (simple concatenation for demo)
        if augmented_context:
            context_str = " ".join([f"{k}: {v}" for k, v in augmented_context.items() if k != "term"])
            text_to_translate = f"{context_str}. {text}" if context_str else text
        else:
            text_to_translate = text

        # MarianMT models typically expect source language prefix for tokenization
        # For example, '>>en<<' for English to Spanish translation
        # The `translation_model_name` already implies the direction (e.g., 'opus-mt-es-en')
        # We can explicitly set the source language for clarity if needed, but the model handles it.
        
        inputs = self.translation_tokenizer(text_to_translate, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated_tokens = self.translation_model.generate(**inputs, max_new_tokens=512)
        translated_text = self.translation_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text

    def summarize_text(self, text: str) -> str:
        """Generates a concise summary of the text segment."""
        inputs = self.summarization_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        summary_tokens = self.summarization_model.generate(**inputs, max_new_tokens=150, min_length=30, do_sample=False)
        summary_text = self.summarization_tokenizer.decode(summary_tokens[0], skip_special_tokens=True)
        return summary_text

class FeedbackHandler:
    """Simulates a feedback loop for refining translations and summaries."""

    def get_feedback(self, original_segment: str, translated_segment: str, identified_terms: list) -> dict:
        """Simulated: Evaluates the translated segment."
           In a real system, this would involve human review or automated metrics.
        """
        # Dummy feedback: Assume good if it contains at least one identified term (very basic check)
        feedback_score = 0.5
        for term in identified_terms:
            if term.lower() in translated_segment.lower():
                feedback_score = 0.8 # Higher score if a medical term is present
                break
        return {"quality_score": feedback_score, "notes": "Simulated feedback: translation appears reasonable."
                if feedback_score > 0.6 else "Simulated feedback: could be improved.",
                "suggested_revisions": []}

    def refine_translation(self, original_segment: str, current_translation: str, feedback: dict) -> str:
        """Simulated: Adjusts the translation based on the feedback."
           For this demo, we simply return the current translation unless feedback is poor.
        """
        if feedback.get("quality_score", 0) < 0.6:
            # In a real system, this would involve re-prompting the LLM or applying rule-based corrections.
            return f"[Refined based on feedback - ORIGINAL: {original_segment}] {current_translation}"
        return current_translation

class OutputGenerator:
    """Formats and presents the final translated and summarized medical report."""

    def generate_final_report(self, translated_sections: dict, summarized_text: str, augmented_terms_info: dict) -> str:
        """Compiles the translated sections and the overall summary into a readable report format."""
        report_parts = ["--- Cross-Lingual Medical Report ---"]
        
        report_parts.append("\n--- Summarized Report ---")
        report_parts.append(summarized_text)

        report_parts.append("\n--- Detailed Translation ---")
        for section_name, segments in translated_sections.items():
            report_parts.append(f"\n### {section_name} ###")
            report_parts.extend(segments)

        report_parts.append("\n--- Augmented Medical Terms Information ---")
        if augmented_terms_info:
            for original_term, info in augmented_terms_info.items():
                report_parts.append(f"- {original_term}: Definition: {info['context']}, Translated Term: {info['translation']}")
        else:
            report_parts.append("No specific medical terms augmented.")

        report_parts.append("\n-----------------------------------------")
        return "\n".join(report_parts)

# --- Main Orchestration Script ---
if __name__ == "__main__":
    # 0. Configuration and Initialization
    SOURCE_LANG = "es" # Example: Spanish
    TARGET_LANG = "en" # Example: English
    TRANSLATION_MODEL = f"Helsinki-NLP/opus-mt-{SOURCE_LANG}-{TARGET_LANG}"
    SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"

    # Dummy Medical Glossary (can be loaded from a file/DB in real-world)
    medical_glossary = {
        "fiebre": {"definition": "Elevación de la temperatura corporal por encima de lo normal.", "translation_en": "fever"},
        "diagnóstico": {"definition": "Identificación de la naturaleza de una enfermedad o problema mediante el examen de los síntomas.", "translation_en": "diagnosis"},
        "tratamiento": {"definition": "Conjunto de medios que se emplean para curar o aliviar una enfermedad.", "translation_en": "treatment"},
        "hipertensión": {"definition": "Presión arterial anormalmente alta.", "translation_en": "hypertension"},
        "diabetes": {"definition": "Enfermedad crónica que afecta la forma en que el cuerpo procesa el azúcar en la sangre.", "translation_en": "diabetes"},
        "cardiología": {"definition": "Rama de la medicina que estudia el corazón y sus enfermedades.", "translation_en": "cardiology"}
    }

    # Instantiate Modules
    processor = MedicalReportProcessor()
    augmenter = ContextAugmenter(medical_glossary)
    decomposer = ReportDecomposer()
    llm_service = LLMService(TRANSLATION_MODEL, SUMMARIZATION_MODEL)
    feedback_handler = FeedbackHandler()
    output_generator = OutputGenerator()

    # 1. Sample Medical Report (in Spanish)
    raw_medical_report = (
        "El paciente, de 65 años, fue admitido con fiebre alta (39.5°C) y dificultad respiratoria. "
        "Historial de hipertensión y diabetes tipo 2. El diagnóstico inicial sugiere neumonía bilateral. "
        "Se inició tratamiento con antibióticos de amplio espectro y oxigenoterapia. "
        "Los resultados de los cultivos están pendientes. Se recomienda consulta con cardiología debido al historial. "
        "La evolución del paciente se monitorizará de cerca en la unidad de cuidados intensivos. "
        "La glucemia se mantiene controlada con insulina. Se espera una pronta mejora con el tratamiento actual."
    )
    print("\n--- Original Medical Report (Spanish) ---")
    print(raw_medical_report)

    # 2. Process Input
    processed_report = processor.preprocess_text(processor.load_report(raw_medical_report))
    print("\n--- Processed Report ---")
    print(processed_report)

    # 3. Context Augmentation
    identified_terms = augmenter.identify_medical_terms(processed_report)
    print(f"\n--- Identified Medical Terms: {identified_terms} ---")
    
    augmented_terms_info = {}
    for term in identified_terms:
        info = augmenter.get_contextual_info(term, SOURCE_LANG, TARGET_LANG)
        # For terms not in the simple glossary, translate via LLMService (simulated here for a single term)
        if not info.get("translation"): # If glossary has no direct translation
             info["translation"] = llm_service.translate_text(term, SOURCE_LANG, TARGET_LANG)
        augmented_terms_info[term] = info
    
    print("\n--- Augmented Terms Information ---")
    for term, info in augmented_terms_info.items():
        print(f"  {term}: Context: '{info['context']}', Translation: '{info['translation']}'")

    # 4. Strategic Planning & Decomposition
    segmented_report = decomposer.segment_report(processed_report)
    structured_sections = decomposer.identify_key_sections(segmented_report)
    print(f"\n--- Report Segmented into {len(segmented_report)} sentences ---")
    # for i, segment in enumerate(segmented_report): print(f"  {i+1}. {segment}")

    # 5. Core Generative AI (Translation and Summarization) with Iterative Feedback
    translated_sections = {"Main Report": []}
    full_translated_text = []
    for section_name, segments in structured_sections.items():
        print(f"\nProcessing section: {section_name}")
        for i, segment in enumerate(segments):
            print(f"  Translating segment {i+1}: {segment[:50]}...")
            
            # Prepare augmented context for this segment
            segment_augmented_context = {}
            for term, info in augmented_terms_info.items():
                if term in segment.lower(): # Check if the term is relevant to the current segment
                    segment_augmented_context[term] = info['translation'] # Use the translated term for context
            
            # Initial Translation
            translated_segment = llm_service.translate_text(segment, SOURCE_LANG, TARGET_LANG, augmented_context=segment_augmented_context)
            print(f"    Initial Translation: {translated_segment[:70]}...")

            # Simulate Feedback and Refinement
            feedback = feedback_handler.get_feedback(segment, translated_segment, identified_terms)
            refined_translation = feedback_handler.refine_translation(segment, translated_segment, feedback)
            
            translated_sections[section_name].append(refined_translation)
            full_translated_text.append(refined_translation)
            print(f"    Refined Translation: {refined_translation[:70]}...")
            print(f"    Feedback Score: {feedback['quality_score']}")
            
    # Summarize the entire translated report
    overall_summary = llm_service.summarize_text(" ".join(full_translated_text))
    print("\n--- Overall Summary Generated ---")
    print(overall_summary)

    # 6. Generate Final Output
    final_report = output_generator.generate_final_report(translated_sections, overall_summary, augmented_terms_info)
    print("\n" + final_report)

