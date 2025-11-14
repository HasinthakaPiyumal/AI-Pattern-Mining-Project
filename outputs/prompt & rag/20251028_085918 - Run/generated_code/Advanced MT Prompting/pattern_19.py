
from transformers import pipeline
import re

class MedicalTranslator:
    def __init__(self):
        # Initialize translation pipelines for various languages.
        # In a real application, you would load specific models for each language pair
        # or use a powerful multilingual model. Here, we'll simulate with English-German
        # and German-English as examples, and a mock for others.
        self.translators = {
            "en-de": pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de") if False else self._mock_translator,
            "de-en": pipeline("translation_de_to_en", model="Helsinki-NLP/opus-mt-de-en") if False else self._mock_translator,
            # Add more language pairs as needed, e.g., en-es, es-en, en-fr, fr-en, etc.
        }
        self.medical_glossary = {
            "en": {
                "fever": "Fieber", "headache": "Kopfschmerzen", "symptom": "Symptom",
                "diagnosis": "Diagnose", "medication": "Medikament", "prescription": "Rezept",
                "blood pressure": "Blutdruck", "diabetes": "Diabetes", "vaccine": "Impfstoff"
            },
            "de": {
                "Fieber": "fever", "Kopfschmerzen": "headache", "Symptom": "symptom",
                "Diagnose": "diagnosis", "Medikament": "medication", "Rezept": "prescription",
                "Blutdruck": "blood pressure", "Diabetes": "diabetes", "Impfstoff": "vaccine"
            }
            # Add glossaries for other languages
        }

    def _mock_translator(self, text):
        # A mock translator for demonstration purposes when actual models are not loaded.
        # It simply capitalizes and adds a prefix to simulate translation.
        print(f"[MOCK TRANSLATION] Translating: '{text}'")
        return [{
            "translation_text": f"[Translated] {text.capitalize()}"
        }]

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translates text from source_lang to target_lang using a suitable model."""
        if source_lang == target_lang:
            return text

        lang_pair = f"{source_lang}-{target_lang}"
        if lang_pair in self.translators:
            # Use the actual or mock pipeline
            translated_output = self.translators[lang_pair](text)
            return translated_output[0]["translation_text"]
        else:
            print(f"Warning: No direct translator for {lang_pair}. Using mock translation.")
            return self._mock_translator(text)[0]["translation_text"]

    def lookup_medical_terms(self, text: str, lang: str) -> str:
        """Highlights medical terms in the text and potentially provides their definitions/translations.
           Implements 'ChainofDictionary' pattern.
        """
        if lang not in self.medical_glossary:
            return text # No glossary for this language

        highlighted_text = text
        for term, translation in self.medical_glossary[lang].items():
            # Use regex for whole word matching to avoid partial matches
            pattern = r'\b' + re.escape(term) + r'\b'
            # Replace with a highlighted version, e.g., adding bold markdown
            highlighted_text = re.sub(pattern, f"**{term}** (translates to '{translation}')", highlighted_text, flags=re.IGNORECASE)
        return highlighted_text

    def translate_first_prompting(self, patient_input: str, source_lang: str, target_lang: str) -> str:
        """Applies 'Translate First Prompting' pattern by translating the patient's input first.
           Also incorporates dictionary lookup after initial translation.
        """
        print(f"Applying 'Translate First Prompting' for {source_lang} to {target_lang}...")
        initial_translation = self.translate_text(patient_input, source_lang, target_lang)
        print(f"Initial Translation: {initial_translation}")
        # Augment with dictionary lookup for accuracy after initial translation
        augmented_translation = self.lookup_medical_terms(initial_translation, target_lang)
        print(f"Augmented Translation (with dictionary): {augmented_translation}")
        return augmented_translation

# Example Usage (for testing this module)
if __name__ == "__main__":
    translator = MedicalTranslator()

    patient_text_de = "Ich habe Fieber und Kopfschmerzen."
    patient_text_en = "I have a fever and headache."
    patient_text_unknown = "Mi siento mal."

    # Translate First Prompting DE -> EN
    translated_prompt_en = translator.translate_first_prompting(patient_text_de, "de", "en")
    print(f"\nTranslated DE input for LLM: {translated_prompt_en}")

    # Translate First Prompting EN -> DE
    translated_prompt_de = translator.translate_first_prompting(patient_text_en, "en", "de")
    print(f"\nTranslated EN input for LLM: {translated_prompt_de}")

    # Test with an unsupported language pair (will use mock)
    translated_prompt_mock = translator.translate_first_prompting(patient_text_unknown, "es", "en")
    print(f"\nTranslated ES input for LLM (mock): {translated_prompt_mock}")

    # Test dictionary lookup directly
    highlighted_en = translator.lookup_medical_terms("The patient has a fever and high blood pressure.", "en")
    print(f"\nHighlighted English: {highlighted_en}")

    highlighted_de = translator.lookup_medical_terms("Der Patient hat Fieber und Diabetes.", "de")
    print(f"\nHighlighted German: {highlighted_de}")
