from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class TranslationService:
    def __init__(self):
        self.translators = {}  # Cache for loaded models and tokenizers

    def _load_model_and_tokenizer(self, model_name):
        """Loads a translation model and its tokenizer, caching it for future use."""
        if model_name not in self.translators:
            try:
                print(f"Attempting to load translation model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.translators[model_name] = {"tokenizer": tokenizer, "model": model}
                print(f"Successfully loaded translation model: {model_name}")
            except Exception as e:
                print(f"Error loading translation model {model_name}: {e}")
                self.translators[model_name] = None # Mark as failed
        return self.translators[model_name]

    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Internal method to perform translation between two specified languages."""
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        model_components = self._load_model_and_tokenizer(model_name)

        if not model_components:
            print(f"Translation model {model_name} not available. Returning original text.")
            return text # Return original text if model not found/loaded

        tokenizer = model_components["tokenizer"]
        model = model_components["model"]

        # Tokenize and translate
        # The `max_length` parameter ensures that long inputs are truncated, preventing errors.
        # `max_new_tokens` controls the length of the generated translation.
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=100, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

    def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translates text from a source language to English."""
        if source_lang.lower() == "en":
            return text
        return self._translate(text, source_lang.lower(), "en")

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translates text from English to a target language."""
        if target_lang.lower() == "en":
            return text
        return self._translate(text, "en", target_lang.lower())