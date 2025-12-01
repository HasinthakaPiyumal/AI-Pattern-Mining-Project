from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

DetectorFactory.seed = 0 # Ensure consistent language detection results

class MultilingualChatbot:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-es"): # Using a smaller, demonstrative translation model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.in_context_examples = [
            {"source_text": "How can I reset my password?", "source_lang": "en", "target_text": "¿Cómo puedo restablecer mi contraseña?", "target_lang": "es"},
            {"source_text": "Mi internet no funciona.", "source_lang": "es", "target_text": "My internet is not working.", "target_lang": "en"},
            {"source_text": "I need help with my account settings.", "source_lang": "en", "target_text": "Necesito ayuda con la configuración de mi cuenta.", "target_lang": "es"}
        ]
        # Map for target languages. For simplicity, we'll try to translate to English if source is Spanish, else Spanish if source is English.
        # In a real scenario, this would be more complex, e.g., user preference or agent language.
        self.default_target_lang_map = {"en": "es", "es": "en"}

    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return "unknown"

    def generate_inclt_prompt(self, query, source_lang, target_lang, examples):
        prompt_parts = []
        for ex in examples:
            # Include examples where source or target language matches the current interaction's source or target
            if ex["source_lang"] == source_lang or ex["target_lang"] == target_lang:
                prompt_parts.append(f"Source ({ex['source_lang']}): {ex['source_text']}")
                prompt_parts.append(f"Target ({ex['target_lang']}): {ex['target_text']}")
        
        # Add the current query at the end
        prompt_parts.append(f"Source ({source_lang}): {query}")
        prompt_parts.append(f"Target ({target_lang}):") # The LLM should complete this
        
        # The prompt for a seq2seq model like mBART or NMT models needs to be structured for translation
        # For simpler NMT models, it's often just 'translate X from Y to Z'
        # Given opus-mt, it expects a prefix like >>src_lang<< followed by the text
        # We'll adapt the prompt to fit a direct translation flow within the ICL concept.
        
        # For an LLM that directly takes ICL examples to *answer* in a target language (not just translate),
        # the prompt structure would be different. Here, we're demonstrating cross-lingual *transfer* in the context of translation.
        
        # Let's simplify for `Helsinki-NLP/opus-mt-en-es` which is a direct translation model.
        # The InCLT concept here means we're implicitly showing the model *how* to translate specific phrases
        # by providing examples. For a true LLM with reasoning, the examples would guide the *style* or *domain* of answers.

        # For opus-mt, the prompt will actually be just the query with a target language prefix.
        # The ICL examples would be used if we were fine-tuning or prompting a larger, more general LLM.
        # To demonstrate the *spirit* of InCLT for boosting cross-lingual capabilities with a simple model:
        # We'll use the examples to potentially influence the *interpretation* of the query, 
        # although the opus-mt model primarily performs direct translation.
        # A more advanced LLM would use the examples to guide generation beyond just literal translation.
        
        # For this specific `opus-mt` model, direct translation is its primary function.
        # The ICL 