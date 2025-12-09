import re
from transformers import pipeline

class MedicalTranslator:
    def __init__(self, translation_model_name="Helsinki-NLP/opus-mt-en-es"):
        self.translator = pipeline("translation", model=translation_model_name)
        self.medical_exemplars = {
            "hypertension": {
                "patient has hypertension": "el paciente tiene hipertensión",
                "blood pressure medication": "medicamento para la presión arterial"
            },
            "diabetes": {
                "type 2 diabetes": "diabetes tipo 2",
                "insulin dosage": "dosis de insulina"
            },
            "cardiac arrest": {
                "suffered a cardiac arrest": "sufrió un paro cardíaco",
                "cardiopulmonary resuscitation": "reanimación cardiopulmonar"
            }
        }
        self.medical_terms_glossary = list(self.medical_exemplars.keys()) + [
            "diagnosis", "treatment", "symptom", "prognosis", "medication", "surgery",
            "inflammation", "infection", "vascular", "neurological", "oncology"
        ]

    def _knowledge_mining(self, text):
        extracted_keywords = []
        text_lower = text.lower()
        for term in self.medical_terms_glossary:
            if term in text_lower:
                extracted_keywords.append(term)
        return list(set(extracted_keywords))

    def _generate_exemplars(self, keywords):
        relevant_exemplars = {}
        for keyword in keywords:
            if keyword in self.medical_exemplars:
                relevant_exemplars.update(self.medical_exemplars[keyword])
        return relevant_exemplars

    def _generate_multiple_translations(self, text, knowledge_context):
        translations = []
        base_translation = self.translator(text, max_length=150)[0]['translation_text']
        translations.append(base_translation)

        if knowledge_context:
            context_prompt = f"Translate the following medical text, ensuring accuracy for terms like {', '.join(knowledge_context)}. Text: '{text}'"
            contextual_translation = self.translator(context_prompt, max_length=150)[0]['translation_text']
            translations.append(contextual_translation)

        if len(translations) == 1:
            translations.append(self.translator(text, max_length=150, num_beams=2, early_stopping=True)[0]['translation_text'])
        
        return list(set(translations))

    def _select_best_translation(self, translations, source_text, knowledge_context):
        if not translations:
            return ""

        best_translation = translations[0]
        max_score = -1

        for translation in translations:
            score = 0
            for term in knowledge_context:
                if term in translation.lower():
                    score += 5
            
            score += len(translation.split())

            for original_phrase, translated_phrase in self._generate_exemplars(knowledge_context).items():
                if translated_phrase.lower() in translation.lower():
                    score += 10

            if score > max_score:
                max_score = score
                best_translation = translation
        
        return best_translation

    def translate(self, text):
        keywords = self._knowledge_mining(text)
        print(f"Extracted Keywords: {keywords}")

        possible_translations = self._generate_multiple_translations(text, keywords)
        print(f"Possible Translations: {possible_translations}")

        final_translation = self._select_best_translation(possible_translations, text, keywords)
        print(f"Final Selected Translation: {final_translation}")
        
        return final_translation