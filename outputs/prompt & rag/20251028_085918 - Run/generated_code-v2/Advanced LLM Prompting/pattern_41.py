import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class MedicalTranslator:
    def __init__(self, target_lang="fr"):
        self.target_lang = target_lang
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.translation_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        self.translation_model.config.forced_bos_token_id = self.translation_tokenizer.lang_code_to_id[self.target_lang]

        self.medical_glossary = {
            "cardiac arrest": "arrêt cardiaque",
            "myocardial infarction": "infarctus du myocarde",
            "hypertension": "hypertension artérielle",
            "diabetes mellitus": "diabète sucré",
            "malignant tumor": "tumeur maligne",
            "chemotherapy": "chimiothérapie",
            "radiation therapy": "radiothérapie",
            "diagnosis": "diagnostic",
            "prognosis": "pronostic",
            "inflammation": "inflammation",
            "pathology": "pathologie"
        }

    def _knowledge_mining(self, text):
        doc = self.nlp(text)

        # Medical Entity Recognition (NER) - Simplified
        medical_terms = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "EVENT"]]
        # Augment with glossary terms if they appear in text
        medical_terms.extend([term for term in self.medical_glossary if term in text.lower()])
        medical_terms = list(set(medical_terms))

        # Contextual Topic Extraction & Key Phrase Extraction (Simplified)
        key_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
        key_sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > 5]

        return {"medical_terms": medical_terms, "key_phrases": key_phrases, "key_sentences": key_sentences}

    def _generate_exemplars(self, mined_knowledge):
        exemplars = []
        for term in mined_knowledge["medical_terms"]:
            if term in self.medical_glossary:
                exemplars.append(f"{term}: {self.medical_glossary[term]}")
        return exemplars

    def _generate_translation_candidates(self, source_text, mined_knowledge, exemplars, num_candidates=3):
        prompt_parts = [
            "Translate the following medical text. Consider the medical terms and context provided.",
            f"Source Text: {source_text}"
        ]
        if mined_knowledge["medical_terms"]:
            prompt_parts.append(f"Medical Terms: {', '.join(mined_knowledge['medical_terms'])}")
        if exemplars:
            prompt_parts.append(f"Translation Exemplars: {'; '.join(exemplars)}")
        if mined_knowledge["key_sentences"]:
            prompt_parts.append(f"Key Contextual Sentences: {' '.join(mined_knowledge['key_sentences'])}")

        prompt = "\n".join(prompt_parts)
        
        inputs = self.translation_tokenizer(prompt, return_tensors="pt")
        
        # Generate multiple candidates using beam search with diverse decoding
        translated_tokens = self.translation_model.generate(
            **inputs,
            forced_bos_token_id=self.translation_model.config.forced_bos_token_id,
            num_return_sequences=num_candidates,
            num_beams=num_candidates * 2,  # Use more beams to get diverse candidates
            diversity_penalty=1.0,  # Encourage diverse outputs
            max_length=512
        )
        
        candidates = [self.translation_tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        return candidates

    def _medical_accuracy_scorer(self, translation_candidate, mined_knowledge):
        score = 0
        if not mined_knowledge["medical_terms"]:
            return 1.0  # If no specific terms, assume accurate

        translated_terms_found = 0
        for original_term, translated_term in self.medical_glossary.items():
            if original_term in mined_knowledge["medical_terms"] and translated_term.lower() in translation_candidate.lower():
                translated_terms_found += 1
        
        if len(mined_knowledge["medical_terms"]) > 0:
            score = translated_terms_found / len(mined_knowledge["medical_terms"])
        return score

    def _contextual_fit_scorer(self, source_text, translation_candidate):
        source_embedding = self.sbert_model.encode([source_text])
        candidate_embedding = self.sbert_model.encode([translation_candidate])
        similarity = cosine_similarity(source_embedding, candidate_embedding)[0][0]
        return similarity

    def _linguistic_quality_scorer(self, translation_candidate):
        # Simplified: Check for basic grammar and fluency. A real system would use a more robust model.
        # For demonstration, we just check for sentence structure and minimum length.
        doc = self.nlp(translation_candidate) # Use English nlp for a basic check, ideally a target language nlp
        num_sentences = len(list(doc.sents))
        num_tokens = len(doc)
        
        if num_sentences > 0 and num_tokens > 5:
            return 1.0 # Basic pass
        return 0.5 # Subpar if very short or no sentences

    def translate_document(self, source_text):
        # 1. Knowledge Mining
        mined_knowledge = self._knowledge_mining(source_text)

        # 2. Translation Exemplar Generation
        exemplars = self._generate_exemplars(mined_knowledge)

        # 3. Multiple Translation Candidates Generation
        candidates = self._generate_translation_candidates(source_text, mined_knowledge, exemplars)

        # 4. Translation Selection and Refinement
        best_translation = ""
        highest_score = -1

        for candidate in candidates:
            medical_accuracy_score = self._medical_accuracy_scorer(candidate, mined_knowledge)
            contextual_fit_score = self._contextual_fit_scorer(source_text, candidate)
            linguistic_quality_score = self._linguistic_quality_scorer(candidate)

            # Weighted scoring
            # Adjust weights based on importance
            total_score = (
                0.5 * medical_accuracy_score +
                0.3 * contextual_fit_score +
                0.2 * linguistic_quality_score
            )

            if total_score > highest_score:
                highest_score = total_score
                best_translation = candidate

        return best_translation


if __name__ == "__main__":
    translator = MedicalTranslator(target_lang="fr")

    medical_text_en = (
        "The patient presented with symptoms indicative of myocardial infarction. "
        "Immediate treatment for cardiac arrest was initiated, including chemotherapy regimen. "
        "The diagnosis confirmed a malignant tumor requiring radiation therapy. "
        "Monitoring for hypertension is crucial for the prognosis." 
        "A new drug for diabetes mellitus is under clinical trial." 
    )

    print("\n--- Source Medical Document (English) ---")
    print(medical_text_en)

    translated_text = translator.translate_document(medical_text_en)

    print("\n--- Translated Medical Document (French) ---")
    print(translated_text)

    print("\n--- Testing with a simpler text ---")
    simple_text_en = "The patient has a fever and requires medication."
    translated_simple_text = translator.translate_document(simple_text_en)
    print(f"Source: {simple_text_en}")
    print(f"Translated: {translated_simple_text}")

    print("\n--- Testing with a different target language (Spanish) ---")
    translator_es = MedicalTranslator(target_lang="es")
    medical_text_en_es = (
        "The patient presented with symptoms indicative of myocardial infarction. "
        "Immediate treatment for cardiac arrest was initiated. "
        "The diagnosis confirmed a malignant tumor." 
    )
    translated_text_es = translator_es.translate_document(medical_text_en_es)
    print(f"Source (ES): {medical_text_en_es}")
    print(f"Translated (ES): {translated_text_es}")
