import difflib
from transformers import pipeline

class MultiAspectTranslator:
    def __init__(self):
        self.ner_pipeline = pipeline("ner", model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple")
        self.translator_pipeline = pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")

        self.exemplars = [
            ("Où est ma commande ?", "Where is my order?"),
            ("Je veux retourner cet article.", "I want to return this item."),
            ("Problème avec la livraison.", "Issue with delivery."),
            ("Comment puis-je contacter le service client ?", "How can I contact customer service?"),
            ("Le produit est défectueux.", "The product is defective."),
            ("Puis-je changer mon adresse de livraison ?", "Can I change my delivery address?"),
            ("Le paiement a échoué.", "Payment failed."),
            ("Quel est le statut de mon remboursement ?", "What is the status of my refund?"),
            ("La taille ne convient pas.", "The size does not fit."),
            ("J'ai reçu un article incorrect.", "I received an incorrect item."),
        ]

    def _knowledge_mining(self, text):
        ner_results = self.ner_pipeline(text)
        keywords = []
        for entity in ner_results:
            keywords.append(entity["word"])
        
        topics = []
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["commande", "livraison", "order", "delivery"]):
            topics.append("delivery_tracking")
        if any(kw in text_lower for kw in ["retourner", "remboursement", "return", "refund"]):
            topics.append("returns_refunds")
        if any(kw in text_lower for kw in ["produit", "article", "item", "product", "défectueux", "incorrect"]):
            topics.append("product_issue")
        if any(kw in text_lower for kw in ["paiement", "payement", "failed"]):
            topics.append("payment_issue")
        
        return {"keywords": list(set(keywords)), "topics": list(set(topics))}

    def _generate_exemplars(self, source_text):
        best_exemplar_match = None
        highest_similarity = -1
        
        for fr_exemplar, en_exemplar in self.exemplars:
            similarity = difflib.SequenceMatcher(None, source_text.lower(), fr_exemplar.lower()).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_exemplar_match = (fr_exemplar, en_exemplar)
        
        if highest_similarity > 0.6:
            return best_exemplar_match
        return None

    def _generate_multi_translations(self, source_text, knowledge):
        base_translation = self.translator_pipeline(source_text)[0]["translation_text"]
        
        translations = {base_translation}
        
        if knowledge["keywords"]:
            keyword_prompt = f"Translate this customer query to English, ensuring key terms like '{', '.join(knowledge['keywords'])}' are accurately reflected: {source_text}"
            try:
                keyword_translation = self.translator_pipeline(keyword_prompt)[0]["translation_text"]
                translations.add(keyword_translation)
            except Exception:
                pass
        
        if knowledge["topics"]:
            topic_prompt = f"Translate this customer support query about {', '.join(knowledge['topics'])} to English: {source_text}"
            try:
                topic_translation = self.translator_pipeline(topic_prompt)[0]["translation_text"]
                translations.add(topic_translation)
            except Exception:
                pass
        
        return list(translations)

    def _select_best_translation(self, source_text, knowledge, exemplars, candidate_translations):
        best_translation = None
        highest_score = -1

        source_keywords = set(k.lower() for k in knowledge["keywords"])
        
        for translation in candidate_translations:
            score = 0
            
            translated_text_lower = translation.lower()
            relevance_keywords = sum(1 for kw in source_keywords if kw in translated_text_lower)
            score += relevance_keywords * 2

            if exemplars:
                exemplar_en = exemplars[1]
                exemplar_similarity = difflib.SequenceMatcher(None, translated_text_lower, exemplar_en.lower()).ratio()
                score += exemplar_similarity * 5

            score += len(translation.split()) * 0.1
            
            if score > highest_score:
                highest_score = score
                best_translation = translation
        
        if best_translation is None and candidate_translations:
            return candidate_translations[0]
        
        return best_translation

    def translate_query(self, query):
        knowledge = self._knowledge_mining(query)
        exemplar_match = self._generate_exemplars(query)
        candidate_translations = self._generate_multi_translations(query, knowledge)
        final_translation = self._select_best_translation(query, knowledge, exemplar_match, candidate_translations)
        
        return final_translation