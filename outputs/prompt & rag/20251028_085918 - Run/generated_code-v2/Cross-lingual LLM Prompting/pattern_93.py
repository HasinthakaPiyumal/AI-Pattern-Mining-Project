from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch
import re

class TranslationService:
    def __init__(self, source_lang_code="en", target_lang_code="es"):
        self.source_lang_code = source_lang_code
        self.target_lang_code = target_lang_code

        self.source_to_target_tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{source_lang_code}-{target_lang_code}")
        self.source_to_target_model = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{source_lang_code}-{target_lang_code}")

        self.target_to_source_tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{target_lang_code}-{source_lang_code}")
        self.target_to_source_model = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{target_lang_code}-{source_lang_code}")

    def translate(self, text, source_lang, target_lang):
        if source_lang == self.source_lang_code and target_lang == self.target_lang_code:
            tokenized = self.source_to_target_tokenizer(text, return_tensors="pt")
            translated = self.source_to_target_model.generate(**tokenized)
            return self.source_to_target_tokenizer.decode(translated[0], skip_special_tokens=True)
        elif source_lang == self.target_lang_code and target_lang == self.source_lang_code:
            tokenized = self.target_to_source_tokenizer(text, return_tensors="pt")
            translated = self.target_to_source_model.generate(**tokenized)
            return self.target_to_source_tokenizer.decode(translated[0], skip_special_tokens=True)
        else:
            return f"Translation not supported for {source_lang} to {target_lang} in this demo's initialized models."

class FAQKnowledgeBase:
    def __init__(self):
        self.faqs = {
            "What is your return policy?": "You can return items within 30 days of purchase with the original receipt.",
            "How do I track my order?": "You can track your order using the tracking number provided in your shipping confirmation email.",
            "Do you offer international shipping?": "Yes, we offer international shipping to most countries. Shipping fees and delivery times vary by destination.",
            "How can I contact customer support?": "You can contact our customer support via email at support@example.com or by calling +1-800-123-4567."
        }
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.faq_questions = list(self.faqs.keys())
        self.faq_embeddings = self.embedder.encode(self.faq_questions, convert_to_tensor=True)

    def retrieve_faq(self, query_embedding, top_k=1):
        cosine_scores = util.cos_sim(query_embedding, self.faq_embeddings)[0]
        top_results = torch.topk(cosine_scores, k=top_k)
        retrieved_q_idx = top_results.indices[0].item()
        retrieved_q = self.faq_questions[retrieved_q_idx]
        retrieved_a = self.faqs[retrieved_q]
        return retrieved_q, retrieved_a

class PromptGenerator:
    def __init__(self):
        pass

    def generate_prompt(self, user_query, target_lang, retrieved_english_faq_q, retrieved_english_faq_a, translator):
        translated_faq_q = translator.translate(retrieved_english_faq_q, "en", target_lang)
        translated_faq_a = translator.translate(retrieved_english_faq_a, "en", target_lang)

        prompt_template = f"""
        You are a helpful multilingual customer support assistant for an e-commerce company.
        Your goal is to answer customer questions accurately and helpfully.

        Here is some background information and examples to help you:

        ---
        In-Context Learning Example (English):
        Question: {retrieved_english_faq_q}
        Answer: {retrieved_english_faq_a}

        In-Context Learning Example ({target_lang.upper()}):
        Question: {translated_faq_q}
        Answer: {translated_faq_a}
        ---

        Customer's Question ({target_lang.upper()}):
        {user_query}

        Please provide a concise answer in {target_lang}.
        Answer:
        """
        return prompt_template

class MultilingualLLM:
    def __init__(self):
        pass

    def simulate_response(self, prompt, target_lang):
        target_lang_upper = target_lang.upper()
        pattern = rf"In-Context Learning Example \({target_lang_upper}\):\nQuestion:.*\nAnswer: (.*)\n---"
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        if target_lang == "es":
            return "Lo siento, no pude encontrar una respuesta específica a su pregunta. ¿Podría reformularla?"
        elif target_lang == "fr":
            return "Je suis désolé, je n'ai pas pu trouver de réponse spécifique à votre question. Pourriez-vous la reformuler ?"
        else:
            return "I apologize, I couldn't find a specific answer to your question. Could you please rephrase it."

class Chatbot:
    def __init__(self, target_lang="es"):
        self.target_lang = target_lang
        self.translation_service = TranslationService(target_lang_code=target_lang)
        self.faq_kb = FAQKnowledgeBase()
        self.prompt_generator = PromptGenerator()
        self.llm = MultilingualLLM()

    def get_response(self, user_query):
        translated_query_en = self.translation_service.translate(user_query, self.target_lang, "en")

        query_embedding = self.faq_kb.embedder.encode(translated_query_en, convert_to_tensor=True)
        retrieved_english_faq_q, retrieved_english_faq_a = self.faq_kb.retrieve_faq(query_embedding)

        prompt = self.prompt_generator.generate_prompt(
            user_query,
            self.target_lang,
            retrieved_english_faq_q,
            retrieved_english_faq_a,
            self.translation_service
        )

        llm_response = self.llm.simulate_response(prompt, self.target_lang)

        return llm_response
