
import random
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

class MultilingualCustomerSupportBot:
    def __init__(self, 
                 llm_model_name="distilgpt2", 
                 embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        
        print(f"Initializing LLM with {llm_model_name}...")
        self.llm_pipeline = pipeline("text-generation", model=llm_model_name)
        print("LLM initialized.")

        print(f"Initializing Embedding model with {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("Embedding model initialized.")

        # Mock Knowledge Base (English source language)
        self.knowledge_base_en = [
            {"id": 1, "question_en": "What is your return policy?", "answer_en": "Our return policy allows returns within 30 days of purchase with a receipt."}, 
            {"id": 2, "question_en": "How can I track my order?", "answer_en": "You can track your order using the tracking number provided in your shipping confirmation email on our website."}, 
            {"id": 3, "question_en": "Do you ship internationally?", "answer_en": "Yes, we ship to over 100 countries worldwide. Shipping costs and times vary by destination."}, 
            {"id": 4, "question_en": "How do I change my shipping address?", "answer_en": "Please contact customer support immediately to change your shipping address if your order has not yet been shipped."}, 
            {"id": 5, "question_en": "What payment methods do you accept?", "answer_en": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay."}, 
            {"id": 6, "question_en": "My item arrived damaged, what should I do?", "answer_en": "Please send us photos of the damaged item and packaging within 48 hours of delivery for a replacement or refund."}, 
            {"id": 7, "question_en": "Can I cancel an order after it has been placed?", "answer_en": "Orders can be canceled only if they have not yet been processed for shipping. Please contact us as soon as possible."} 
        ]
        
        print("Pre-computing knowledge base embeddings...")
        self.kb_embeddings = self.embedding_model.encode(
            [item["question_en"] for item in self.knowledge_base_en],
            convert_to_tensor=True
        )
        print("Knowledge base embeddings computed.")

    def _mock_translate(self, text, target_lang="es"):
        """A mock translation function. In a real application, this would use a dedicated translation API 
        or a more robust translation model (e.g., mBART, NLLB via transformers)."""
        if target_lang == "es":
            # Simple placeholder for Spanish translation
            translation_map = {
                "What is your return policy?": "¿Cuál es su política de devoluciones?",
                "Our return policy allows returns within 30 days of purchase with a receipt.": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra con un recibo.",
                "How can I track my order?": "¿Cómo puedo rastrear mi pedido?",
                "You can track your order using the tracking number provided in your shipping confirmation email on our website.": "Puede rastrear su pedido utilizando el número de seguimiento proporcionado en su correo electrónico de confirmación de envío en nuestro sitio web.",
                "Do you ship internationally?": "¿Realizan envíos internacionales?",
                "Yes, we ship to over 100 countries worldwide. Shipping costs and times vary by destination.": "Sí, enviamos a más de 100 países en todo el mundo. Los costos y tiempos de envío varían según el destino.",
                "How do I change my shipping address?": "¿Cómo cambio mi dirección de envío?",
                "Please contact customer support immediately to change your shipping address if your order has not yet been shipped.": "Póngase en contacto con el servicio de atención al cliente inmediatamente para cambiar su dirección de envío si su pedido aún no ha sido enviado.",
                "What payment methods do you accept?": "¿Qué métodos de pago aceptan?",
                "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.": "Aceptamos Visa, Mastercard, American Express, PayPal y Apple Pay.",
                "My item arrived damaged, what should I do?": "Mi artículo llegó dañado, ¿qué debo hacer?",
                "Please send us photos of the damaged item and packaging within 48 hours of delivery for a replacement or refund.": "Por favor, envíenos fotos del artículo dañado y el embalaje dentro de las 48 horas posteriores a la entrega para un reemplazo o reembolso.",
                "Can I cancel an order after it has been placed?": "¿Puedo cancelar un pedido después de haberlo realizado?",
                "Orders can be canceled only if they have not yet been processed for shipping. Please contact us as soon as possible.": "Los pedidos solo se pueden cancelar si aún no han sido procesados para el envío. Póngase en contacto con nosotros lo antes posible."
            }
            return translation_map.get(text, f"[TRANSLATED_ES] {text}")
        # Add more languages if needed
        return f"[TRANSLATED_{target_lang.upper()}] {text}"

    def _retrieve_and_translate_examples(self, user_query_embedding, top_k=3, target_lang="es"):
        """Retrieves relevant examples from the KB and applies InCLT by translating them.
        Prioritizes an existing target language example if available, otherwise translates.
        """
        # Compute cosine similarities
        cos_scores = util.cos_sim(user_query_embedding, self.kb_embeddings)[0]
        top_results = sorted(zip(cos_scores, range(len(cos_scores))), key=lambda x: x[0], reverse=True)

        icl_examples = []
        for score, idx in top_results[0:top_k]:
            kb_item = self.knowledge_base_en[idx]

            # InCLT Logic: Leverage both source (English) and target language for examples
            english_question = kb_item["question_en"]
            english_answer = kb_item["answer_en"]
            
            # Translate to target language for InCLT
            translated_question = self._mock_translate(english_question, target_lang)
            translated_answer = self._mock_translate(english_answer, target_lang)

            example_str = (
                f"Example:\n"
                f"English Query: {english_question}\n"
                f"English Answer: {english_answer}\n"
                f"Translated Query ({target_lang}): {translated_question}\n"
                f"Translated Answer ({target_lang}): {translated_answer}\n"
            )
            icl_examples.append(example_str)
            
        return "\n".join(icl_examples)

    def get_bot_response(self, user_query, target_lang="es", max_new_tokens=150):
        """Generates a response to the user query using InCLT prompting."""
        print(f"\nUser ({target_lang}): {user_query}")
        user_query_embedding = self.embedding_model.encode(user_query, convert_to_tensor=True)
        
        icl_prompt_examples = self._retrieve_and_translate_examples(user_query_embedding, target_lang=target_lang)

        # Combine InCLT examples with the user's query for the final prompt
        full_prompt = (
            f"You are an e-commerce customer support bot. Provide helpful and concise answers based on the examples provided.\n\n"
            f"{icl_prompt_examples}\n"
            f"Customer Query ({target_lang}): {user_query}\n"
            f"Bot Answer ({target_lang}):"
        )
        
        print(f"\n--- Constructed Prompt ---\n{full_prompt}\n--------------------------")

        # Generate response using the LLM
        # For distilgpt2, we need to limit the input length as it's a small model.
        # In a real scenario, a larger multilingual LLM would handle longer contexts better.
        generated_text = self.llm_pipeline(
            full_prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            truncation=True # Enable truncation for models with context length limits
        )[0]["generated_text"]
        
        # Extract only the bot's answer part, as distilgpt2 might just continue the prompt.
        response_start_tag = f"Bot Answer ({target_lang}):"
        if response_start_tag in generated_text:
            bot_answer = generated_text.split(response_start_tag, 1)[1].strip()
            # Sometimes LLMs echo the query, try to clean that up
            if user_query.lower() in bot_answer.lower():
                # Simple heuristic: if query is echoed, take content after that or just a snippet
                clean_answer = bot_answer.split(user_query, 1)[-1].strip()
                if clean_answer and len(clean_answer) < len(bot_answer): # Ensure we actually cut something meaningful
                    return clean_answer
            return bot_answer
        
        return generated_text.replace(full_prompt, "").strip() # Fallback if tag not found or model just continues


if __name__ == "__main__":
    # Using a smaller LLM like 'distilgpt2' for demonstration due to resource constraints 
    # and faster initialization. For production, consider larger multilingual LLMs 
    # such as 'facebook/mbart-large-50' or 'Helsinki-NLP/opus-mt-en-es' for translation 
    # and a powerful multilingual LLM for generation.
    bot = MultilingualCustomerSupportBot(llm_model_name="distilgpt2")

    # Example interaction in Spanish
    print("\n--- Starting Multilingual Customer Support Bot (Spanish) ---")
    query1 = "Quiero devolver un artículo, ¿cuánto tiempo tengo?"
    response1 = bot.get_bot_response(query1, target_lang="es")
    print(f"Bot: {response1}")

    query2 = "¿Cómo puedo ver dónde está mi compra?"
    response2 = bot.get_bot_response(query2, target_lang="es")
    print(f"Bot: {response2}")

    query3 = "Mi paquete llegó roto, ¿qué hago?"
    response3 = bot.get_bot_response(query3, target_lang="es")
    print(f"Bot: {response3}")

    query4 = "¿Qué formas de pago puedo usar?"
    response4 = bot.get_bot_response(query4, target_lang="es")
    print(f"Bot: {response4}")

    print("\n--- Demonstration End ---\n")
