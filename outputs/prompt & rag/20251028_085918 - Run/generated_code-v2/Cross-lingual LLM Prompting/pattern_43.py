
class MultilingualChatbot:
    def __init__(self):
        # Mock Knowledge Base - simplified for demonstration
        # In a real application, this would be a database or a more complex RAG system.
        self.knowledge_base = {
            "shipping": {
                "en": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days. Free shipping for orders over $50.",
                "es": "El envío estándar tarda de 5 a 7 días hábiles. El envío exprés tarda de 2 a 3 días hábiles. Envío gratuito para pedidos superiores a $50.",
                "fr": "L'expédition standard prend 5 à 7 jours ouvrables. L'expédition express prend 2 à 3 jours ouvrables. Livraison gratuite pour les commandes de plus de 50 $."
            },
            "return_policy": {
                "en": "You can return items within 30 days of purchase with the original receipt. Refunds are processed within 7 business days.",
                "es": "Puede devolver artículos dentro de los 30 días posteriores a la compra con el recibo original. Los reembolsos se procesan dentro de los 7 días hábiles.",
                "fr": "Vous pouvez retourner des articles dans les 30 jours suivant l'achat avec le reçu original. Les remboursements sont traités dans les 7 jours ouvrables."
            },
            "product_warranty": {
                "en": "All electronics come with a 1-year manufacturer's warranty.",
                "es": "Todos los productos electrónicos vienen con una garantía de fabricante de 1 año.",
                "fr": "Tous les produits électroniques sont livrés avec une garantie du fabricant d'un an."
            },
            "payment_methods": {
                "en": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
                "es": "Aceptamos Visa, Mastercard, American Express, PayPal y Apple Pay.",
                "fr": "Nous acceptons Visa, Mastercard, American Express, PayPal et Apple Pay."
            }
        }
        # Supported languages for this demo
        self.supported_languages = ['en', 'es', 'fr']
        self.source_language = 'en' # English as the primary source for KB and initial examples

    def _detect_language(self, text: str) -> str:
        # Simplified language detection for demo purposes
        # In a real application, use a robust library like 'langdetect' or 'fasttext'
        text_lower = text.lower()
        if "envío" in text_lower or "devoluciones" in text_lower or "garantía" in text_lower or "pago" in text_lower or "hola" in text_lower:
            return 'es'
        if "expédition" in text_lower or "retours" in text_lower or "garantie" in text_lower or "paiement" in text_lower or "bonjour" in text_lower:
            return 'fr'
        return 'en' # Default to English

    def _translate(self, text: str, target_lang: str) -> str:
        # Mock translation function
        # In a real application, integrate with a translation API (e.g., Google Translate, DeepL)
        print(f"[MOCK TRANSLATION] Translating '{text[:30]}...' to {target_lang}")
        # For this demo, we assume pre-translated knowledge base snippets are available.
        # If translating arbitrary text, you'd call an external API here.
        return text # Return original if no specific translation logic for arbitrary text

    def _get_knowledge_base_info(self, query: str, target_lang: str) -> dict:
        # Simple keyword-based retrieval from knowledge base
        query_lower = query.lower()
        relevant_info = {}
        for key, translations in self.knowledge_base.items():
            if key in query_lower or any(word in query_lower for word in translations[target_lang].lower().split()):
                relevant_info[key] = translations
        return relevant_info

    def _construct_inclt_prompt(self, customer_query: str, target_lang: str, relevant_kb_info: dict) -> str:
        prompt_parts = [
            "You are a helpful customer support chatbot for an e-commerce platform. Your task is to answer customer questions accurately and concisely, using the provided information. If you don't know the answer, state that you don't have enough information.",
            "\nHere are some examples of how to answer questions, presented in both English (source) and the target language, to help you understand cross-lingual contexts:",
            ""
        ]

        # --- In-Context Examples (Source + Target Languages) ---
        # Example 1: Shipping Information
        prompt_parts.append(f"English Question: How long does standard shipping take?\nEnglish Answer: Standard shipping usually takes 5-7 business days.")
        if target_lang != self.source_language:
            prompt_parts.append(f"\n{target_lang.capitalize()} Question: ¿Cuánto tiempo tarda el envío estándar? (es) / Combien de temps prend l'expédition standard ? (fr)\n{target_lang.capitalize()} Answer: El envío estándar suele tardar de 5 a 7 días hábiles. (es) / L'expédition standard prend généralement 5 à 7 jours ouvrables. (fr)")
        prompt_parts.append("\n")

        # Example 2: Return Policy
        prompt_parts.append(f"English Question: What is your return policy?\nEnglish Answer: Our return policy allows returns within 30 days of purchase with a receipt.")
        if target_lang != self.source_language:
            prompt_parts.append(f"\n{target_lang.capitalize()} Question: ¿Cuál es su política de devoluciones? (es) / Quelle est votre politique de retour ? (fr)\n{target_lang.capitalize()} Answer: Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra con un recibo. (es) / Notre politique de retour permet les retours dans les 30 jours suivant l'achat avec un reçu. (fr)")
        prompt_parts.append("\n")

        # --- Relevant Knowledge Base Information (for current query) ---
        if relevant_kb_info:
            prompt_parts.append("Here is additional relevant information from our knowledge base:")
            for key, translations in relevant_kb_info.items():
                # Provide KB info in both source and target languages for maximum transfer
                if self.source_language in translations:
                    prompt_parts.append(f"- {key.replace('_', ' ').capitalize()} (English): {translations[self.source_language]}")
                if target_lang in translations and target_lang != self.source_language:
                    prompt_parts.append(f"- {key.replace('_', ' ').capitalize()} ({target_lang.capitalize()}): {translations[target_lang]}")
            prompt_parts.append("\n")

        # --- Customer's Actual Query ---
        prompt_parts.append(f"Now, please answer the following customer question in {target_lang.capitalize()}:\nCustomer: {customer_query}\nChatbot:")

        return "\n".join(prompt_parts)

    def _call_llm(self, prompt: str) -> str:
        # Mock LLM call
        # In a real application, integrate with an actual LLM API (e.g., OpenAI, Google Gemini, Hugging Face Transformers)
        print("[MOCK LLM] Simulating LLM response...")
        # For demonstration, we'll try to extract an answer based on keywords in the prompt
        # This is a very simplistic simulation.
        response = "I'm sorry, I don't have enough information to answer that question accurately." # Default response

        if "shipping" in prompt.lower() and "standard shipping takes 5-7 business days" in prompt.lower():
            if "es" in prompt.lower() and "envío estándar" in prompt.lower():
                response = "El envío estándar tarda de 5 a 7 días hábiles. Si su pedido supera los $50, el envío es gratuito."
            elif "fr" in prompt.lower() and "expédition standard" in prompt.lower():
                response = "L'expédition standard prend 5 à 7 jours ouvrables. Si votre commande dépasse 50 $, la livraison est gratuite."
            else:
                response = "Standard shipping takes 5-7 business days. If your order is over $50, shipping is free."
        elif "return_policy" in prompt.lower() and "return items within 30 days of purchase" in prompt.lower():
            if "es" in prompt.lower() and "devolver artículos dentro de los 30 días" in prompt.lower():
                response = "Puede devolver artículos dentro de los 30 días posteriores a la compra con el recibo original. Los reembolsos se procesan en 7 días hábiles."
            elif "fr" in prompt.lower() and "retourner des articles dans les 30 jours" in prompt.lower():
                response = "Vous pouvez retourner des articles dans les 30 jours suivant l'achat avec le reçu original. Les remboursements sont traités en 7 jours ouvrables."
            else:
                response = "You can return items within 30 days of purchase with the original receipt. Refunds are processed within 7 business days."
        elif "product_warranty" in prompt.lower() and "1-year manufacturer's warranty" in prompt.lower():
            if "es" in prompt.lower() and "garantía de fabricante de 1 año" in prompt.lower():
                response = "Todos los productos electrónicos vienen con una garantía de fabricante de 1 año."
            elif "fr" in prompt.lower() and "garantie du fabricant d'un an" in prompt.lower():
                response = "Tous les produits électroniques sont livrés avec une garantie du fabricant d'un an."
            else:
                response = "All electronics come with a 1-year manufacturer's warranty."
        elif "payment_methods" in prompt.lower() and "visa, mastercard, american express" in prompt.lower():
            if "es" in prompt.lower() and "aceptamos visa, mastercard" in prompt.lower():
                response = "Aceptamos Visa, Mastercard, American Express, PayPal y Apple Pay."
            elif "fr" in prompt.lower() and "acceptons visa, mastercard" in prompt.lower():
                response = "Nous acceptons Visa, Mastercard, American Express, PayPal et Apple Pay."
            else:
                response = "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay."

        # Fallback to general answer if no specific match, trying to be helpful
        if response == "I'm sorry, I don't have enough information to answer that question accurately.":
            if "es" in prompt.lower():
                response = "Lo siento, no tengo suficiente información para responder a esa pregunta con precisión. ¿Hay algo más en lo que pueda ayudarle?"
            elif "fr" in prompt.lower():
                response = "Je suis désolé, je n'ai pas assez d'informations pour répondre précisément à cette question. Puis-je vous aider avec autre chose ?"
            else:
                response = "I'm sorry, I don't have enough information to answer that question accurately. Is there anything else I can help you with?"

        return response


    def process_query(self, customer_query: str) -> str:
        print(f"\nCustomer Query: '{customer_query}'")

        # 1. Detect language
        target_lang = self._detect_language(customer_query)
        print(f"Detected Language: {target_lang}")

        if target_lang not in self.supported_languages:
            return f"I apologize, but I currently only support English, Spanish, and French. Your query was detected in {target_lang}."

        # 2. Retrieve relevant information from Knowledge Base
        relevant_kb_info = self._get_knowledge_base_info(customer_query, target_lang)
        print(f"Retrieved KB Info Keys: {list(relevant_kb_info.keys())}")

        # 3. Construct the InCLT prompt for the LLM
        inclt_prompt = self._construct_inclt_prompt(customer_query, target_lang, relevant_kb_info)
        # print(f"\n--- Generated InCLT Prompt ---\n{inclt_prompt}\n-----------------------------") # Uncomment to see the full prompt

        # 4. Call the LLM with the constructed prompt
        llm_response = self._call_llm(inclt_prompt)

        # 5. Return the LLM's response (which is ideally in the target language)
        return llm_response

# --- Example Usage ---
if __name__ == "__main__":
    chatbot = MultilingualChatbot()

    print("\n--- English Query ---")
    response_en = chatbot.process_query("What is your return policy?")
    print(f"Chatbot: {response_en}")

    print("\n--- Spanish Query ---")
    response_es = chatbot.process_query("¿Cuánto tarda el envío estándar y hay envío gratis?")
    print(f"Chatbot: {response_es}")

    print("\n--- French Query ---")
    response_fr = chatbot.process_query("Quelle est la garantie pour les produits électroniques?")
    print(f"Chatbot: {response_fr}")

    print("\n--- Spanish Query (Payment Methods) ---")
    response_es_payment = chatbot.process_query("¿Qué métodos de pago aceptan?")
    print(f"Chatbot: {response_es_payment}")

    print("\n--- English Query (Payment Methods) ---")
    response_en_payment = chatbot.process_query("What payment methods do you accept?")
    print(f"Chatbot: {response_en_payment}")

    print("\n--- Query in unsupported language (mock) ---")
    response_unsupported = chatbot.process_query("Wie lange dauert der Versand?") # German
    print(f"Chatbot: {response_unsupported}")
