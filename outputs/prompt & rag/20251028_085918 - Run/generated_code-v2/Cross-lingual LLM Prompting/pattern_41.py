class MultilingualChatbot:
    def __init__(self):
        self.knowledge_base = {
            "What are your operating hours?": "Our operating hours are Monday to Friday, 9 AM to 5 PM EST.",
            "How can I reset my password?": "You can reset your password by clicking on the 'Forgot Password' link on the login page.",
            "What products do you offer?": "We offer a wide range of products including electronics, home goods, and apparel.",
            "How do I contact customer support?": "You can contact customer support via email at support@example.com or call us at 1-800-123-4567."
        }
        self.translation_map = {
            "en": {
                "What are your operating hours?": "What are your operating hours?",
                "Our operating hours are Monday to Friday, 9 AM to 5 PM EST.": "Our operating hours are Monday to Friday, 9 AM to 5 PM EST.",
                "How can I reset my password?": "How can I reset my password?",
                "You can reset your password by clicking on the 'Forgot Password' link on the login page.": "You can reset your password by clicking on the 'Forgot Password' link on the login page.",
                "What products do you offer?": "What products do you offer?",
                "We offer a wide range of products including electronics, home goods, and apparel.": "We offer a wide range of products including electronics, home goods, and apparel.",
                "How do I contact customer support?": "How do I contact customer support?",
                "You can contact customer support via email at support@example.com or call us at 1-800-123-4567.": "You can contact customer support via email at support@example.com or call us at 1-800-123-4567.",
                "hello": "hello",
                "hi": "hi",
                "goodbye": "goodbye",
                "thank you": "thank you",
                "thanks": "thanks",
                "hours": "hours",
                "password": "password",
                "products": "products",
                "support": "support"
            },
            "es": {
                "What are your operating hours?": "¿Cuáles son sus horarios de atención?",
                "Our operating hours are Monday to Friday, 9 AM to 5 PM EST.": "Nuestros horarios de atención son de lunes a viernes, de 9 a.m. a 5 p.m. EST.",
                "How can I reset my password?": "¿Cómo puedo restablecer mi contraseña?",
                "You can reset your password by clicking on the 'Forgot Password' link on the login page.": "Puede restablecer su contraseña haciendo clic en el enlace 'Olvidé mi contraseña' en la página de inicio de sesión.",
                "What products do you offer?": "¿Qué productos ofrecen?",
                "We offer a wide range of products including electronics, home goods, and apparel.": "Ofrecemos una amplia gama de productos que incluyen electrónica, artículos para el hogar y ropa.",
                "How do I contact customer support?": "¿Cómo me comunico con el servicio de atención al cliente?",
                "You can contact customer support via email at support@example.com or call us at 1-800-123-4567.": "Puede comunicarse con el servicio de atención al cliente por correo electrónico a support@example.com o llamarnos al 1-800-123-4567.",
                "hello": "hola",
                "hi": "hola",
                "goodbye": "adiós",
                "thank you": "gracias",
                "thanks": "gracias",
                "hours": "horarios",
                "password": "contraseña",
                "products": "productos",
                "support": "soporte"
            }
        }

    def _detect_language(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in ["hola", "adiós", "gracias", "contraseña", "productos", "soporte", "horarios"]):
            return "es"
        return "en"

    def _translate(self, text, target_lang):
        source_lang = self._detect_language(text) # In a real scenario, this would be more robust
        if source_lang == target_lang:
            return text

        # Simple lookup for demonstration
        for original_en, translated_es in self.translation_map["es"].items():
            if text == original_en:
                return translated_es
            if text == translated_es and target_lang == "en":
                return original_en
        return text # Return original if no translation found

    def _multilingual_llm(self, prompt, target_lang):
        # Simulated LLM behavior based on keywords in the prompt
        if "Our operating hours are Monday to Friday, 9 AM to 5 PM EST." in prompt:
            return self._translate("Our operating hours are Monday to Friday, 9 AM to 5 PM EST.", target_lang)
        elif "You can reset your password by clicking on the 'Forgot Password' link on the login page." in prompt:
            return self._translate("You can reset your password by clicking on the 'Forgot Password' link on the login page.", target_lang)
        elif "We offer a wide range of products including electronics, home goods, and apparel." in prompt:
            return self._translate("We offer a wide range of products including electronics, home goods, and apparel.", target_lang)
        elif "You can contact customer support via email at support@example.com or call us at 1-800-123-4567." in prompt:
            return self._translate("You can contact customer support via email at support@example.com or call us at 1-800-123-4567.", target_lang)
        elif "hello" in prompt.lower() or "hi" in prompt.lower():
            return self._translate("Hello! How can I assist you today?", target_lang)
        elif "goodbye" in prompt.lower() or "bye" in prompt.lower():
            return self._translate("Goodbye! Have a great day!", target_lang)
        elif "thank you" in prompt.lower() or "thanks" in prompt.lower():
            return self._translate("You're welcome!", target_lang)
        return self._translate("I'm sorry, I don't have enough information to answer that question.", target_lang)

    def _get_relevant_kb_examples(self, english_query, num_examples=2):
        relevant_examples = []
        # Simple keyword matching for relevance
        query_words = english_query.lower().split()
        for q_en, a_en in self.knowledge_base.items():
            if any(word in q_en.lower() for word in query_words):
                relevant_examples.append((q_en, a_en))
            if len(relevant_examples) >= num_examples:
                break
        return relevant_examples

    def _construct_inclt_prompt(self, original_query, target_lang, english_query, relevant_examples):
        prompt_parts = []
        prompt_parts.append(f"You are a helpful multilingual customer support assistant.")
        prompt_parts.append(f"Answer the following question in {target_lang.upper()}.")
        prompt_parts.append(f"\nHere are some examples that use both English and {target_lang.upper()} to help you.\n")

        for i, (q_en, a_en) in enumerate(relevant_examples):
            q_target = self._translate(q_en, target_lang)
            a_target = self._translate(a_en, target_lang)
            prompt_parts.append(f"Example {i+1} (English):\nQuestion: {q_en}\nAnswer: {a_en}\n")
            prompt_parts.append(f"Example {i+1} ({target_lang.upper()}):\nQuestion: {q_target}\nAnswer: {a_target}\n")

        prompt_parts.append(f"---\n")
        prompt_parts.append(f"Original Query ({target_lang.upper()}): {original_query}")
        prompt_parts.append(f"English Query: {english_query}")
        prompt_parts.append(f"Answer in {target_lang.upper()}:\n")

        return "\n".join(prompt_parts)

    def get_answer(self, customer_query):
        target_lang = self._detect_language(customer_query)
        print(f"Detected Language: {target_lang.upper()}")

        english_query = self._translate(customer_query, "en")
        print(f"Translated English Query: {english_query}")

        relevant_examples = self._get_relevant_kb_examples(english_query)
        print(f"Found {len(relevant_examples)} relevant KB examples.")

        inclt_prompt = self._construct_inclt_prompt(customer_query, target_lang, english_query, relevant_examples)
        print(f"\n--- Generated InCLT Prompt ---\n{inclt_prompt}\n-----------------------------\n")

        llm_response = self._multilingual_llm(inclt_prompt, target_lang)
        return llm_response

if __name__ == "__main__":
    chatbot = MultilingualChatbot()
    print("Multilingual Customer Support Chatbot (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = chatbot.get_answer(user_input)
        print(f"Chatbot: {response}")
