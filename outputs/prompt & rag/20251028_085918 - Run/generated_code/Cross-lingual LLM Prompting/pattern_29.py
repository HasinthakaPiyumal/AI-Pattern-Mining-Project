class LLMIntegrationLayer:
    def get_response(self, prompt: str) -> str:
        if "reset password" in prompt.lower() and "spanish" in prompt.lower():
            return "Para restablecer su contraseña, vaya a la sección 'Mi cuenta' y seleccione 'Restablecer contraseña'."
        if "billing inquiry" in prompt.lower() and "french" in prompt.lower():
            return "Pour toute question de facturation, veuillez contacter notre service client."
        if "shipping status" in prompt.lower() and "german" in prompt.lower():
            return "Um den Versandstatus zu überprüfen, geben Sie bitte Ihre Bestellnummer ein."
        if "how do i reset my password" in prompt.lower():
            return "To reset your password, please go to the 'My Account' section and select 'Reset Password'."
        return f"LLM Response to: {prompt}"

class InCLTPromptingModule:
    def __init__(self):
        self._in_context_examples = [
            {
                "source_query": "How do I reset my password?",
                "target_language": "Spanish",
                "source_answer": "To reset your password, please go to the 'My Account' section and select 'Reset Password'.",
                "target_answer": "Para restablecer su contraseña, vaya a la sección 'Mi cuenta' y seleccione 'Restablecer contraseña'."
            },
            {
                "source_query": "I have a billing inquiry.",
                "target_language": "French",
                "source_answer": "For any billing inquiries, please contact our customer support.",
                "target_answer": "Pour toute question de facturation, veuillez contacter notre service client."
            },
            {
                "source_query": "What is the shipping status of my order?",
                "target_language": "German",
                "source_answer": "To check your shipping status, please provide your order number.",
                "target_answer": "Um den Versandstatus zu überprüfen, geben Sie bitte Ihre Bestellnummer ein."
            }
        ]

    def construct_prompt(self, customer_query: str, target_language: str) -> str:
        prompt_parts = []
        for example in self._in_context_examples:
            if example["target_language"].lower() == target_language.lower():
                prompt_parts.append(f"Source Query: {example['source_query']}")
                prompt_parts.append(f"Source Answer: {example['source_answer']}")
                prompt_parts.append(f"Target Query ({example['target_language']}): {self._translate_query_for_example(example['source_query'], example['target_language'])}")
                prompt_parts.append(f"Target Answer ({example['target_language']}): {example['target_answer']}")
                prompt_parts.append("")

        prompt_parts.append(f"Customer Query (English): {customer_query}")
        prompt_parts.append(f"Provide the answer in {target_language}.")
        return "\n".join(prompt_parts)

    def _translate_query_for_example(self, query: str, target_language: str) -> str:
        if target_language.lower() == "spanish":
            if "reset password" in query.lower():
                return "¿Cómo restablezco mi contraseña?"
            if "billing inquiry" in query.lower():
                return "Tengo una consulta de facturación."
        elif target_language.lower() == "french":
            if "billing inquiry" in query.lower():
                return "J'ai une question de facturation."
        elif target_language.lower() == "german":
            if "shipping status" in query.lower():
                return "Wie ist der Versandstatus meiner Bestellung?"
        return query # Fallback

class FAQManagementModule:
    def __init__(self):
        self._faqs = {
            "how to reset password": {
                "en": "To reset your password, please go to the 'My Account' section and select 'Reset Password'.",
                "es": "Para restablecer su contraseña, vaya a la sección 'Mi cuenta' y seleccione 'Restablecer contraseña'.",
                "fr": "Pour réinitialiser votre mot de passe, veuillez vous rendre dans la section 'Mon Compte' et sélectionner 'Réinitialiser le mot de passe'."
            },
            "billing inquiry": {
                "en": "For any billing inquiries, please contact our customer support.",
                "es": "Para cualquier consulta de facturación, por favor contacte a nuestro servicio al cliente.",
                "fr": "Pour toute question de facturation, veuillez contacter notre service client."
            },
            "shipping status": {
                "en": "To check your shipping status, please provide your order number.",
                "es": "Para verificar el estado de su envío, por favor proporcione su número de pedido.",
                "de": "Um den Versandstatus zu überprüfen, geben Sie bitte Ihre Bestellnummer ein."
            }
        }

    def get_faq_answer(self, query: str, target_language_code: str) -> str or None:
        query_lower = query.lower()
        for faq_key, answers in self._faqs.items():
            if faq_key in query_lower:
                return answers.get(target_language_code, answers.get("en"))
        return None

class Chatbot:
    def __init__(self):
        self.llm_integration = LLMIntegrationLayer()
        self.inclt_prompting = InCLTPromptingModule()
        self.faq_management = FAQManagementModule()

    def ask(self, customer_query: str, target_language: str) -> str:
        target_lang_code = {
            "english": "en",
            "spanish": "es",
            "french": "fr",
            "german": "de"
        }.get(target_language.lower(), "en")

        faq_answer = self.faq_management.get_faq_answer(customer_query, target_lang_code)
        if faq_answer:
            return faq_answer

        prompt = self.inclt_prompting.construct_prompt(customer_query, target_language)
        llm_response = self.llm_integration.get_response(prompt)
        return llm_response

if __name__ == "__main__":
    chatbot = Chatbot()
    print("Multilingual Customer Support Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("Enter your query (e.g., 'How do I reset my password?') or 'exit': ")
        if user_input.lower() == 'exit':
            break
        lang_input = input("Enter desired response language (e.g., 'Spanish', 'English', 'French', 'German'): ")
        response = chatbot.ask(user_input, lang_input)
        print(f"Chatbot: {response}")
