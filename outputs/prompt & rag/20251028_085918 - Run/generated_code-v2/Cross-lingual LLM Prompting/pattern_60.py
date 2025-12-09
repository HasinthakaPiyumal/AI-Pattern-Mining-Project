class MockMultilingualLLM:
    def __init__(self):
        self.responses = {
            "how to reset your password": "To reset your password, please navigate to your account settings and select the 'Forgot Password' option.",
            "how do i change my email": "You can change your email address in your profile settings under 'Account Information'.",
            "¿Cómo puedo reiniciar mi contraseña?": "Para reiniciar tu contraseña, por favor ve a la configuración de tu cuenta y selecciona la opción 'Olvidé mi contraseña'.",
            "¿Cómo cambio mi correo electrónico?": "Puedes cambiar tu dirección de correo electrónico en la configuración de tu perfil, en 'Información de la cuenta'.",
            "reset password": "To reset your password, please go to your account settings and click 'Forgot Password'.",
            "cambiar contraseña": "Para cambiar tu contraseña, ve a la configuración de tu cuenta y haz clic en 'Olvidé mi contraseña'.",
        }

    def generate_response(self, prompt):
        print(f"\n--- LLM Input Prompt ---\n{prompt}\n---\n")
        # Simple keyword matching for demonstration
        for key, value in self.responses.items():
            if key.lower() in prompt.lower():
                return value
        return "I am sorry, I couldn't find an answer to that question in my knowledge base. Can I help you with anything else?"

class KnowledgeBase:
    def __init__(self):
        self.kb = {
            "how to reset your password": "Go to settings and click 'Forgot Password'.",
            "how do i change my email": "Navigate to profile settings and update your email in 'Account Information'.",
            "where is my order": "You can track your order status in the 'My Orders' section of your account."
        }

    def retrieve_info(self, query):
        query_lower = query.lower()
        for q, a in self.kb.items():
            if query_lower in q.lower() or q.lower() in query_lower:
                return {"question": q, "answer": a}
        return None

class MockTranslationService:
    def __init__(self):
        self.translations = {
            "en_es": {
                "how to reset your password": "¿Cómo reiniciar tu contraseña?",
                "go to settings and click 'forgot password'.": "Ve a la configuración y haz clic en 'Olvidé mi contraseña'.",
                "how do i change my email": "¿Cómo cambio mi correo electrónico?",
                "navigate to profile settings and update your email in 'account information'.": "Navega a la configuración del perfil y actualiza tu correo electrónico en 'Información de la cuenta'.",
                "where is my order": "¿Dónde está mi pedido?",
                "you can track your order status in the 'my orders' section of your account.": "Puedes rastrear el estado de tu pedido en la sección 'Mis pedidos' de tu cuenta.",
                "i am sorry, i couldn't find an answer to that question in my knowledge base. can i help you with anything else?": "Lo siento, no pude encontrar una respuesta a esa pregunta en mi base de conocimientos. ¿Puedo ayudarte con algo más?"
            },
            "es_en": {
                "¿cómo puedo reiniciar mi contraseña?": "How to reset your password?",
                "¿cómo cambio mi correo electrónico?": "How do I change my email?",
                "¿dónde está mi pedido?": "Where is my order?"
            }
        }

    def translate(self, text, source_lang, target_lang):
        key = f"{source_lang}_{target_lang}"
        return self.translations.get(key, {}).get(text.lower(), text) # Return original text if no translation found for simplicity

class InCLTPromptEngineering:
    def __init__(self, translation_service, knowledge_base, source_lang="en"):
        self.translation_service = translation_service
        self.knowledge_base = knowledge_base
        self.source_lang = source_lang

    def generate_icl_prompt(self, customer_query, target_lang):
        # 1. Query Translator (Target to Source)
        translated_query_to_source = self.translation_service.translate(customer_query, target_lang, self.source_lang)
        print(f"Translated query to source ({self.source_lang}): {translated_query_to_source}")

        # 2. KB Retriever
        retrieved_info = self.knowledge_base.retrieve_info(translated_query_to_source)
        
        prompt_parts = []
        if retrieved_info:
            # 3. Example Generator (Source + Target)
            original_q_source = retrieved_info["question"]
            original_a_source = retrieved_info["answer"]
            
            translated_q_target = self.translation_service.translate(original_q_source, self.source_lang, target_lang)
            translated_a_target = self.translation_service.translate(original_a_source, self.source_lang, target_lang)
            
            # 4. Prompt Formatter - InCLT examples
            prompt_parts.append("Here are some examples of questions and answers:")
            prompt_parts.append(f"Question ({self.source_lang}): {original_q_source}")
            prompt_parts.append(f"Answer ({self.source_lang}): {original_a_source}")
            prompt_parts.append(f"Question ({target_lang}): {translated_q_target}")
            prompt_parts.append(f"Answer ({target_lang}): {translated_a_target}\n")
            
        prompt_parts.append(f"Based on the above examples and your knowledge, please answer the following question:")
        prompt_parts.append(f"Question ({target_lang}): {customer_query}")
        prompt_parts.append(f"Answer ({target_lang}):")

        return "\n".join(prompt_parts)

class ChatbotOrchestration:
    def __init__(self, llm, translation_service, knowledge_base):
        self.llm = llm
        self.translation_service = translation_service
        self.icl_prompt_engineer = InCLTPromptEngineering(translation_service, knowledge_base)
        self.source_lang = "en"

    def determine_language(self, text):
        # Simple mock language detection
        if any(char in text for char in "¿¡ñÑáéíóúÁÉÍÓÚ"): # Basic check for Spanish characters
            return "es"
        return "en"

    def get_response(self, customer_input):
        target_lang = self.determine_language(customer_input)
        print(f"Customer input detected in: {target_lang}")

        prompt = self.icl_prompt_engineer.generate_icl_prompt(customer_input, target_lang)
        llm_response = self.llm.generate_response(prompt)

        # Optional: Translate LLM response back if it's in source language and customer input was target
        if target_lang != self.source_lang and self.determine_language(llm_response) == self.source_lang:
            final_response = self.translation_service.translate(llm_response, self.source_lang, target_lang)
            print(f"Translated LLM response back to {target_lang}: {final_response}")
            return final_response

        return llm_response

if __name__ == "__main__":
    # Initialize components
    mock_llm = MockMultilingualLLM()
    knowledge_base = KnowledgeBase()
    translation_service = MockTranslationService()

    # Initialize Chatbot Orchestrator
    chatbot = ChatbotOrchestration(mock_llm, translation_service, knowledge_base)

    print("Multilingual Customer Support Chatbot (type 'exit' to quit)")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break

        response = chatbot.get_response(user_query)
        print(f"Chatbot: {response}\n")