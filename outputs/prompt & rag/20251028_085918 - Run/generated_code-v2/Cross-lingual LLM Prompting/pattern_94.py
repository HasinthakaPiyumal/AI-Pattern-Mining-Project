class MockTranslationService:
    def translate(self, text, target_lang, source_lang="en"):
        if source_lang == "en" and target_lang == "es":
            if "status" in text.lower():
                return "¿Cuál es el estado de mi pedido?"
            elif "return" in text.lower():
                return "¿Cómo puedo devolver un artículo?"
            elif "cancel" in text.lower():
                return "¿Cómo puedo cancelar un pedido?"
            else:
                return f"[Translated to Spanish: {text}]"
        elif source_lang == "es" and target_lang == "en":
            if "estado" in text.lower():
                return "What is the status of my order?"
            elif "devolver" in text.lower():
                return "How can I return an item?"
            elif "cancelar" in text.lower():
                return "How can I cancel an order?"
            else:
                return f"[Translated to English: {text}]"
        return text # No translation if languages don't match mock

class ICLPromptGenerator:
    def __init__(self, source_lang="en"):
        self.source_lang = source_lang
        # In-context examples with dual-language queries and responses
        self.in_context_examples = [
            {
                "source_query": "What is the status of my recent order?",
                "target_query": "¿Cuál es el estado de mi pedido reciente?",
                "source_response": "Your order with number #XYZ123 has been shipped and should arrive within 2-3 business days.",
                "target_response": "Su pedido con número #XYZ123 ha sido enviado y debería llegar en 2-3 días hábiles."
            },
            {
                "source_query": "How can I initiate a return for a damaged product?",
                "target_query": "¿Cómo puedo iniciar una devolución de un producto dañado?",
                "source_response": "To initiate a return for a damaged product, please visit our 'Returns and Refunds' section and follow the instructions.",
                "target_response": "Para iniciar una devolución de un producto dañado, por favor visite nuestra sección de 'Devoluciones y Reembolsos' y siga las instrucciones."
            },
            {
                "source_query": "Can I change the shipping address for my order?",
                "target_query": "¿Puedo cambiar la dirección de envío de mi pedido?",
                "source_response": "Unfortunately, shipping addresses cannot be changed once an order has been placed. Please contact support for assistance.",
                "target_response": "Lamentablemente, las direcciones de envío no se pueden cambiar una vez que se ha realizado un pedido. Por favor, póngase en contacto con soporte para obtener ayuda."
            }
        ]

    def generate_prompt(self, customer_query_target_lang, target_lang):
        prompt_parts = [
            "You are a helpful customer support assistant for a global e-commerce platform.",
            f"Your task is to respond to customer queries in {target_lang.capitalize()}.",
            f"Here are examples of customer queries and their appropriate responses, presented in both {self.source_lang.capitalize()} (source language) and {target_lang.capitalize()} (target language). Use this information to understand and respond to the user's query in {target_lang.capitalize()}:\n"
        ]

        for i, example in enumerate(self.in_context_examples):
            prompt_parts.append(f"--- Example {i+1} ---")
            prompt_parts.append(f"Source Query ({self.source_lang.capitalize()}): {example['source_query']}")
            prompt_parts.append(f"Target Query ({target_lang.capitalize()}): {example['target_query']}")
            prompt_parts.append(f"Response ({target_lang.capitalize()}): {example['target_response']}")
            prompt_parts.append(f"Response ({self.source_lang.capitalize()}): {example['source_response']}\n")

        prompt_parts.append(f"--- Customer Query ---")
        prompt_parts.append(f"Target Query ({target_lang.capitalize()}): {customer_query_target_lang}")
        prompt_parts.append(f"Response ({target_lang.capitalize()}):")

        return "\n".join(prompt_parts)

class MockLLMInterface:
    def generate_response(self, prompt, target_lang="es"):
        # Simulate LLM behavior based on keywords for demonstration
        prompt_lower = prompt.lower()
        if "estado de mi pedido" in prompt_lower:
            return "Su pedido está actualmente en tránsito y se espera su entrega dentro de 2 días hábiles."
        elif "devolver un producto" in prompt_lower:
            return "Para iniciar una devolución, por favor, visite la sección de 'Devoluciones' en nuestro sitio web y siga las instrucciones. Necesitará su número de pedido."
        elif "cambiar la dirección" in prompt_lower:
            return "Lamentablemente, no podemos modificar la dirección de envío una vez que el pedido ha sido procesado. Le recomendamos que se ponga en contacto con la empresa de mensajería directamente."
        elif "cancelar un pedido" in prompt_lower:
            return "Si su pedido aún no ha sido enviado, puede cancelarlo desde su historial de pedidos en nuestra web. En caso contrario, por favor, póngase en contacto con soporte."
        else:
            return f"Gracias por contactarnos. Estoy aquí para ayudarle con su consulta en {target_lang}."

class MultilingualCustomerChatbot:
    def __init__(self, source_lang="en"):
        self.translation_service = MockTranslationService()
        self.prompt_generator = ICLPromptGenerator(source_lang=source_lang)
        self.llm_interface = MockLLMInterface()
        self.source_lang = source_lang

    def get_customer_support_response(self, customer_query, target_lang):
        print(f"\nCustomer Query ({target_lang.upper()}): {customer_query}")

        # 1. (Conceptual) Language Detection - assumed `target_lang` is provided

        # 2. (Conceptual) Query Translation to source_lang (for internal analysis, not directly for ICL prompt input)
        # For InCLT, the target_lang query is directly used in the prompt for the LLM.
        # source_query_translation = self.translation_service.translate(customer_query, self.source_lang, target_lang)
        # print(f"(Simulated Internal Source Query Translation: {source_query_translation})")

        # 3. Generate InCLT Prompt
        icl_prompt = self.prompt_generator.generate_prompt(customer_query, target_lang)
        print("\n--- Generated ICL Prompt ---")
        print(icl_prompt)
        print("----------------------------")

        # 4. Get LLM Response
        llm_response = self.llm_interface.generate_response(icl_prompt, target_lang)
        print(f"\nChatbot Response ({target_lang.upper()}): {llm_response}")
        return llm_response

if __name__ == "__main__":
    chatbot = MultilingualCustomerChatbot()

    # Test with a Spanish query
    spanish_query = "¿Cuál es el estado de mi pedido?"
    chatbot.get_customer_support_response(spanish_query, "es")

    # Test with another Spanish query
    spanish_query_2 = "Quiero devolver un producto que recibí ayer."
    chatbot.get_customer_support_response(spanish_query_2, "es")

    # Test with a query not explicitly in examples, to see general response
    spanish_query_3 = "Necesito ayuda para cambiar mis datos de contacto."
    chatbot.get_customer_support_response(spanish_query_3, "es")

    # Example of a conceptual flow if we were to translate a source query to target for some reason (not directly InCLT for input)
    # english_query_for_translation = "I need to cancel my subscription."
    # translated_query = chatbot.translation_service.translate(english_query_for_translation, "es", "en")
    # print(f"\nTranslated English Query to Spanish: {translated_query}")
    # chatbot.get_customer_support_response(translated_query, "es")
