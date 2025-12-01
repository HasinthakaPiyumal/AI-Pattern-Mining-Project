class MultilingualPromptGenerator:
    def __init__(self):
        self.icl_examples = []

    def add_icl_example(self, english_query: str, english_answer: str, target_lang_query: str, target_lang_answer: str):
        self.icl_examples.append({
            "english_query": english_query,
            "english_answer": english_answer,
            "target_lang_query": target_lang_query,
            "target_lang_answer": target_lang_answer
        })

    def generate_prompt(self, customer_query: str, detected_language: str) -> str:
        prompt_parts = [
            "You are a helpful customer support assistant for an e-commerce platform. ",
            "Answer the following questions based on the provided examples. ",
            "If you cannot find a direct answer, apologize and ask for more details.",
            "Provide concise and helpful responses.\n"
        ]

        for example in self.icl_examples:
            prompt_parts.append(f"User: {example['english_query']}\nAssistant: {example['english_answer']}")

            if detected_language.lower() == "spanish":
                prompt_parts.append(f"User: {example['target_lang_query']}\nAssistant: {example['target_lang_answer']}")

        prompt_parts.append(f"User: {customer_query}\nAssistant:")

        return "\n".join(prompt_parts)


class LLMService:
    def get_response(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        if "return policy" in prompt_lower and "english" in prompt_lower:
            return "Our return policy allows returns within 30 days of purchase with a valid receipt."
        elif "politica de devolucion" in prompt_lower or "devoluciones" in prompt_lower:
            return "Nuestra política de devoluciones permite devoluciones dentro de los 30 días de la compra con un recibo válido."
        elif "shipping cost" in prompt_lower or "shipping" in prompt_lower:
            return "Standard shipping is free for orders over $50. Expedited shipping costs vary by location."
        elif "costo de envio" in prompt_lower or "envio" in prompt_lower:
            return "El envío estándar es gratuito para pedidos superiores a $50. Los costos de envío urgente varían según la ubicación."
        elif "problem with my order" in prompt_lower:
            return "I understand you have a problem with your order. Please provide your order number for me to assist you further."
        elif "problema con mi pedido" in prompt_lower:
            return "Entiendo que tiene un problema con su pedido. Por favor, proporcione su número de pedido para que pueda ayudarle mejor."
        else:
            return "I am sorry, I could not find a direct answer to your question. Could you please rephrase it or provide more details?"


class CustomerSupportAssistant:
    def __init__(self):
        self.prompt_generator = MultilingualPromptGenerator()
        self.llm_service = LLMService()
        self._initialize_icl_examples()

    def _initialize_icl_examples(self):
        self.prompt_generator.add_icl_example(
            english_query="What is your return policy?",
            english_answer="Our return policy allows returns within 30 days with a receipt.",
            target_lang_query="¿Cuál es su política de devoluciones?",
            target_lang_answer="Nuestra política de devoluciones permite devoluciones dentro de los 30 días con el recibo."
        )
        self.prompt_generator.add_icl_example(
            english_query="How much does shipping cost?",
            english_answer="Shipping costs vary based on location and speed. Standard shipping is usually free for orders over $50.",
            target_lang_query="¿Cuánto cuesta el envío?",
            target_lang_answer="Los costos de envío varían según la ubicación y la velocidad. El envío estándar suele ser gratuito para pedidos superiores a $50."
        )

    def _detect_language(self, text: str) -> str:
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ["¿", "ñ", "á", "é", "í", "ó", "ú", "qué", "cuánto", "política", "devoluciones", "envío", "problema"]):
            return "Spanish"
        else:
            return "English"

    def get_answer(self, customer_query: str) -> str:
        detected_language = self._detect_language(customer_query)
        print(f"\nCustomer Query: '{customer_query}' (Detected Language: {detected_language})")

        prompt = self.prompt_generator.generate_prompt(customer_query, detected_language)
        print(f"\n--- Generated Prompt for LLM ---\n{prompt}\n---\n")

        llm_response = self.llm_service.get_response(prompt)
        return llm_response


if __name__ == "__main__":
    assistant = CustomerSupportAssistant()

    print("\n===== Testing Customer Support Assistant =====")

    response1 = assistant.get_answer("What is your return policy?")
    print(f"Assistant Response: {response1}")

    response2 = assistant.get_answer("¿Cuál es su política de devoluciones?")
    print(f"Assistant Response: {response2}")

    response3 = assistant.get_answer("How much does shipping cost?")
    print(f"Assistant Response: {response3}")

    response4 = assistant.get_answer("¿Cuál es el costo de envío?")
    print(f"Assistant Response: {response4}")

    response5 = assistant.get_answer("I have a problem with my order.")
    print(f"Assistant Response: {response5}")

    response6 = assistant.get_answer("Tengo un problema con mi pedido.")
    print(f"Assistant Response: {response6}")

    response7 = assistant.get_answer("Do you sell blue widgets?")
    print(f"Assistant Response: {response7}")

    response8 = assistant.get_answer("¿Venden aparatos azules?")
    print(f"Assistant Response: {response8}")
