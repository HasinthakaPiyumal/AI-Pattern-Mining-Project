class MultilingualFAQChatbot:
    def __init__(self):
        self.knowledge_base = {
            "en": {
                "returns": {"question": "How do I return a faulty product?", "answer": "To return a faulty product, please visit our returns page and follow the instructions provided."},
                "shipping": {"question": "What are your shipping options and costs?", "answer": "We offer standard and express shipping. Costs vary by destination and speed. Please check our shipping page for details."},
                "warranty": {"question": "What is the warranty policy for electronics?", "answer": "Most electronics come with a 1-year manufacturer's warranty. Extended warranties may be available for purchase."}
            },
            "es": {
                "returns": {"question": "¿Quiero devolver un artículo defectuoso. ¿Cómo lo hago?", "answer": "Para devolver un producto defectuoso, visite nuestra página de devoluciones y siga las instrucciones proporcionadas."},
                "shipping": {"question": "¿Cuáles son sus opciones y costos de envío?", "answer": "Ofrecemos envío estándar y exprés. Los costos varían según el destino y la velocidad. Consulte nuestra página de envío para obtener detalles."},
                "warranty": {"question": "¿Cuál es la política de garantía para productos electrónicos?", "answer": "La mayoría de los productos electrónicos vienen con una garantía del fabricante de 1 año. Las garantías extendidas pueden estar disponibles para su compra."}
            }
        }

    def _mock_embedding_model(self, text: str):
        if "return" in text.lower() or "devolver" in text.lower():
            return "returns"
        elif "shipping" in text.lower() or "envío" in text.lower():
            return "shipping"
        elif "warranty" in text.lower() or "garantía" in text.lower():
            return "warranty"
        return "general"

    def _construct_inclt_prompt(self, user_query: str, target_lang: str, source_example: dict, target_example: dict) -> str:
        prompt = f"""You are a helpful and multilingual customer support agent. Your goal is to provide accurate answers to customer queries in the specified target language.

Here are some examples to help you understand how to answer cross-lingual questions:

Source Language Example (English):
Question: {source_example['question']}
Answer: {source_example['answer']}

Target Language Example ({target_lang.upper()}):
Question: {target_example['question']}
Answer: {target_example['answer']}

Now, answer the following question in {target_lang.upper()}:
Customer Query: {user_query}
Answer:"""
        return prompt

    def _mock_llm_response(self, prompt: str, target_lang: str) -> str:
        # Simplified LLM behavior based on prompt content
        if "return" in prompt.lower() or "devolver" in prompt.lower():
            if target_lang == "es":
                return "Para iniciar una devolución, por favor visite nuestra página de devoluciones y siga las instrucciones. Necesitará su número de pedido." # Spanish specific
            else:
                return "To initiate a return, please visit our returns page and follow the instructions. You will need your order number."
        elif "shipping" in prompt.lower() or "envío" in prompt.lower():
            if target_lang == "es":
                return "Nuestras opciones de envío y los costos detallados se encuentran en la sección de envíos de nuestro sitio web." # Spanish specific
            else:
                return "Our shipping options and detailed costs can be found on the shipping section of our website."
        elif "warranty" in prompt.lower() or "garantía" in prompt.lower():
            if target_lang == "es":
                return "La política de garantía estándar para la mayoría de los productos electrónicos es de un año. Consulte el manual del producto para detalles específicos." # Spanish specific
            else:
                return "The standard warranty policy for most electronics is one year. Please refer to your product manual for specific details."
        return f"I'm sorry, I cannot provide a specific answer to that in {target_lang.upper()} at the moment. Please visit our help center."

    def get_response(self, user_query: str, target_lang: str = "es") -> str:
        identified_topic = self._mock_embedding_model(user_query)

        source_example = self.knowledge_base["en"].get(identified_topic, self.knowledge_base["en"]["returns"])
        target_example = self.knowledge_base[target_lang].get(identified_topic, self.knowledge_base[target_lang]["returns"])

        inclt_prompt = self._construct_inclt_prompt(user_query, target_lang, source_example, target_example)
        llm_answer = self._mock_llm_response(inclt_prompt, target_lang)
        return llm_answer

if __name__ == "__main__":
    chatbot = MultilingualFAQChatbot()

    print("\n--- Multilingual Customer Support Chatbot (InCLT) ---")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nEnter your query (e.g., in Spanish): ")
        if user_input.lower() == 'exit':
            break

        # For demonstration, assume Spanish as target language
        response = chatbot.get_response(user_input, target_lang="es")
        print(f"Chatbot ({'ES'.upper()}): {response}")

        user_input_en = input("\nEnter your query (e.g., in English): ")
        if user_input_en.lower() == 'exit':
            break
        response_en = chatbot.get_response(user_input_en, target_lang="en")
        print(f"Chatbot ({'EN'.upper()}): {response_en}")
