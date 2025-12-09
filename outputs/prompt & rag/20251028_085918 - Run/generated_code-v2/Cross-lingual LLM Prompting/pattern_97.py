class MultilingualChatbot:
    def __init__(self):
        self.knowledge_base = [
            {
                "en_query": "What is your return policy?",
                "en_answer": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
                "es_query": "¿Cuál es su política de devoluciones?",
                "es_answer": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra con un recibo válido."
            },
            {
                "en_query": "How can I track my order?",
                "en_answer": "You can track your order using the tracking number provided in your shipping confirmation email.",
                "es_query": "¿Cómo puedo rastrear mi pedido?",
                "es_answer": "Puede rastrear su pedido utilizando el número de seguimiento proporcionado en su correo electrónico de confirmación de envío."
            },
            {
                "en_query": "Do you offer international shipping?",
                "en_answer": "Yes, we offer international shipping to most countries. Shipping fees and delivery times vary by destination.",
                "es_query": "¿Ofrecen envíos internacionales?",
                "es_answer": "Sí, ofrecemos envíos internacionales a la mayoría de los países. Las tarifas de envío y los tiempos de entrega varían según el destino."
            }
        ]

    def _retrieve_examples(self, query: str, target_language_code: str, num_examples: int = 2) -> list:
        return self.knowledge_base[:num_examples]

    def build_inclt_prompt(self, query: str, target_language_full_name: str, in_context_examples: list) -> str:
        prompt_parts = [
            f"You are a helpful customer support assistant for an e-commerce company.",
            f"The user's query is in {target_language_full_name}. Please provide a helpful and concise answer in {target_language_full_name}.",
            f"Here are some examples of customer questions and answers in both English and {target_language_full_name}:"
        ]

        for i, example in enumerate(in_context_examples):
            prompt_parts.append(f"\n--- Example {i+1} ---")
            prompt_parts.append(f"English Question: {example['en_query']}")
            prompt_parts.append(f"English Answer: {example['en_answer']}")
            prompt_parts.append(f"{target_language_full_name} Question: {example[f'{target_language_full_name.lower()[:2]}_query']}") # Use lowercased first two chars for dict key
            prompt_parts.append(f"{target_language_full_name} Answer: {example[f'{target_language_full_name.lower()[:2]}_answer']}")

        prompt_parts.append(f"\n--- Current Customer Query ---")
        prompt_parts.append(f"Question in {target_language_full_name}: {query}")
        prompt_parts.append(f"Answer in {target_language_full_name}:")

        return "\n".join(prompt_parts)

    def _simulate_llm_response(self, prompt: str) -> str:
        query_tag_prefix = "Question in "
        
        last_query_index = prompt.rfind(query_tag_prefix)
        if last_query_index == -1:
            return "I am unable to understand the query from the prompt."
        
        start_of_query_line = prompt.rfind('\n', 0, last_query_index) + 1
        end_of_query_line = prompt.find('\n', last_query_index)
        if end_of_query_line == -1:
            end_of_query_line = len(prompt)
            
        customer_question_line = prompt[start_of_query_line:end_of_query_line].strip()
        
        try:
            lang_part, question_text = customer_question_line.split(": ", 1)
            target_language_cap = lang_part.replace(query_tag_prefix, "").strip()
            target_lang_code = "es" if "Spanish" in target_language_cap else "en"
            
            customer_question = question_text.strip()
        except ValueError:
            return "I am unable to parse the customer query from the prompt."

        for example in self.knowledge_base:
            if target_lang_code == "es" and customer_question.lower() in example["es_query"].lower():
                return example["es_answer"]
            elif target_lang_code == "en" and customer_question.lower() in example["en_query"].lower():
                return example["en_answer"]

        if target_lang_code == "es":
            return f"Gracias por su pregunta sobre \"{customer_question}\". Estoy aquí para ayudarle con cualquier otra consulta que pueda tener."
        else:
            return f"Thank you for your question about \"{customer_question}\". I am here to assist you with any further inquiries you might have."

    def get_response(self, customer_query: str, target_language_code: str) -> str:
        lang_map = {"en": "English", "es": "Spanish"}
        if target_language_code not in lang_map:
            return "Unsupported language code. Please use 'en' for English or 'es' for Spanish."
        
        target_language_full = lang_map[target_language_code]

        in_context_examples = self._retrieve_examples(customer_query, target_language_code)
        prompt = self.build_inclt_prompt(customer_query, target_language_full, in_context_examples)
        response = self._simulate_llm_response(prompt)
        return response

if __name__ == "__main__":
    chatbot = MultilingualChatbot()

    print("--- English Customer Query ---")
    query_en = "What is your return policy?"
    response_en = chatbot.get_response(query_en, "en")
    print(f"Customer: {query_en}")
    print(f"Chatbot: {response_en}\n")

    print("--- Spanish Customer Query ---")
    query_es = "¿Cómo puedo rastrear mi pedido?"
    response_es = chatbot.get_response(query_es, "es")
    print(f"Customer: {query_es}")
    print(f"Chatbot: {response_es}\n")

    print("--- English Customer Query (unlisted) ---")
    query_unlisted_en = "Where is my package?"
    response_unlisted_en = chatbot.get_response(query_unlisted_en, "en")
    print(f"Customer: {query_unlisted_en}")
    print(f"Chatbot: {response_unlisted_en}\n")

    print("--- Spanish Customer Query (unlisted) ---")
    query_unlisted_es = "¿Necesito una factura para la devolución?"
    response_unlisted_es = chatbot.get_response(query_unlisted_es, "es")
    print(f"Customer: {query_unlisted_es}")
    print(f"Chatbot: {response_unlisted_es}\n")