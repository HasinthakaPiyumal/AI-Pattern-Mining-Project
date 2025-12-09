class MultilingualChatbot:
    def __init__(self):
        self.icl_example_database = [
            {
                "source_lang": "English",
                "target_lang": "Spanish",
                "source_query": "Where is my order?",
                "source_response": "Please provide your order number so I can check its status.",
                "cross_lingual_query": "¿Dónde está mi pedido?",
                "target_response": "Por favor, proporcione su número de pedido para que pueda verificar su estado."
            },
            {
                "source_lang": "English",
                "target_lang": "French",
                "source_query": "How can I return an item?",
                "source_response": "You can initiate a return from your order history page on our website.",
                "cross_lingual_query": "Comment puis-je retourner un article ?",
                "target_response": "Vous pouvez initier un retour depuis votre page d'historique de commandes sur notre site web."
            },
            {
                "source_lang": "Spanish",
                "target_lang": "English",
                "source_query": "¿Cuál es la política de reembolso?",
                "source_response": "Nuestra política de reembolso permite devoluciones dentro de los 30 días posteriores a la compra.",
                "cross_lingual_query": "What is the refund policy?",
                "target_response": "Our refund policy allows returns within 30 days of purchase."
            },
            {
                "source_lang": "French",
                "target_lang": "Spanish",
                "source_query": "Mon produit est arrivé endommagé.",
                "source_response": "Veuillez nous envoyer des photos du produit endommagé et de son emballage.",
                "cross_lingual_query": "Mi producto llegó dañado.",
                "target_response": "Por favor, envíenos fotos del producto dañado y su embalaje."
            }
        ]

    def _select_icl_examples(self, target_lang, num_examples=2):
        selected_examples = []
        for example in self.icl_example_database:
            if example["target_lang"] == target_lang:
                selected_examples.append(example)
            if len(selected_examples) == num_examples:
                break
        # If not enough examples for target_lang, take general ones
        if len(selected_examples) < num_examples:
            for example in self.icl_example_database:
                if example not in selected_examples:
                    selected_examples.append(example)
                if len(selected_examples) == num_examples:
                    break
        return selected_examples

    def generate_icl_prompt(self, current_query: str, target_lang: str) -> str:
        examples = self._select_icl_examples(target_lang)
        prompt_parts = []

        for ex in examples:
            prompt_parts.append(f"Source Language: {ex["source_lang"]}")
            prompt_parts.append(f"Customer: {ex["source_query"]}")
            prompt_parts.append(f"Agent: {ex["source_response"]}\n")

            prompt_parts.append(f"Target Language: {ex["target_lang"]}")
            prompt_parts.append(f"Customer: {ex["cross_lingual_query"]}")
            prompt_parts.append(f"Agent: {ex["target_response"]}")
            prompt_parts.append("---\n")

        prompt_parts.append(f"Target Language: {target_lang}")
        prompt_parts.append(f"Customer: {current_query}")
        prompt_parts.append(f"Agent:") # LLM will complete this

        return "\n".join(prompt_parts)

    # Simulate an LLM call - in a real application, this would integrate with an actual LLM API
    def _call_llm(self, prompt: str) -> str:
        # For demonstration, a simple mock response based on keywords
        if "order status" in prompt.lower() or "estado de mi pedido" in prompt.lower():
            return "Please provide your order ID to check the status. / Por favor, proporcione su ID de pedido para verificar el estado."
        elif "return" in prompt.lower() or "devolver" in prompt.lower():
            return "You can find return instructions on our FAQ page. / Puede encontrar instrucciones de devolución en nuestra página de preguntas frecuentes."
        elif "refund" in prompt.lower() or "reembolso" in prompt.lower():
            return "Refunds are processed within 5-7 business days. / Los reembolsos se procesan en 5-7 días hábiles."
        else:
            return "I understand you have a question. How can I assist you further? / Entiendo que tiene una pregunta. ¿Cómo puedo ayudarle más?"

    def get_chatbot_response(self, user_query: str, query_lang: str, response_lang: str) -> str:
        # In a real scenario, `user_query` might need translation to `target_lang` if it's not already, 
        # or the LLM is expected to handle it directly in the prompt context.
        # For this demo, we assume the `current_query` passed to `generate_icl_prompt` 
        # is already in a suitable format for the LLM to process in the context of `response_lang`.
        # If the LLM is truly multilingual, it might infer, but for explicit InCLT, 
        # the prompt structure guides it.
        
        # For simplicity, we'll directly use the user_query as the 'current_query' in the target context for the prompt
        # assuming the LLM can handle the cross-lingual intent.
        prompt = self.generate_icl_prompt(user_query, response_lang)
        print("\n--- Generated Prompt ---")
        print(prompt)
        print("------------------------\n")
        llm_response = self._call_llm(prompt)
        return llm_response

if __name__ == "__main__":
    chatbot = MultilingualChatbot()

    print("--- Test Case 1: English Query, Spanish Response ---")
    user_query_1 = "I need to know the status of my order."
    response = chatbot.get_chatbot_response(user_query_1, "English", "Spanish")
    print(f"Chatbot Response: {response}\n")

    print("--- Test Case 2: Spanish Query, English Response ---")
    user_query_2 = "Quisiera devolver un artículo."
    response = chatbot.get_chatbot_response(user_query_2, "Spanish", "English")
    print(f"Chatbot Response: {response}\n")

    print("--- Test Case 3: French Query, Spanish Response ---")
    user_query_3 = "Quel est le délai de remboursement ?"
    response = chatbot.get_chatbot_response(user_query_3, "French", "Spanish")
    print(f"Chatbot Response: {response}\n")

    print("--- Test Case 4: English Query, French Response (General) ---")
    user_query_4 = "I have a question about a product."
    response = chatbot.get_chatbot_response(user_query_4, "English", "French")
    print(f"Chatbot Response: {response}\n")

