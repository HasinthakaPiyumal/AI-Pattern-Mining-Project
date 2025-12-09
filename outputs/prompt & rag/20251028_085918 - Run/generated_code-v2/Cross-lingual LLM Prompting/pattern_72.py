class MultilingualChatbot:
    def __init__(self):
        self.icl_examples = [
            {
                "EN_query": "What is your return policy?",
                "EN_response": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
                "ES_query": "¿Cuál es su política de devoluciones?",
                "ES_response": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días de la compra con un recibo válido.",
                "FR_query": "Quelle est votre politique de retour?",
                "FR_response": "Notre politique de retour permet les retours dans les 30 jours suivant l'achat avec un reçu valide."
            },
            {
                "EN_query": "How can I track my order?",
                "EN_response": "You can track your order using the tracking number provided in your shipping confirmation email.",
                "ES_query": "¿Cómo puedo rastrear mi pedido?",
                "ES_response": "Puede rastrear su pedido utilizando el número de seguimiento proporcionado en su correo electrónico de confirmación de envío.",
                "FR_query": "Comment puis-je suivre ma commande?",
                "FR_response": "Vous pouvez suivre votre commande en utilisant le numéro de suivi fourni dans votre e-mail de confirmation d'expédition."
            },
            {
                "EN_query": "What are your operating hours?",
                "EN_response": "We are open from 9 AM to 5 PM, Monday to Friday.",
                "ES_query": "¿Cuáles son sus horas de operación?",
                "ES_response": "Estamos abiertos de 9 AM a 5 PM, de lunes a viernes.",
                "FR_query": "Quelles sont vos heures d'ouverture?",
                "FR_response": "Nous sommes ouverts de 9h à 17h, du lundi au vendredi."
            }
        ]

    def _construct_prompt(self, customer_query: str, source_language: str, target_language: str) -> str:
        prompt = (
            "Instructions: You are a helpful customer support assistant. Provide concise and accurate answers. "
            "For cross-lingual queries, understand the source and respond in the target language, "
            "leveraging the examples provided in both source and target languages.\n\n"
            "In-Context Examples:\n"
        )

        for i, example in enumerate(self.icl_examples):
            prompt += f"Example {i+1}:\n"
            prompt += f"Query ({source_language}): {example.get(f'{source_language.upper()}_query', 'N/A')}\n"
            prompt += f"Response ({source_language}): {example.get(f'{source_language.upper()}_response', 'N/A')}\n"
            prompt += f"Query ({target_language}): {example.get(f'{target_language.upper()}_query', 'N/A')}\n"
            prompt += f"Response ({target_language}): {example.get(f'{target_language.upper()}_response', 'N/A')}\n\n"

        prompt += (
            f"Current Customer Query (Source Language: {source_language}, Target Language: {target_language}):\n"
            f"Customer: {customer_query}\n"
            f"Response ({target_language}):"
        )
        return prompt

    def _simulate_llm_response(self, prompt: str, target_language: str) -> str:
        # In a real application, this would call a multilingual LLM API (e.g., OpenAI, HuggingFace)
        # For simulation, we'll provide a generic response based on the target language.

        if target_language.lower() == "es":
            return "Gracias por su pregunta. Un agente de soporte se pondrá en contacto con usted pronto para ayudarle.\n"
        elif target_language.lower() == "fr":
            return "Merci pour votre question. Un agent de support vous contactera bientôt pour vous aider.\n"
        else:
            # Default to English or handle other languages
            return "Thank you for your question. A support agent will contact you shortly to assist you.\n"

    def ask_chatbot(self, customer_query: str, source_language: str, target_language: str) -> str:
        print(f"\n--- Customer Query in {source_language.upper()} ---")
        print(f"Query: {customer_query}")
        print(f"Intended Response Language: {target_language.upper()}")

        prompt = self._construct_prompt(customer_query, source_language, target_language)
        print("\n--- Constructed Prompt (abbreviated) ---")
        # Print only a part of the prompt to avoid excessive output, as ICL examples can be long
        print(prompt[:500] + "...\n") 

        response = self._simulate_llm_response(prompt, target_language)
        print(f"--- Chatbot Response in {target_language.upper()} ---")
        print(response)
        return response

# Example Usage:
if __name__ == "__main__":
    chatbot = MultilingualChatbot()

    # Spanish query, English response
    chatbot.ask_chatbot("¿Cómo puedo devolver un producto?", "es", "en")

    # English query, Spanish response
    chatbot.ask_chatbot("I need help with my recent purchase.", "en", "es")

    # French query, English response
    chatbot.ask_chatbot("J'ai une question concernant ma facture.", "fr", "en")

    # English query, French response
    chatbot.ask_chatbot("Where is your store located?", "en", "fr")
