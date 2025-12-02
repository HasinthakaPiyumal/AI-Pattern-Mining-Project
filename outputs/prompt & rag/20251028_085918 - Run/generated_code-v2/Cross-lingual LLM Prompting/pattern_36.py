import os

class InCLTPromptManager:
    def __init__(self, cross_lingual_examples=None):
        self.cross_lingual_examples = cross_lingual_examples if cross_lingual_examples is not None else []
        self.base_system_message = (
            "You are a helpful multilingual customer support assistant for an e-commerce platform."
            "Your goal is to understand and respond to customer queries in their language, "
            "even if the query mixes languages or requires cross-lingual understanding."
            "Leverage the provided examples to inform your responses."
        )

    def add_example(self, query, response):
        self.cross_lingual_examples.append({'query': query, 'response': response})

    def _format_examples(self):
        formatted_string = ""
        for i, example in enumerate(self.cross_lingual_examples):
            formatted_string += f"\nExample {i+1} Query: {example['query']}"
            formatted_string += f"\nExample {i+1} Response: {example['response']}"
        return formatted_string

    def create_prompt(self, user_query, target_language_hint=None):
        prompt_parts = [self.base_system_message]

        if self.cross_lingual_examples:
            prompt_parts.append("\n\nHere are some cross-lingual in-context examples:")
            prompt_parts.append(self._format_examples())

        prompt_parts.append(f"\n\nCustomer Query: {user_query}")
        
        if target_language_hint:
            prompt_parts.append(f"Please respond in {target_language_hint} based on the query and examples.")
        else:
            prompt_parts.append("Please respond appropriately based on the query and examples.")

        return "".join(prompt_parts)


class MultilingualLLMService:
    def __init__(self, model_name="mock-multilingual-llm", api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("LLM_API_KEY")

    def generate_response(self, prompt):
        print(f"\n[DEBUG] LLM received prompt (first 200 chars): {prompt[:200]}...")

        if "order status" in prompt.lower() or "estado de mi pedido" in prompt.lower():
            return "Your order is currently being processed and is expected to ship within 2 business days. Su pedido está siendo procesado y se espera que se envíe dentro de 2 días hábiles."
        elif "return product" in prompt.lower() or "devolver un producto" in prompt.lower():
            return "To return a product, please visit our returns portal at example.com/returns. Para devolver un producto, visite nuestro portal de devoluciones en example.com/returns."
        elif "change size" in prompt.lower() or "cambiar mi talla" in prompt.lower():
            return "You can exchange for a different size through our exchange policy at example.com/exchanges. Puede cambiar por una talla diferente a través de nuestra política de cambios en example.com/exchanges."
        elif "cancel" in prompt.lower() or "anular" in prompt.lower():
            return "Your request to cancel has been noted. Please allow 24 hours for processing. Su solicitud de cancelación ha sido anotada. Por favor, espere 24 horas para su procesamiento."
        elif "refund" in prompt.lower() or "reembolso" in prompt.lower():
            return "Refunds typically take 5-7 business days to process. Los reembolsos suelen tardar de 5 a 7 días hábiles en procesarse."
        else:
            return "Thank you for contacting support. How else can I assist you today? Gracias por contactar a soporte. ¿En qué más puedo ayudarle hoy?"


class ECommerceChatbot:
    def __init__(self, llm_model_name="mock-multilingual-llm", llm_api_key=None):
        initial_icl_examples = [
            {'query': 'Where is my order (es)?', 'response': 'Su pedido está en camino. Rastréalo aquí: [Enlace de seguimiento]'},
            {'query': 'I want to return a product (quiero devolver un producto).', 'response': 'Please visit our returns page: [Returns Link] for instructions on how to return your item.'},
            {'query': '¿Cómo puedo cambiar mi talla? (How can I change my size?)', 'response': 'You can exchange your item by visiting our exchanges page: [Exchanges Link] and following the steps.'},
            {'query': 'My item is damaged (mi artículo está dañado).', 'response': 'We apologize for the inconvenience. Please contact our live support for immediate assistance or visit our warranty page.'}
        ]

        self.prompt_manager = InCLTPromptManager(initial_icl_examples)
        self.llm_service = MultilingualLLMService(model_name=llm_model_name, api_key=llm_api_key)
        print("E-commerce Chatbot initialized. Type 'quit' to exit.")

    def chat(self):
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                print("Chatbot: Goodbye!")
                break

            target_lang_hint = self._detect_language_hint(user_input)

            full_prompt = self.prompt_manager.create_prompt(user_input, target_lang_hint)
            
            bot_response = self.llm_service.generate_response(full_prompt)

            print(f"Chatbot: {bot_response}")

    def _detect_language_hint(self, text):
        text_lower = text.lower()
        if "hola" in text_lower or "pedido" in text_lower or "devolver" in text_lower or "español" in text_lower:
            return "Spanish"
        elif "bonjour" in text_lower or "commande" in text_lower or "français" in text_lower:
            return "French"
        return "English"

if __name__ == "__main__":
    chatbot = ECommerceChatbot()
    chatbot.chat()