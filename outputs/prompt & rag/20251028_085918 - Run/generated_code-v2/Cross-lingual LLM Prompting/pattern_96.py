class TranslationService:
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        # Simplified simulation of a translation service
        print(f"[TranslationService] Translating '{text}' from {source_lang} to {target_lang}")
        if source_lang == "es" and target_lang == "en":
            translations = {
                "¿Dónde está mi pedido?": "Where is my order?",
                "No funciona mi producto.": "My product is not working.",
                "Quiero devolver un artículo.": "I want to return an item."
            }
            return translations.get(text, f"Translated_en({text})")
        elif source_lang == "en" and target_lang == "es":
            translations = {
                "Where is my order?": "¿Dónde está mi pedido?",
                "My product is not working.": "No funciona mi producto.",
                "I want to return an item.": "Quiero devolver un artículo."
            }
            return translations.get(text, f"Translated_es({text})")
        return text # Return original if no specific translation rule

class MultilingualLLM:
    def __init__(self):
        # In a real scenario, this would initialize a pre-trained multilingual LLM
        pass

    def generate_response(self, prompt: str) -> str:
        # Simplified simulation of an LLM generating a response
        print("[MultilingualLLM] Processing prompt...")
        # Based on the prompt structure, try to simulate a relevant response
        if "Where is my order?" in prompt and "pedido" in prompt:
            return "Your order is currently being processed and is expected to arrive within 3-5 business days. Please check your order tracking for more details.\nSu pedido está siendo procesado y se espera que llegue en 3-5 días hábiles. Consulte el seguimiento de su pedido para obtener más detalles."
        elif "product is not working" in prompt and "no funciona mi producto" in prompt:
            return "We apologize for the inconvenience. Please describe the issue in more detail or visit our troubleshooting page for assistance.\nLamentamos el inconveniente. Por favor, describa el problema con más detalle o visite nuestra página de solución de problemas para obtener ayuda."
        elif "return an item" in prompt and "devolver un artículo" in prompt:
            return "To return an item, please visit our returns portal on our website and follow the instructions. You will need your order number.\nPara devolver un artículo, visite nuestro portal de devoluciones en nuestro sitio web y siga las instrucciones. Necesitará su número de pedido."
        return "I understand your query. Could you please provide more details?\nEntiendo su consulta. ¿Podría proporcionar más detalles?"

class CustomerSupportChatbot:
    def __init__(self):
        self.llm = MultilingualLLM()
        self.translation_service = TranslationService()
        self.in_context_examples = [
            {
                "source_lang": "es",
                "source_query": "¿Dónde está mi pedido?",
                "target_lang": "en",
                "target_query": "Where is my order?",
                "response": "Your order is currently being processed and is expected to arrive within 3-5 business days. Please check your order tracking for more details.\nSu pedido está siendo procesado y se espera que llegue en 3-5 días hábiles. Consulte el seguimiento de su pedido para obtener más detalles."
            },
            {
                "source_lang": "es",
                "source_query": "No funciona mi producto.",
                "target_lang": "en",
                "target_query": "My product is not working.",
                "response": "We apologize for the inconvenience. Please describe the issue in more detail or visit our troubleshooting page for assistance.\nLamentamos el inconveniente. Por favor, describa el problema con más detalle o visite nuestra página de solución de problemas para obtener ayuda."
            },
            {
                "source_lang": "en",
                "source_query": "I want to return an item.",
                "target_lang": "es",
                "target_query": "Quiero devolver un artículo.",
                "response": "To return an item, please visit our returns portal on our website and follow the instructions. You will need your order number.\nPara devolver un artículo, visite nuestro portal de devoluciones en nuestro sitio web y siga las instrucciones. Necesitará su número de pedido."
            }
        ]

    def _construct_inclt_prompt(self, user_query: str, query_lang: str, default_target_lang: str = "en") -> str:
        prompt_parts = []

        # Add cross-lingual in-context examples
        for example in self.in_context_examples:
            prompt_parts.append(f"Customer query ({example['source_lang']}): {example['source_query']}")
            prompt_parts.append(f"Customer query ({example['target_lang']}): {example['target_query']}")
            prompt_parts.append(f"Chatbot Response: {example['response']}\n")

        # Add the actual user query, translated if necessary for the prompt context
        # Here, we ensure the prompt includes both original and a common pivot language (e.g., English)
        if query_lang != default_target_lang:
            translated_query = self.translation_service.translate(user_query, query_lang, default_target_lang)
            prompt_parts.append(f"Customer query ({query_lang}): {user_query}")
            prompt_parts.append(f"Customer query ({default_target_lang}): {translated_query}")
        else:
            # If query is already in target language, we might still want a placeholder for cross-lingual thinking
            # For simplicity here, we just add the original.
            prompt_parts.append(f"Customer query ({query_lang}): {user_query}")

        prompt_parts.append(f"Chatbot Response:")

        return "\n".join(prompt_parts)

    def get_response(self, user_query: str, lang: str) -> str:
        print(f"\n[Chatbot] Received query in {lang}: '{user_query}'")
        
        # 1. Simulate Language Detection (pre-provided 'lang' argument)
        query_lang = lang

        # 2. Construct InCLT Prompt
        prompt = self._construct_inclt_prompt(user_query, query_lang)
        
        # 3. LLM Interaction
        llm_response = self.llm.generate_response(prompt)
        
        return llm_response

# --- Example Usage --- 
if __name__ == "__main__":
    chatbot = CustomerSupportChatbot()

    print("--- Spanish Query Example ---")
    spanish_query = "Quiero devolver un artículo."
    response = chatbot.get_response(spanish_query, "es")
    print(f"Chatbot Final Response: {response}")

    print("\n--- English Query Example ---")
    english_query = "Where is my order?"
    response = chatbot.get_response(english_query, "en")
    print(f"Chatbot Final Response: {response}")

    print("\n--- Another Spanish Query Example ---")
    another_spanish_query = "No funciona mi producto."
    response = chatbot.get_response(another_spanish_query, "es")
    print(f"Chatbot Final Response: {response}")

    print("\n--- Unrecognized Query Example (Spanish) ---")
    unrecognized_spanish_query = "Necesito ayuda con mi factura."
    response = chatbot.get_response(unrecognized_spanish_query, "es")
    print(f"Chatbot Final Response: {response}")