
class MultilingualChatbot:
    def __init__(self, model=None):
        # In a real scenario, you'd load a multilingual LLM here
        # self.model = model or transformers.AutoModelForCausalLM.from_pretrained("path/to/multilingual/llm")
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained("path/to/multilingual/llm")
        print("Multilingual Chatbot initialized (using simulated LLM).")
        self.knowledge_base = {
            "shipping_tracking": "You can track your order using the tracking number provided in your shipping confirmation email. Please visit our website's 'Track Order' section.",
            "return_policy": "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition. Please see our FAQs for more details.",
            "payment_methods": "We accept major credit cards (Visa, MasterCard, Amex) and PayPal. We do not currently accept bank transfers.",
            "product_warranty": "All our electronics come with a 1-year manufacturer's warranty. For specific product warranty details, please check the product page.",
            "customer_service_contact": "You can contact customer service via live chat on our website, email at support@ecom.com, or call us at 1-800-123-4567."
        }
        # Simplified mapping for demonstration
        self.query_to_kb_key = {
            "track my order": "shipping_tracking",
            "how to return": "return_policy",
            "what payment methods": "payment_methods",
            "warranty on products": "product_warranty",
            "contact support": "customer_service_contact",
            "rastrear mi pedido": "shipping_tracking",
            "cómo devolver": "return_policy",
            "métodos de pago": "payment_methods",
            "garantía de productos": "product_warranty",
            "contacto de soporte": "customer_service_contact"
        }

    def _get_kb_answer(self, query_in_target_lang):
        """Simulates fetching an answer from a knowledge base, assumed to be in the target language (e.g., English)."""
        query_in_target_lang_lower = query_in_target_lang.lower()
        for key_phrase, kb_key in self.query_to_kb_key.items():
            if key_phrase in query_in_target_lang_lower:
                return self.knowledge_base.get(kb_key, "I'm sorry, I don't have information on that topic in our knowledge base.")
        return "I'm sorry, I couldn't find a direct answer to your question."

    def generate_inclt_prompt(self, user_query: str, source_lang: str, target_lang: str, in_context_examples: list) -> str:
        """
        Generates a prompt leveraging the InCLT Crosslingual Transfer Prompting pattern.
        It includes examples that show the cross-lingual transfer from source to target and back.
        """
        prompt = f"You are a helpful multilingual customer support assistant. Answer the user's question accurately.\n\n"
        prompt += f"The user's original query is in {source_lang}. Translate it to {target_lang} to find the answer, then provide the response back in {source_lang}.\n\n"

        for ex in in_context_examples:
            prompt += f"Example {source_lang} Query: {ex['user_query_source_lang']}\n"
            prompt += f"Translation to {target_lang}: {ex['user_query_target_lang']}\n"
            prompt += f"Knowledge Base Answer in {target_lang}: {ex['kb_answer_target_lang']}\n"
            prompt += f"Response in {source_lang}: {ex['final_response_source_lang']}\n\n"

        prompt += f"Now, answer the following query:\n"
        prompt += f"{source_lang} Query: {user_query}\n"
        prompt += f"Translation to {target_lang}: " # LLM is expected to fill this
        return prompt

    def simulate_llm_response(self, prompt: str, user_query: str, source_lang: str, target_lang: str) -> str:
        """
        Simulates an LLM's response based on the InCLT prompt.
        In a real LLM, the model would complete the prompt. Here, we'll extract
        the expected translation and use our simple KB, then "translate" back.
        """
        print(f"\n--- DEBUG: Generated Prompt ---\n{prompt}\n------------------------------\n")

        # Simulate translation of the user's query by the LLM
        # For this simulation, we'll hardcode some translations or use a very basic lookup
        translation_map = {
            "rastrear mi pedido": "track my order",
            "cómo devolver un producto": "how to return a product",
            "cuáles son los métodos de pago": "what are the payment methods",
            "cuál es la garantía de los productos": "what is the product warranty",
            "quiero contactar a soporte": "I want to contact support",
            "qual o status do meu pedido": "what is the status of my order" # Portuguese example
        }
        # Try to find a direct translation, otherwise, a simple placeholder
        translated_query = translation_map.get(user_query.lower(), f"TRANSLATED: {user_query} (assuming LLM translates)")
        print(f"Simulated LLM Translation to {target_lang}: {translated_query}")

        # Simulate getting an answer from the knowledge base (which is in target_lang)
        kb_answer = self._get_kb_answer(translated_query)
        print(f"Simulated KB Answer in {target_lang}: {kb_answer}")

        # Simulate translating the KB answer back to the source language
        # This is a simplification; a real LLM would generate the response directly.
        # Here we're just reversing the process or using a placeholder.
        # For a real system, you might use a translation API or the LLM itself to translate.
        if "track my order" in translated_query.lower():
            response_map = {
                "shipping_tracking": {
                    "es": "Puede rastrear su pedido utilizando el número de seguimiento proporcionado en su correo electrónico de confirmación de envío. Visite la sección 'Rastrear Pedido' de nuestro sitio web.",
                    "pt": "Você pode rastrear seu pedido usando o número de rastreamento fornecido em seu e-mail de confirmação de envio. Por favor, visite a seção 'Rastrear Pedido' do nosso site."
                }
            }
            final_response = response_map.get(self.query_to_kb_key.get(translated_query.lower(), ""), {}).get(source_lang.lower(), f"Please check your shipping confirmation email for tracking information (translated to {source_lang}).")
        elif "return policy" in translated_query.lower():
            final_response = f"Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra, siempre que el artículo esté en su estado original. Consulte nuestras preguntas frecuentes para obtener más detalles (translated to {source_lang})." if source_lang.lower() == "es" else f"Our return policy allows returns within 30 days of purchase, provided the item is in its original condition (translated to {source_lang})."
        elif "payment methods" in translated_query.lower():
            final_response = f"Aceptamos las principales tarjetas de crédito (Visa, MasterCard, Amex) y PayPal. Actualmente no aceptamos transferencias bancarias (translated to {source_lang})." if source_lang.lower() == "es" else f"We accept major credit cards and PayPal. (translated to {source_lang})."
        elif "product warranty" in translated_query.lower():
             final_response = f"Todos nuestros productos electrónicos tienen una garantía del fabricante de 1 año. Para detalles específicos de la garantía del producto, consulte la página del producto (translated to {source_lang})." if source_lang.lower() == "es" else f"All our electronics come with a 1-year manufacturer's warranty. (translated to {source_lang})."
        elif "customer service contact" in translated_query.lower():
             final_response = f"Puede contactar al servicio de atención al cliente a través de chat en vivo en nuestro sitio web, correo electrónico a support@ecom.com, o llamarnos al 1-800-123-4567 (translated to {source_lang})." if source_lang.lower() == "es" else f"You can contact customer service via live chat or email. (translated to {source_lang})."
        else:
            final_response = f"I'm sorry, I cannot fulfill your request in {source_lang} based on the available information. (Simulated LLM Response)."


        return final_response

# Main demonstration
if __name__ == "__main__":
    chatbot = MultilingualChatbot()

    # Define in-context examples for InCLT
    # These examples demonstrate the cross-lingual transfer (e.g., Spanish to English to Spanish)
    in_context_examples = [
        {
            "user_query_source_lang": "Cómo puedo rastrear mi pedido?",
            "user_query_target_lang": "How can I track my order?",
            "kb_answer_target_lang": "You can track your order using the tracking number provided in your shipping confirmation email. Please visit our website's 'Track Order' section.",
            "final_response_source_lang": "Puede rastrear su pedido utilizando el número de seguimiento que se le proporcionó en el correo electrónico de confirmación de envío. Visite la sección 'Rastrear Pedido' de nuestro sitio web."
        },
        {
            "user_query_source_lang": "Cuál es la política de devoluciones?",
            "user_query_target_lang": "What is the return policy?",
            "kb_answer_target_lang": "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition. Please see our FAQs for more details.",
            "final_response_source_lang": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra, siempre que el artículo esté en su estado original. Consulte nuestras preguntas frecuentes para obtener más detalles."
        }
    ]

    print("--- Multilingual Customer Support Chatbot Demo (InCLT) ---")

    # Scenario 1: Spanish query, English KB
    user_query_es = "Quiero saber el estado de mi envío."
    source_lang_es = "Spanish"
    target_lang_en = "English"

    print(f"\nUser ({source_lang_es}): {user_query_es}")
    inclt_prompt_es = chatbot.generate_inclt_prompt(user_query_es, source_lang_es, target_lang_en, in_context_examples)
    response_es = chatbot.simulate_llm_response(inclt_prompt_es, user_query_es, source_lang_es, target_lang_en)
    print(f"Chatbot ({source_lang_es}): {response_es}")

    # Scenario 2: Portuguese query, English KB
    user_query_pt = "Qual o status do meu pedido?"
    source_lang_pt = "Portuguese"
    target_lang_en = "English"

    print(f"\nUser ({source_lang_pt}): {user_query_pt}")
    # Using the same examples, but the prompt structure helps the LLM
    inclt_prompt_pt = chatbot.generate_inclt_prompt(user_query_pt, source_lang_pt, target_lang_en, in_context_examples)
    response_pt = chatbot.simulate_llm_response(inclt_prompt_pt, user_query_pt, source_lang_pt, target_lang_en)
    print(f"Chatbot ({source_lang_pt}): {response_pt}")

    # Scenario 3: Another Spanish query (less direct match for KB sim)
    user_query_es_2 = "¿Qué métodos de pago aceptan?"
    print(f"\nUser ({source_lang_es}): {user_query_es_2}")
    inclt_prompt_es_2 = chatbot.generate_inclt_prompt(user_query_es_2, source_lang_es, target_lang_en, in_context_examples)
    response_es_2 = chatbot.simulate_llm_response(inclt_prompt_es_2, user_query_es_2, source_lang_es, target_lang_en)
    print(f"Chatbot ({source_lang_es}): {response_es_2}")
