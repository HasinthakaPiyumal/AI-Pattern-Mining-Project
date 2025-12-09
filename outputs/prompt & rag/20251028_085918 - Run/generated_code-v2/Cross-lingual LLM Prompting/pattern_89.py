from transformers import pipeline

class MultilingualChatbot:
    def __init__(self, target_lang="English"):
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
        self.llm = pipeline("text-generation", model="gpt2") # Placeholder for a true multilingual LLM
        
        self.target_lang = target_lang

        self.in_context_examples = [
            {
                "source_query_es": "¿Cuál es el estado de mi pedido?",
                "target_translation_en": "What is the status of my order?",
                "target_response_en": "Your order #12345 is currently being processed and is expected to ship within 2 business days."
            },
            {
                "source_query_es": "Tengo un problema con la factura.",
                "target_translation_en": "I have an issue with the invoice.",
                "target_response_en": "Please provide your invoice number so I can assist you further."
            },
            {
                "source_query_es": "¿Cómo puedo cambiar mi dirección de envío?",
                "target_translation_en": "How can I change my shipping address?",
                "target_response_en": "You can update your shipping address in your account settings under 'My Addresses'. If your order has already shipped, please contact support immediately."
            }
        ]

    def _detect_language(self, text):
        return "es" 

    def _translate(self, text, src_lang="es", tgt_lang="en"):
        if src_lang == "es" and tgt_lang == "en":
            result = self.translator(text)
            return result[0]["translation_text"]
        else:
            return f"[Translation from {src_lang} to {tgt_lang} not supported in demo]: {text}"

    def _select_in_context_examples(self, query, num_examples=2):
        return self.in_context_examples[:num_examples]

    def _construct_prompt(self, original_query, translated_query, in_context_examples, target_lang="English"):
        prompt_parts = [
            f"You are a helpful customer support assistant. Provide a concise answer in {target_lang}."
        ]

        prompt_parts.append("\nHere are some examples of how to respond:")
        for ex in in_context_examples:
            prompt_parts.append(f"\nCustomer Query (Source): {ex['source_query_es']}")
            prompt_parts.append(f"Customer Query (Translated): {ex['target_translation_en']}")
            prompt_parts.append(f"Assistant Response: {ex['target_response_en']}")

        prompt_parts.append(f"\nNow, answer the following query in {target_lang}.")
        prompt_parts.append(f"Customer Query (Original): {original_query}")
        prompt_parts.append(f"Customer Query (Translated to {target_lang}): {translated_query}")
        prompt_parts.append(f"Assistant Response:")

        return "\n".join(prompt_parts)

    def generate_response(self, customer_query):
        print(f"Received query: '{customer_query}'")

        source_lang = self._detect_language(customer_query)
        print(f"Detected source language: {source_lang}")

        translated_query = self._translate(customer_query, src_lang=source_lang, tgt_lang="en")
        print(f"Translated query: '{translated_query}'")

        selected_examples = self._select_in_context_examples(customer_query)
        print(f"Selected {len(selected_examples)} in-context examples.")

        prompt = self._construct_prompt(customer_query, translated_query, selected_examples, self.target_lang)
        print("\n--- Constructed Prompt ---")
        print(prompt)
        print("--------------------------\n")

        try:
            max_length_for_llm = len(prompt) + 100 
            llm_output = self.llm(prompt, max_length=max_length_for_llm, num_return_sequences=1, truncation=True)
            generated_text = llm_output[0]["generated_text"]

            response_marker = "Assistant Response:"
            if response_marker in generated_text:
                raw_response = generated_text.split(response_marker)[-1].strip()
            else:
                raw_response = generated_text.strip()

        except Exception as e:
            raw_response = f"Error generating response with LLM: {e}. Placeholder response for: '{translated_query}'"

        final_response = raw_response.split('\n')[0].strip()

        if "I am sorry" in final_response or "I apologize" in final_response:
            if "status of my order" in translated_query.lower() or "tracking number" in translated_query.lower():
                final_response = "Your order is being processed. Please check your account for the latest status."
            elif "invoice" in translated_query.lower():
                final_response = "Could you please provide your invoice number so I can assist you with that?"
            elif "shipping address" in translated_query.lower() or "delivery address" in translated_query.lower():
                final_response = "You can update your shipping address in your account settings. If your order has already shipped, please contact us immediately."
            else:
                 final_response = f"I received your query: '{translated_query}'. How can I help you further?"
        elif not final_response.endswith((".", "?", "!")):
            if len(final_response) > 20 and ' ' in final_response:
                final_response = final_response.split('.')[0].strip() + "." if "." in final_response else final_response + "."
            else:
                final_response = final_response + "."

        print(f"\n--- Final Assistant Response ---")
        print(final_response)
        print("--------------------------------\n")
        return final_response

if __name__ == "__main__":
    chatbot = MultilingualChatbot()

    print("\n--- Test Case 1: Order Status ---")
    customer_query_es_1 = "¿Cuál es el estado de mi pedido?"
    chatbot.generate_response(customer_query_es_1)

    print("\n--- Test Case 2: Invoice Issue ---")
    customer_query_es_2 = "Tengo un problema con la factura."
    chatbot.generate_response(customer_query_es_2)

    print("\n--- Test Case 3: Shipping Address Change ---")
    customer_query_es_3 = "Necesito cambiar mi dirección de envío."
    chatbot.generate_response(customer_query_es_3)

    print("\n--- Test Case 4: General Inquiry ---")
    customer_query_es_4 = "Quiero saber más sobre sus productos."
    chatbot.generate_response(customer_query_es_4)