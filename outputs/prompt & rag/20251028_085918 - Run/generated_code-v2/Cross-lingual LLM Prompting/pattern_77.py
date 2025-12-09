import os
from transformers import pipeline

# Mock API key for demonstration. In a real application, use environment variables.
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Placeholder for a real LLM client. Using a mock for demonstration.
class MockLLM:
    def __init__(self):
        pass

    def generate(self, prompt: str) -> str:
        # Simulate LLM behavior. In a real scenario, this would call an actual LLM API.
        print("\n--- LLM Prompt Sent ---")
        print(prompt)
        print("-----------------------\n")
        if "cómo puedo" in prompt.lower() or "ayuda con mi pedido" in prompt.lower():
            return "Claro, para ayudarte con tu pedido, por favor proporciónanos el número de seguimiento o el ID del pedido."  # Spanish response
        elif "delivery issue" in prompt.lower():
            return "We apologize for the delivery issue. Please provide your order number so we can investigate." # English response
        else:
            return "Lo siento, no entendí tu pregunta. ¿Podrías reformularla?" # Default Spanish response

class MultilingualCustomerSupportChatbot:
    def __init__(self, source_lang='en', target_lang='es'):
        self.source_lang = source_lang
        self.target_lang = target_lang

        # Initialize translation pipelines
        # Using Helsinki-NLP models for demonstration. Ensure they are downloaded or available.
        try:
            self.translator_en_to_es = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}")
            self.translator_es_to_en = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{target_lang}-{source_lang}")
        except Exception as e:
            print(f"Warning: Could not load translation models. Please ensure 'Helsinki-NLP/opus-mt-{{source_lang}}-{{target_lang}}' and 'Helsinki-NLP/opus-mt-{{target_lang}}-{{source_lang}}' are available. Error: {e}")
            print("Proceeding with a mock translator.")
            self.translator_en_to_es = self._mock_translator_en_to_es
            self.translator_es_to_en = self._mock_translator_es_to_en

        self.llm = MockLLM() # Using the mock LLM

        # Mock Knowledge Base for In-Context Learning examples
        # In a real system, this would be a vector store with semantic search.
        self.knowledge_base = [
            {
                "source_query": "My package is delayed. What should I do?",
                "source_resolution": "Please provide your order number and we will check the status for you.",
                "target_query": "Mi paquete está retrasado. ¿Qué debo hacer?",
                "target_resolution": "Por favor, proporcione su número de pedido y verificaremos el estado por usted."
            },
            {
                "source_query": "How can I return an item?",
                "source_resolution": "You can initiate a return from your order history page on our website.",
                "target_query": "¿Cómo puedo devolver un artículo?",
                "target_resolution": "Puede iniciar una devolución desde la página de historial de pedidos en nuestro sitio web."
            },
            {
                "source_query": "I received a damaged product.",
                "source_resolution": "We apologize for the inconvenience. Please send us a photo of the damaged item and your order number.",
                "target_query": "Recibí un producto dañado.",
                "target_resolution": "Lamentamos el inconveniente. Por favor, envíenos una foto del artículo dañado y su número de pedido."
            }
        ]

    def _mock_translator_en_to_es(self, text):
        # Simple mock translation for demonstration
        translations = {
            "My package is delayed. What should I do?": "Mi paquete está retrasado. ¿Qué debo hacer?",
            "How can I return an item?": "¿Cómo puedo devolver un artículo?",
            "I received a damaged product.": "Recibí un producto dañado.",
            "I need help with my order": "Necesito ayuda con mi pedido",
            "What is the status of my refund?": "¿Cuál es el estado de mi reembolso?",
            "where is my order": "dónde está mi pedido",
            "delivery issue": "problema de entrega"
        }
        return [{"translation_text": translations.get(text, f"[Mock ES] {text}")}]

    def _mock_translator_es_to_en(self, text):
        # Simple mock translation for demonstration
        translations = {
            "Mi paquete está retrasado. ¿Qué debo hacer?": "My package is delayed. What should I do?",
            "¿Cómo puedo devolver un artículo?": "How can I return an item?",
            "Recibí un producto dañado.": "I received a damaged product.",
            "Necesito ayuda con mi pedido": "I need help with my order",
            "¿Cuál es el estado de mi reembolso?": "What is the status of my refund?",
            "dónde está mi pedido": "where is my order",
            "problema de entrega": "delivery issue"
        }
        return [{"translation_text": translations.get(text, f"[Mock EN] {text}")}]

    def _translate_text(self, text: str, src_lang: str, dest_lang: str) -> str:
        if src_lang == self.source_lang and dest_lang == self.target_lang:
            return self.translator_en_to_es(text)[0]['translation_text']
        elif src_lang == self.target_lang and dest_lang == self.source_lang:
            return self.translator_es_to_en(text)[0]['translation_text']
        else:
            # Fallback for other cases or direct pass-through if languages are same
            return text

    def _get_icl_examples(self, num_examples=2) -> list:
        # In a real application, this would involve semantic search in a vector store.
        # For demonstration, we just return the first 'num_examples' from our mock KB.
        return self.knowledge_base[:num_examples]

    def _construct_prompt(self, user_query_target_lang: str, icl_examples: list) -> str:
        prompt_parts = []

        prompt_parts.append(
            f"You are a helpful customer support assistant for an e-commerce platform. "
            f"Your goal is to assist customers with their queries in {self.target_lang} (Spanish). "
            f"Below are some examples of past customer interactions in both {self.source_lang} (English) and {self.target_lang} (Spanish), "
            f"followed by the current customer query. Use these examples to provide an accurate "
            f"and helpful response in {self.target_lang} (Spanish) for the current query.\n\n"
        )

        # Add In-Context Learning examples (source and target languages)
        for i, example in enumerate(icl_examples):
            prompt_parts.append(f"Example {i+1}:\n")
            prompt_parts.append(f"  English Query: {example['source_query']}\n")
            prompt_parts.append(f"  English Resolution: {example['source_resolution']}\n")
            prompt_parts.append(f"  Spanish Query: {example['target_query']}\n")
            prompt_parts.append(f"  Spanish Resolution: {example['target_resolution']}\n\n")

        # Translate the current user query to the source language for cross-lingual transfer
        user_query_source_lang = self._translate_text(user_query_target_lang, self.target_lang, self.source_lang)

        prompt_parts.append(f"Current Customer Query:\n")
        prompt_parts.append(f"  English (Translated): {user_query_source_lang}\n")
        prompt_parts.append(f"  Spanish (Original): {user_query_target_lang}\n\n")
        prompt_parts.append(f"Please provide the helpful response in {self.target_lang} (Spanish) for the current query:\n")
        prompt_parts.append(f"Response: ")

        return "".join(prompt_parts)

    def generate_response(self, user_query: str) -> str:
        # 1. Get ICL examples (simulated retrieval)
        icl_examples = self._get_icl_examples(num_examples=2)

        # 2. Construct the prompt using InCLT pattern
        prompt = self._construct_prompt(user_query, icl_examples)

        # 3. Send prompt to LLM and get response
        llm_response = self.llm.generate(prompt)

        return llm_response

# --- Demonstration --- F
if __name__ == "__main__":
    chatbot = MultilingualCustomerSupportChatbot()

    print("\n--- Chatbot Initialized ---")
    print(f"Source Language: {chatbot.source_lang}")
    print(f"Target Language: {chatbot.target_lang}\n")

    # Example 1: User query in Spanish, expecting Spanish response
    spanish_query_1 = "Necesito ayuda con mi pedido"
    print(f"User (ES): {spanish_query_1}")
    response_1 = chatbot.generate_response(spanish_query_1)
    print(f"Chatbot (ES): {response_1}\n")

    # Example 2: Another user query in Spanish
    spanish_query_2 = "Mi paquete nunca llegó"
    print(f"User (ES): {spanish_query_2}")
    response_2 = chatbot.generate_response(spanish_query_2)
    print(f"Chatbot (ES): {response_2}\n")

    # Example 3: User query in Spanish related to a damaged product (closer to an ICL example)
    spanish_query_3 = "Recibí un producto dañado. ¿Qué hago?"
    print(f"User (ES): {spanish_query_3}")
    response_3 = chatbot.generate_response(spanish_query_3)
    print(f"Chatbot (ES): {response_3}\n")

    # Example 4: A query that is translated but might not have direct ICL matches leading to generic response
    spanish_query_4 = "¿Cuál es el estado de mi reembolso?"
    print(f"User (ES): {spanish_query_4}")
    response_4 = chatbot.generate_response(spanish_query_4)
    print(f"Chatbot (ES): {response_4}\n")

    # Example 5: Another query in Spanish
    spanish_query_5 = "Problema de entrega"
    print(f"User (ES): {spanish_query_5}")
    response_5 = chatbot.generate_response(spanish_query_5)
    print(f"Chatbot (ES): {response_5}\n")





