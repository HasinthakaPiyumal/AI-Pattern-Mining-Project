import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class InCLTExampleGenerator:
    """
    Generates in-context learning examples with cross-lingual transfer.
    Stores example pairs of (source_lang_query, source_lang_response, target_lang_query, target_lang_response).
    """
    def __init__(self):
        # A simulated knowledge base of cross-lingual examples
        # Each entry contains a concept and its representation in different languages.
        # For simplicity, we assume we have these pre-translated or generated.
        self.examples = [
            {
                "concept": "order_status",
                "en_query": "What is the status of my order #ORD123?",
                "en_response": "Your order #ORD123 is currently in transit and expected to arrive by next Tuesday.",
                "es_query": "¿Cuál es el estado de mi pedido #ORD123?",
                "es_response": "Su pedido #ORD123 está actualmente en tránsito y se espera que llegue el próximo martes.",
                "de_query": "Wie ist der Status meiner Bestellung #ORD123?",
                "de_response": "Ihre Bestellung #ORD123 ist derzeit unterwegs und wird voraussichtlich nächste Woche Dienstag ankommen.",
            },
            {
                "concept": "return_policy",
                "en_query": "What is your return policy?",
                "en_response": "You can return items within 30 days of purchase with the original receipt.",
                "es_query": "¿Cuál es su política de devoluciones?",
                "es_response": "Puede devolver artículos dentro de los 30 días posteriores a la compra con el recibo original.",
                "de_query": "Was ist Ihre Rückgaberichtlinie?",
                "de_response": "Sie können Artikel innerhalb von 30 Tagen nach dem Kauf mit dem Originalbeleg zurücksenden.",
            },
            {
                "concept": "shipping_cost",
                "en_query": "How much does shipping cost?",
                "en_response": "Shipping costs vary based on your location and the weight of the items. Standard shipping is $5.99.",
                "es_query": "¿Cuánto cuesta el envío?",
                "es_response": "Los costos de envío varían según su ubicación y el peso de los artículos. El envío estándar cuesta $5.99.",
                "de_query": "Wie viel kostet der Versand?",
                "de_response": "Die Versandkosten variieren je nach Standort und Gewicht der Artikel. Der Standardversand kostet 5,99 $.",
            },
        ]

    def get_relevant_icl_examples(self, target_lang_query: str, target_lang_code: str, source_lang_code: str = "en", num_examples: int = 2) -> list:
        """
        Selects relevant cross-lingual examples based on a simulated intent matching.
        In a real system, this would involve embedding and similarity search.
        For this demo, we'll pick examples that roughly match keywords or are just top N.
        """
        relevant_examples = []
        # Simple keyword matching for demo purposes
        query_lower = target_lang_query.lower()

        if "order" in query_lower or "pedido" in query_lower or "bestellung" in query_lower:
            relevant_examples.append(next(item for item in self.examples if item["concept"] == "order_status"))
        if "return" in query_lower or "devolucion" in query_lower or "rückgabe" in query_lower:
            relevant_examples.append(next(item for item in self.examples if item["concept"] == "return_policy"))
        if "shipping" in query_lower or "envio" in query_lower or "versand" in query_lower:
            relevant_examples.append(next(item for item in self.examples if item["concept"] == "shipping_cost"))
        
        # If no specific examples match, just take the first 'num_examples' or all if less than num_examples
        if not relevant_examples:
            relevant_examples = self.examples[:num_examples]
        elif len(relevant_examples) > num_examples:
            relevant_examples = relevant_examples[:num_examples]
            
        return relevant_examples

class MultilingualChatbot:
    """
    A multilingual customer support chatbot leveraging InCLT Crosslingual Transfer Prompting.
    """
    def __init__(self, model_name: str = "facebook/mbart-large-50-many-to-many-mmt"):
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.example_generator = InCLTExampleGenerator()
        
        # Supported language codes for MBart-50
        self.lang_codes = {
            "en": "en_XX",
            "es": "es_XX",
            "de": "de_DE",
            "fr": "fr_XX",
        }

    def _construct_icl_prompt(self, user_query: str, target_lang_code: str, source_lang_code: str = "en") -> str:
        """
        Constructs the InCLT prompt with interleaved source and target language examples.
        """
        prompt_parts = ["You are a helpful customer support assistant for an e-commerce company."]
        prompt_parts.append("Here are some examples of customer interactions and responses:")

        icl_examples = self.example_generator.get_relevant_icl_examples(user_query, target_lang_code, source_lang_code)

        for ex in icl_examples:
            # Source language example
            prompt_parts.append(f"User ({source_lang_code.upper()}): {ex[f'{source_lang_code}_query']}")
            prompt_parts.append(f"Assistant ({source_lang_code.upper()}): {ex[f'{source_lang_code}_response']}")
            
            # Target language example
            prompt_parts.append(f"User ({target_lang_code.upper()}): {ex[f'{target_lang_code}_query']}")
            prompt_parts.append(f"Assistant ({target_lang_code.upper()}): {ex[f'{target_lang_code}_response']}")

        prompt_parts.append(f"Now, please respond to the following customer query in {target_lang_code.upper()}:")
        prompt_parts.append(f"User ({target_lang_code.upper()}): {user_query}")
        prompt_parts.append(f"Assistant ({target_lang_code.upper()}): ")

        return "\n".join(prompt_parts)

    def respond_to_query(self, user_query: str, target_lang: str) -> str:
        """
        Generates a response to a user query in the specified target language
        using InCLT cross-lingual transfer prompting.
        """
        if target_lang not in self.lang_codes:
            return f"Sorry, I currently do not support {target_lang}. Please choose from English (en), Spanish (es), German (de), or French (fr)."
        
        target_lang_code = self.lang_codes[target_lang]
        source_lang_code = self.lang_codes["en"] # Default source language is English

        prompt = self._construct_icl_prompt(user_query, target_lang, source_lang_code)
        
        self.tokenizer.src_lang = source_lang_code # Set source language for tokenization
        encoded_input = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        
        # Set target language for generation
        # The `decoder_start_token_id` is set to the token ID of the target language.
        # This guides the model to generate in the specified language.
        generated_tokens = self.model.generate(
            **encoded_input,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang_code],
            max_new_tokens=200, 
            num_beams=5, 
            early_stopping=True
        )
        
        response = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        # The model might generate the prompt context again, so we try to extract only the assistant's part
        assistant_prefix = f"Assistant ({target_lang.upper()}): "
        if assistant_prefix in response:
            response = response.split(assistant_prefix, 1)[1].strip()
            
        return response

if __name__ == "__main__":
    print("Initializing Multilingual Customer Support Chatbot...")
    print("Loading facebook/mbart-large-50-many-to-many-mmt. This may take a moment.")
    chatbot = MultilingualChatbot()
    print("Chatbot initialized. Type 'exit' to quit.")
    print("Supported languages: en (English), es (Spanish), de (German), fr (French)")

    while True:
        user_input = input("\nEnter your query (e.g., 'es: ¿Cuál es mi pedido #ORD123?'): ")
        if user_input.lower() == "exit":
            break

        if ":" not in user_input:
            print("Please specify language, e.g., 'en: What is my order status?' or 'es: ¿Cuál es mi pedido?'.")
            continue

        lang_prefix, query_text = user_input.split(':', 1)
        target_language = lang_prefix.strip().lower()
        query_text = query_text.strip()

        if not query_text:
            print("Query cannot be empty.")
            continue

        response = chatbot.respond_to_query(query_text, target_language)
        print(f"\nChatbot ({target_language.upper()}): {response}")
