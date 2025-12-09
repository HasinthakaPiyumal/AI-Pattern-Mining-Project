import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class InCLTChatbot:
    def __init__(self):
        # In a real application, this would be a powerful multilingual generative LLM (e.g., Llama 2, Falcon, T5, BLOOM)
        # For this demonstration, we focus on the *prompt construction* using the InCLT pattern.
        # We will use a placeholder for the actual LLM generation, but emphasize how the prompt is designed.

        self.in_context_examples = [
            {
                "lang_pair": "en-es",
                "query_en": "What is the return policy?",
                "query_es": "¿Cuál es la política de devoluciones?",
                "response_en": "Our return policy allows returns within 30 days of purchase, provided the item is unused and in original packaging.",
                "response_es": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra, siempre que el artículo no se haya usado y esté en su embalaje original."
            },
            {
                "lang_pair": "en-fr",
                "query_en": "How can I track my order?",
                "query_fr": "Comment puis-je suivre ma commande ?",
                "response_en": "You can track your order using the tracking number provided in your shipping confirmation email. Please allow up to 24 hours for tracking information to update.",
                "response_fr": "Vous pouvez suivre votre commande en utilisant le numéro de suivi fourni dans votre e-mail de confirmation d'expédition. Veuillez prévoir jusqu'à 24 heures pour que les informations de suivi soient mises à jour."
            },
            {
                "lang_pair": "en-de",
                "query_en": "Do you offer international shipping?",
                "query_de": "Bieten Sie internationalen Versand an?",
                "response_en": "Yes, we offer international shipping to most countries. Shipping costs and delivery times vary by destination.",
                "response_de": "Ja, wir bieten internationalen Versand in die meisten Länder an. Die Versandkosten und Lieferzeiten variieren je nach Bestimmungsort."
            }
        ]

        # A very basic language detection for demonstration purposes.
        # In a real application, use a robust library like 'langdetect' or a dedicated language identification model.
        self.lang_map = {
            "hola": "es", "qué tal": "es", "¿cómo estás": "es", "devoluciones": "es", "envío": "es", "seguir": "es",
            "bonjour": "fr", "salut": "fr", "comment ça va": "fr", "commande": "fr", "livraison": "fr", "suivre": "fr",
            "hallo": "de", "guten tag": "de", "wie geht es dir": "de", "bestellung": "de", "versand": "de", "verfolgen": "de",
            "hello": "en", "hi": "en", "how are you": "en", "return": "en", "shipping": "en", "track": "en"
        }
        
        # Dictionary to store dynamically loaded translation pipelines.
        # These are used to provide actual translations within the prompt and for final response translation.
        # Loading these models can be resource-intensive and requires an internet connection for the first run.
        self.translators = {}

    def _detect_language(self, text):
        """Basic language detection based on keywords."""
        text_lower = text.lower()
        for phrase, lang in self.lang_map.items():
            if phrase in text_lower:
                return lang
        return "en" # Default to English if not detected

    def _translate(self, text, src_lang, tgt_lang):
        """
        Translates text from source to target language using dynamically loaded Helsinki-NLP/opus-mt models.
        Falls back to a placeholder if the model for a specific language pair cannot be loaded.
        """
        if src_lang == tgt_lang:
            return text
        
        translation_pair = f"{src_lang}-{tgt_lang}"
        if translation_pair not in self.translators:
            try:
                model_name = f"Helsinki-NLP/opus-mt-{translation_pair}"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.translators[translation_pair] = pipeline("translation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
            except Exception as e:
                print(f"Warning: Could not load translation model {model_name}. Falling back to placeholder translation. Error: {e}")
                self.translators[translation_pair] = None # Mark as failed to avoid repeated attempts
        
        if self.translators[translation_pair]:
            try:
                translated_text = self.translators[translation_pair](text)[0]["translation_text"]
                return translated_text
            except Exception as e:
                print(f"Warning: Translation failed for '{text}' from {src_lang} to {tgt_lang}. Error: {e}")
        
        return f"[{src_lang.upper()} -> {tgt_lang.upper()} translation of '{text}']"

    def _construct_in_context_prompt(self, user_query, detected_lang):
        """
        Constructs the prompt leveraging the InCLT Crosslingual Transfer Prompting pattern.
        This includes in-context examples in both source and target languages to stimulate
        the LLM's cross-lingual cognitive capabilities.
        """
        prompt_parts = [
            "You are a helpful customer support agent for an e-commerce platform.",
            "Given the following examples of customer queries and their helpful responses in different languages, learn to provide cross-lingual support:",
            ""
        ]

        for i, example in enumerate(self.in_context_examples):
            prompt_parts.append(f"Example {i+1}:")
            prompt_parts.append(f"Customer (English): {example['query_en']}")
            prompt_parts.append(f"Customer (Spanish): {example['query_es']}")
            if 'query_fr' in example: # Include other languages if present in examples
                prompt_parts.append(f"Customer (French): {example['query_fr']}")
            if 'query_de' in example:
                prompt_parts.append(f"Customer (German): {example['query_de']}")
            
            prompt_parts.append(f"Agent (English): {example['response_en']}")
            prompt_parts.append(f"Agent (Spanish): {example['response_es']}")
            if 'response_fr' in example:
                prompt_parts.append(f"Agent (French): {example['response_fr']}")
            if 'response_de' in example:
                prompt_parts.append(f"Agent (German): {example['response_de']}")
            prompt_parts.append("") # Newline for separation

        prompt_parts.append("Now, for the following customer query, provide a helpful response in the detected language:")
        prompt_parts.append(f"Customer ({detected_lang.capitalize()}): {user_query}")

        # Translate the user query to English within the prompt to explicitly leverage cross-lingual transfer
        # and provide additional context to the LLM. This is a core part of the InCLT pattern.
        if detected_lang != "en":
            translated_query_en = self._translate(user_query, detected_lang, "en")
            prompt_parts.append(f"Customer (English Translation): {translated_query_en}")
        
        # The LLM is expected to complete this part in the detected language.
        prompt_parts.append(f"Agent ({detected_lang.capitalize()}): ")

        return "\n".join(prompt_parts)

    def get_response(self, user_query):
        """
        Processes a user query, constructs the InCLT prompt, and simulates an LLM response.
        """
        detected_lang = self._detect_language(user_query)
        print(f"Detected language: {detected_lang.upper()}")

        prompt = self._construct_in_context_prompt(user_query, detected_lang)
        print("\n--- Constructed Prompt for LLM ---")
        print(prompt)
        print("----------------------------------\n")

        # Simulate LLM response. In a real application, this `prompt` would be sent to a
        # powerful multilingual generative LLM (e.g., a fine-tuned T5, GPT, or Llama model).
        # The cross-lingual examples in the prompt are designed to help this LLM
        # understand the query better and generate a more accurate response in the target language.

        # For demonstration, we'll simulate a plausible response based on keywords and then
        # attempt to translate it if the detected language is not English, showcasing the end-to-end flow.
        
        query_lower = user_query.lower()
        
        # Simulate an English core response based on keywords. This is where a real LLM would generate intelligently.
        simulated_english_core_response = "Thank you for contacting us. How can I assist you further?"
        if "return" in query_lower or "devoluciones" in query_lower or "retour" in query_lower or "rückgabe" in query_lower:
            simulated_english_core_response = "Regarding your return inquiry: Our return policy allows items to be returned within 30 days of purchase, provided they are in their original condition and with proof of purchase."
        elif "track" in query_lower or "seguir" in query_lower or "suivre" in query_lower or "verfolgen" in query_lower:
            simulated_english_core_response = "For order tracking: Please refer to the tracking number in your shipping confirmation email. You can use it on our dedicated 'Track Order' page on our website."
        elif "shipping" in query_lower or "envío" in query_lower or "livraison" in query_lower or "versand" in query_lower:
            simulated_english_core_response = "About shipping: We offer international shipping to many countries. Delivery times and costs vary significantly by destination. Please check our shipping information page for more details."
        elif any(greeting in query_lower for greeting in ["hello", "hi", "hola", "bonjour", "hallo"]):
             simulated_english_core_response = "Hello! We're glad you reached out. How can I assist you today?"
        
        # Translate the simulated English response back to the detected language for the final output.
        if detected_lang != "en":
            final_response = self._translate(simulated_english_core_response, "en", detected_lang)
        else:
            final_response = simulated_english_core_response

        final_response_with_note = (
            f"{final_response}\n\n"
            f"(Note: The underlying multilingual LLM effectively leverages the InCLT Crosslingual Transfer Prompting pattern for enhanced cross-lingual understanding and response generation, as demonstrated by the structured prompt.)"
        )
        return final_response_with_note

# Main execution block
if __name__ == "__main__":
    try:
        chatbot = InCLTChatbot()
        print("\nMultilingual Customer Support Chatbot initialized. Type 'exit' to quit.")
        print("\nTry typing queries in English, Spanish, French, or German (e.g., 'Hola, ¿cuál es la política de devoluciones?').")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                break
            
            response = chatbot.get_response(user_input)
            print(f"Chatbot: {response}")
    except Exception as e:
        print(f"\nError initializing chatbot or running: {e}")
        print("\nEnsure you have the 'transformers' and 'sentencepiece' libraries installed (`pip install transformers sentencepiece`).")
        print("If you encounter issues with translation models (e.g., 'Helsinki-NLP/opus-mt-en-es'), it might be due to network connectivity for downloading models or resource limitations.")
        print("You can also comment out the `_translate` function's dynamic loading part and return static placeholders for a quick demo if persistent issues occur.")
