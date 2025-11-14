from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class MultilingualChatbot:
    def __init__(self, model_name="t5-small"): # Using t5-small for demonstration purposes
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Pre-defined in-context examples for different scenarios and language pairs
        # These examples demonstrate the InCLT Crosslingual Transfer Prompting pattern
        # by providing both source and target language exemplars.
        self_en = "You can manage your subscriptions in the 'Account Settings' section of our website."
        self_es = "Puede gestionar sus suscripciones en la sección 'Configuración de la cuenta' de nuestro sitio web."
        self_fr = "Vous pouvez gérer vos abonnements dans la section 'Paramètres du compte' de notre site web."

        self_en2 = "Our new pricing plans will be available starting next month."
        self_es2 = "Nuestros nuevos planes de precios estarán disponibles a partir del próximo mes."
        self_fr2 = "Nos nouveaux plans tarifaires seront disponibles à partir du mois prochain."

        self.in_context_examples = [
            {
                "user_en": "How do I manage my subscriptions?",
                "bot_en": self_en,
                "user_es": "¿Cómo gestiono mis suscripciones?",
                "bot_es": self_es,
                "user_fr": "Comment gérer mes abonnements?",
                "bot_fr": self_fr,
            },
            {
                "user_en": "When will the new pricing plans be released?",
                "bot_en": self_en2,
                "user_es": "¿Cuándo se lanzarán los nuevos planes de precios?",
                "bot_es": self_es2,
                "user_fr": "Quand les nouveaux plans tarifaires seront-ils publiés?",
                "bot_fr": self_fr2,
            },
             {
                "user_en": "I forgot my password, what should I do?",
                "bot_en": "Please visit the 'Forgot Password' link on the login page to reset it.",
                "user_es": "Olvidé mi contraseña, ¿qué debo hacer?",
                "bot_es": "Por favor, visite el enlace 'Olvidé mi contraseña' en la página de inicio de sesión para restablecerla.",
                "user_fr": "J'ai oublié mon mot de passe, que dois-je faire?",
                "bot_fr": "Veuillez visiter le lien 'Mot de passe oublié' sur la page de connexion pour le réinitialiser.",
            },
        ]

    def _get_in_context_prompt_part(self, target_lang: str) -> str:
        """Constructs the in-context learning part of the prompt using both source (English) and target languages."""
        prompt_parts = []
        for ex in self.in_context_examples:
            # Always include English (source language) as part of the cross-lingual transfer
            prompt_parts.append(f"User (EN): {ex['user_en']}\nBot (EN): {ex['bot_en']}")
            # Include the target language example
            prompt_parts.append(f"User ({target_lang.upper()}): {ex[f'user_{target_lang}']}\nBot ({target_lang.upper()}): {ex[f'bot_{target_lang}']}")
        return "\n\n".join(prompt_parts)

    def generate_response(self, user_query: str, query_lang: str, target_lang: str) -> str:
        """
        Generates a chatbot response using InCLT prompting, adapting to the query language
        and generating a response in the target language.

        Args:
            user_query (str): The user's query.
            query_lang (str): The language of the user's query (e.g., 'en', 'es', 'fr').
            target_lang (str): The desired language for the bot's response (e.g., 'en', 'es', 'fr').
        """
        # Construct the in-context learning part of the prompt
        in_context_part = self._get_in_context_prompt_part(target_lang)

        # Construct the full prompt, including instructions and the current query
        full_prompt = (
            f"You are a helpful multilingual customer support assistant.\n"
            f"Your goal is to provide concise and relevant answers in the specified target language.\n"
            f"Here are some examples of user queries and bot responses in English and {target_lang.upper()}:\n\n"
            f"{in_context_part}\n\n"
            f"Based on the examples, please provide a concise and helpful response to the following query.\n"
            f"User ({query_lang.upper()}): {user_query}\n"
            f"Bot ({target_lang.upper()}):"
        )

        print(f"--- PROMPT (for debugging) ---\n{full_prompt}\n--------------") # For debugging purposes

        input_ids = self.tokenizer(full_prompt, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

        # Generate response using the model
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=100,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Post-process the response to remove any leading prompt phrases the model might have regenerated
        expected_prefix = f"Bot ({target_lang.upper()}):"
        if expected_prefix in response:
            response = response.split(expected_prefix, 1)[-1].strip()
        # T5 models often echo the input prefix if it's the last token. Handle cases where it might only output the prefix.
        if response.startswith(f"User ({query_lang.upper()}):") or response.startswith(f"Bot ({target_lang.upper()}):"):
             # This means the model didn't generate much beyond the prompt structure, return empty or a generic error.
             return "I apologize, I could not generate a relevant response at this moment." # Or re-try generation
        return response

if __name__ == "__main__":
    chatbot = MultilingualChatbot()

    print("\n--- Multilingual Customer Support Chatbot Demo ---")
    print("Type your query. Specify the target language for the bot's response (en, es, fr).")
    print("Example: 'How do I manage my account? target_lang=es'")
    print("Type 'exit' to quit.\n")

    while True:
        user_input_raw = input("You (type query & target_lang, e.g., 'Hello target_lang=es'): ")
        if user_input_raw.lower() == 'exit':
            break

        parts = user_input_raw.split('target_lang=')
        if len(parts) < 2:
            print("Please specify target_lang, e.g., 'My query target_lang=es'")
            continue

        user_query = parts[0].strip()
        target_lang = parts[1].strip().lower()

        if target_lang not in ['en', 'es', 'fr']:
            print("Invalid target language. Please choose 'en', 'es', or 'fr'.")
            continue

        # For this demonstration, we assume the user's input language is English for simplicity.
        # In a real application, you'd use a language detection model (e.g., from `langdetect` or `fasttext`).
        query_lang = 'en'

        print(f"\nUser query: '{user_query}' (in {query_lang.upper()}), desired response in {target_lang.upper()}\n")

        try:
            response = chatbot.generate_response(user_query, query_lang, target_lang)
            print(f"Bot ({target_lang.upper()}): {response}\n")
        except Exception as e:
            print(f"An error occurred: {e}\n")
