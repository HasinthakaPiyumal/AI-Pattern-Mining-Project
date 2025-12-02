import random

class DummyLLMClient:
    def generate(self, prompt: str) -> str:
        if "price" in prompt.lower() and "headphone" in prompt.lower() and "spanish" in prompt.lower():
            return "El precio de los auriculares es de 50 euros."
        elif "delivery" in prompt.lower() and "time" in prompt.lower() and "french" in prompt.lower():
            return "Le délai de livraison est généralement de 3 à 5 jours ouvrables."
        elif "return" in prompt.lower() and "policy" in prompt.lower() and "german" in prompt.lower():
            return "Unsere Rückgaberichtlinien besagen, dass Sie Artikel innerhalb von 30 Tagen zurücksenden können."
        elif "thank you" in prompt.lower() or "gracias" in prompt.lower() or "merci" in prompt.lower() or "danke" in prompt.lower():
            return "You're welcome! How else can I assist you?"
        return f"[Dummy LLM Response for: {prompt[:50]}...]"

class MultilingualChatbot:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client if llm_client else DummyLLMClient()
        self.in_context_examples = self._load_in_context_examples()

    def _load_in_context_examples(self) -> list:
        return [
            {
                "source_lang_query": "What is the price of the headphones?",
                "source_lang": "en",
                "target_lang_query": "¿Cuál es el precio de los auriculares?",
                "target_lang": "es",
                "target_lang_response": "El precio de los auriculares es de 50 euros."
            },
            {
                "source_lang_query": "How long does delivery take?",
                "source_lang": "en",
                "target_lang_query": "Quel est le délai de livraison ?",
                "target_lang": "fr",
                "target_lang_response": "Le délai de livraison est généralement de 3 à 5 jours ouvrables."
            },
            {
                "source_lang_query": "What is your return policy?",
                "source_lang": "en",
                "target_lang_query": "Wie sind Ihre Rückgaberichtlinien?",
                "target_lang": "de",
                "target_lang_response": "Unsere Rückgaberichtlinien besagen, dass Sie Artikel innerhalb von 30 Tagen zurücksenden können."
            },
            {
                "source_lang_query": "Can I track my order?",
                "source_lang": "en",
                "target_lang_query": "¿Puedo rastrear mi pedido?",
                "target_lang": "es",
                "target_lang_response": "Sí, puedes rastrear tu pedido usando el número de seguimiento que te enviamos por correo electrónico."
            },
            {
                "source_lang_query": "Where can I find my order details?",
                "source_lang": "en",
                "target_lang_query": "Où puis-je trouver les détails de ma commande ?",
                "target_lang": "fr",
                "target_lang_response": "Vous pouvez trouver les détails de votre commande dans votre compte, sous la section 'Mes commandes'."
            }
        ]

    def _get_relevant_examples(self, user_query: str, target_language: str, num_examples: int = 2) -> list:
        # For simplicity, we'll pick examples that match the target language if available,
        # and then random ones if not enough, or just random if no language match.
        relevant_by_lang = [ex for ex in self.in_context_examples if ex["target_lang"] == target_language]
        
        if len(relevant_by_lang) >= num_examples:
            return random.sample(relevant_by_lang, num_examples)
        else:
            # Fill up with other examples if not enough language-specific ones
            other_examples = [ex for ex in self.in_context_examples if ex["target_lang"] != target_language]
            combined_examples = relevant_by_lang + random.sample(other_examples, min(num_examples - len(relevant_by_lang), len(other_examples)))
            return combined_examples

    def _construct_in_clt_prompt(self, user_query: str, target_language: str, in_context_examples: list) -> str:
        prompt_parts = ["You are a multilingual customer support chatbot."]
        prompt_parts.append(f"Please respond to the user's query in {target_language}.")
        prompt_parts.append("Here are some examples of cross-lingual queries and their appropriate responses:")

        for i, example in enumerate(in_context_examples):
            prompt_parts.append(f"\n--- Example {i+1} ---")
            prompt_parts.append(f"Source Language ({example['source_lang'].upper()}) Query: {example['source_lang_query']}")
            prompt_parts.append(f"Target Language ({example['target_lang'].upper()}) Query: {example['target_lang_query']}")
            prompt_parts.append(f"Target Language ({example['target_lang'].upper()}) Response: {example['target_lang_response']}")
        
        prompt_parts.append(f"\n--- User Query ---")
        prompt_parts.append(f"User Query (Target Language {target_language.upper()}): {user_query}")
        prompt_parts.append(f"Expected Response (Target Language {target_language.upper()}):")
        
        return "\n".join(prompt_parts)

    def _call_llm(self, prompt: str) -> str:
        return self.llm_client.generate(prompt)

    def respond(self, user_query: str, target_language: str) -> str:
        relevant_examples = self._get_relevant_examples(user_query, target_language)
        prompt = self._construct_in_clt_prompt(user_query, target_language, relevant_examples)
        response = self._call_llm(prompt)
        return response

if __name__ == "__main__":
    chatbot = MultilingualChatbot()

    print("\n--- Spanish Query ---")
    spanish_query = "¿Cuál es el precio de los auriculares?"
    spanish_response = chatbot.respond(spanish_query, "es")
    print(f"User: {spanish_query}")
    print(f"Chatbot: {spanish_response}")

    print("\n--- French Query ---")
    french_query = "Quel est le délai de livraison ?"
    french_response = chatbot.respond(french_query, "fr")
    print(f"User: {french_query}")
    print(f"Chatbot: {french_response}")

    print("\n--- German Query ---")
    german_query = "Wie sind Ihre Rückgaberichtlinien?"
    german_response = chatbot.respond(german_query, "de")
    print(f"User: {german_query}")
    print(f"Chatbot: {german_response}")

    print("\n--- Another Spanish Query (less direct match in examples) ---")
    another_spanish_query = "Necesito saber sobre el envío de mi pedido."
    another_spanish_response = chatbot.respond(another_spanish_query, "es")
    print(f"User: {another_spanish_query}")
    print(f"Chatbot: {another_spanish_response}")

    print("\n--- New Language (Italian) Query ---")
    italian_query = "Qual è il tempo di consegna?"
    italian_response = chatbot.respond(italian_query, "it")
    print(f"User: {italian_query}")
    print(f"Chatbot: {italian_response}")