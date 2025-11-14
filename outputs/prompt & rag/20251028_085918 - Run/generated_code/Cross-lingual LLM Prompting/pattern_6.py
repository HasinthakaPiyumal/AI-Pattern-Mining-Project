from transformers import pipeline
from langdetect import detect, DetectorFactory

# Ensure consistent language detection results
DetectorFactory.seed = 0

class MultilingualCustomerSupportChatbot:
    def __init__(self, in_context_examples: list):
        """
        Initializes the MultilingualCustomerSupportChatbot.

        Args:
            in_context_examples (list): A list of dictionaries, each containing
                                        'source_lang', 'user_query', and 'llm_response_en'.
        """
        print("Loading text2text-generation pipeline with google/flan-t5-small...")
        self.llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
        self.in_context_examples = in_context_examples
        self.target_lang = "en"
        print("Chatbot initialized.")

    def _get_relevant_examples(self, user_query_lang: str) -> str:
        """
        Filters in-context examples based on the user query's language
        and formats them into a string for the prompt.

        Args:
            user_query_lang (str): The detected language of the user's query.

        Returns:
            str: A formatted string of relevant in-context examples.
        """
        relevant_examples = [ex for ex in self.in_context_examples if ex["source_lang"] == user_query_lang]
        
        if not relevant_examples:
            # If no direct match, try to find examples in a common language (e.g., English) if available
            # or simply return an empty string to avoid breaking the prompt
            print(f"No direct in-context examples found for language: {user_query_lang}. Trying general examples.")
            relevant_examples = [ex for ex in self.in_context_examples if ex["source_lang"] == self.target_lang]
            if not relevant_examples:
                return ""

        formatted_examples = []
        for i, example in enumerate(relevant_examples):
            formatted_examples.append(
                f"Example {i+1} (Source: {example['source_lang']}, Target: {self.target_lang}):\n"
                f"User: {example['user_query']}\n"
                f"Chatbot: {example['llm_response_en']}"
            )
        return "\n\n" + "\n\n".join(formatted_examples) if formatted_examples else ""

    def generate_response(self, user_query: str) -> str:
        """
        Generates a cross-lingual response to the user's query using in-context learning.

        Args:
            user_query (str): The user's input query.

        Returns:
            str: The chatbot's response in the target language (English).
        """
        try:
            user_query_lang = detect(user_query)
            print(f"Detected user query language: {user_query_lang}")
        except Exception as e:
            user_query_lang = self.target_lang # Default to target language if detection fails
            print(f"Language detection failed: {e}. Defaulting to target language: {user_query_lang}")

        relevant_examples_str = self._get_relevant_examples(user_query_lang)
        
        # Construct the prompt with the InCLT Crosslingual Transfer Prompting pattern
        prompt = (
            f"You are a multilingual customer support chatbot. Your goal is to understand questions in various languages "
            f"and respond in {self.target_lang}. Use the following examples to guide your response and cross-lingual understanding.\n"
            f"{relevant_examples_str}\n\n"
            f"Now, respond to the following user query in {self.target_lang}:\n"
            f"User: {user_query}\n"
            f"Chatbot:"
        )

        print(f"Generated Prompt:\n---\n{prompt}\n---")
        
        # Generate response using the LLM
        # The max_new_tokens is set to prevent overly long or irrelevant generations
        response = self.llm_pipeline(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)[0]["generated_text"]
        
        # Clean up the response if the LLM generates the prompt back or similar artifacts
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response.strip()

# --- Example Usage ---
if __name__ == "__main__":
    # Define in-context examples following the InCLT pattern
    # These examples demonstrate how to respond in English regardless of the source query language.
    in_context_data = [
        {
            "source_lang": "fr",
            "user_query": "J'ai un problème avec ma commande numéro 12345.",
            "llm_response_en": "I understand you have an issue with your order number 12345. Could you please provide more details about the problem?"
        },
        {
            "source_lang": "es",
            "user_query": "Mi producto llegó dañado. ¿Qué puedo hacer?",
            "llm_response_en": "I'm sorry to hear that your product arrived damaged. Please provide your order number so I can assist you with a replacement or refund."
        },
        {
            "source_lang": "de",
            "user_query": "Wie kann ich mein Abonnement kündigen?",
            "llm_response_en": "To cancel your subscription, please go to your account settings and follow the instructions under 'Manage Subscription'."
        },
        {
            "source_lang": "fr",
            "user_query": "Quel est le statut de ma livraison ?",
            "llm_response_en": "Please provide your tracking number or order ID so I can check the status of your delivery."
        },
        {
            "source_lang": "en",
            "user_query": "I need help with my account settings.",
            "llm_response_en": "Sure, I can help you with your account settings. What specific issue are you encountering?"
        },
    ]

    chatbot = MultilingualCustomerSupportChatbot(in_context_data)

    print("\n--- Testing Chatbot ---")

    # Test with a French query
    french_query = "J'ai besoin d'aide pour mon mot de passe oublié."
    print(f"\nUser (FR): {french_query}")
    response_fr = chatbot.generate_response(french_query)
    print(f"Chatbot (EN): {response_fr}")

    # Test with a Spanish query
    spanish_query = "¿Cómo puedo cambiar mi dirección de envío?"
    print(f"\nUser (ES): {spanish_query}")
    response_es = chatbot.generate_response(spanish_query)
    print(f"Chatbot (EN): {response_es}")

    # Test with a German query (should still work due to cross-lingual transfer, even if no direct example for 'de' specific topic)
    german_query = "Mein Paket ist noch nicht angekommen."
    print(f"\nUser (DE): {german_query}")
    response_de = chatbot.generate_response(german_query)
    print(f"Chatbot (EN): {response_de}")

    # Test with an English query
    english_query = "Where can I find my order history?"
    print(f"\nUser (EN): {english_query}")
    response_en = chatbot.generate_response(english_query)
    print(f"Chatbot (EN): {response_en}")

    # Test with an unsupported language (e.g., Italian) - should still try to respond in English
    italian_query = "Vorrei sapere lo stato del mio ordine."
    print(f"\nUser (IT): {italian_query}")
    response_it = chatbot.generate_response(italian_query)
    print(f"Chatbot (EN): {response_it}")
