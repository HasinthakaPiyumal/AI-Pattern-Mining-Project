
class ExampleManager:
    """
    Manages a collection of cross-lingual in-context examples.
    Each example contains a source query, its translation, and the desired response.
    """
    def __init__(self):
        # Storing examples as a list of dictionaries. Each dict has 'source_lang', 'source_query',
        # 'target_lang', 'target_query', and 'target_response'.
        self.examples = [
            {
                "source_lang": "en",
                "source_query": "Hello, how can I help you?",
                "target_lang": "fr",
                "target_query": "Bonjour, comment puis-je vous aider ?",
                "target_response": "Je suis intéressé par un produit en particulier."
            },
            {
                "source_lang": "en",
                "source_query": "What is your return policy?",
                "target_lang": "es",
                "target_query": "¿Cuál es su política de devoluciones?",
                "target_response": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra."
            },
            {
                "source_lang": "fr",
                "source_query": "J'ai un problème avec ma commande.",
                "target_lang": "en",
                "target_query": "I have an issue with my order.",
                "target_response": "Please provide your order number so I can assist you."
            },
            {
                "source_lang": "es",
                "source_query": "¿Cuándo llegará mi paquete?",
                "target_lang": "en",
                "target_query": "When will my package arrive?",
                "target_response": "Please provide your tracking number for an update."
            },
            {
                "source_lang": "en",
                "source_query": "I want to cancel my subscription.",
                "target_lang": "de",
                "target_query": "Ich möchte mein Abonnement kündigen.",
                "target_response": "Bitte bestätigen Sie Ihre E-Mail-Adresse, um die Kündigung zu bearbeiten."
            }
        ]

    def get_examples(self, num_examples=3, current_source_lang=None, current_target_lang=None):
        """
        Retrieves a specified number of examples, prioritizing those relevant to the current languages.
        For simplicity, this version just returns the first `num_examples`.
        A more advanced version would match based on source/target languages.
        """
        # In a real-world scenario, this would involve a more sophisticated selection
        # algorithm, potentially based on similarity to the current query or language pair.
        return self.examples[:num_examples]


class PromptBuilder:
    """
    Constructs prompts for a multilingual LLM using InCLT Crosslingual Transfer Prompting.
    It incorporates examples in both source and target languages into the prompt.
    """
    def __init__(self, example_manager: ExampleManager):
        self.example_manager = example_manager

    def build_prompt(self, user_query: str, query_lang: str, target_lang: str, num_examples: int = 3) -> str:
        """
        Builds a prompt that includes in-context examples for cross-lingual transfer.
        The examples provided are in both the source and target language of the query.

        Args:
            user_query (str): The user's input query.
            query_lang (str): The language of the user's query.
            target_lang (str): The desired language for the LLM's response.
            num_examples (int): The number of in-context examples to include.

        Returns:
            str: The constructed prompt for the LLM.
        """
        examples = self.example_manager.get_examples(num_examples, query_lang, target_lang)

        prompt_parts = [
            f"You are a multilingual customer support assistant. Your task is to respond to customer queries in {target_lang}."
            "Leverage cross-lingual knowledge from the provided examples to understand and respond accurately."
        ]

        if examples:
            prompt_parts.append("\nHere are some examples of customer interactions for cross-lingual understanding and response:\n")
            for i, ex in enumerate(examples):
                prompt_parts.append(f"Example {i + 1}:")
                prompt_parts.append(f"Source ({ex['source_lang']}): {ex['source_query']}")
                # This is the core of InCLT: providing the query in the target language as well in the example
                prompt_parts.append(f"Target ({ex['target_lang']}) Query: {ex['target_query']}")
                prompt_parts.append(f"Target ({ex['target_lang']}) Response: {ex['target_response']}\n")
        
        prompt_parts.append(f"\nCustomer Query ({query_lang}): {user_query}")
        prompt_parts.append(f"Assistant Response ({target_lang}):")

        return "\n".join(prompt_parts)
