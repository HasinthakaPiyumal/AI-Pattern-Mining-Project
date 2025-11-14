
class GlobalCareAIAssistant:
    """
    A simulated AI Assistant for multinational e-commerce customer support,
    leveraging InCLT Crosslingual Transfer Prompting.
    """

    def __init__(self, cross_lingual_examples: list):
        """
        Initializes the GlobalCareAIAssistant with cross-lingual examples.

        Args:
            cross_lingual_examples (list): A list of dictionaries, each containing
                                           'source_language', 'target_language',
                                           'source_text', and 'target_text'.
        """
        self.cross_lingual_examples = cross_lingual_examples

    def _construct_in_context_prompt(self, customer_query: str, customer_lang: str, target_lang: str) -> str:
        """
        Constructs an in-context prompt for the simulated LLM using cross-lingual examples.

        Args:
            customer_query (str): The customer's original query.
            customer_lang (str): The language of the customer's query.
            target_lang (str): The desired language for the AI's response.

        Returns:
            str: The formatted prompt including relevant cross-lingual examples.
        """
        prompt_parts = []
        relevant_examples = []

        # Prioritize examples matching customer_lang and target_lang
        for example in self.cross_lingual_examples:
            if (example["source_language"] == customer_lang and
                    example["target_language"] == target_lang):
                relevant_examples.append(example)
            elif (example["source_language"] == target_lang and
                  example["target_language"] == customer_lang):
                # Also consider reverse pairs if they help transfer
                relevant_examples.append(example)

        # If no specific examples, just pick a few general ones
        if not relevant_examples and self.cross_lingual_examples:
            relevant_examples = self.cross_lingual_examples[:2] # Take first 2 as general examples

        # Format the selected examples into the prompt
        if relevant_examples:
            prompt_parts.append("Here are some examples of queries and their cross-lingual responses:")
            for i, example in enumerate(relevant_examples):
                prompt_parts.append(f"Example {i+1} ({example['source_language']} to {example['target_language']}):")
                prompt_parts.append(f"Source Query: {example['source_text']}")
                prompt_parts.append(f"Target Response: {example['target_text']}")
                prompt_parts.append("") # Add a blank line for separation

        prompt_parts.append(f"Now, please respond to the following query in {target_lang}:")
        prompt_parts.append(f"Customer Query ({customer_lang}): {customer_query}")
        prompt_parts.append(f"AI Assistant Response ({target_lang}):")

        return "\n".join(prompt_parts)

    def simulate_llm_response(self, prompt: str) -> str:
        """
        Simulates an LLM's response based on the constructed prompt.
        In a real application, this would involve calling an actual LLM API.
        For this simulation, it will provide a placeholder or a very basic 'translation'.

        Args:
            prompt (str): The prompt generated for the LLM.

        Returns:
            str: A simulated response from the LLM.
        """
        # Very basic simulation: if the query asks about shipping, provide a shipping answer.
        # In a real scenario, the LLM would interpret the prompt and examples to generate a relevant response.
        if "shipping" in prompt.lower() or "delivery" in prompt.lower():
            return "We offer standard and express shipping options. Delivery times vary by region."
        elif "return policy" in prompt.lower() or "refund" in prompt.lower():
            return "Our return policy allows returns within 30 days of purchase with a valid receipt."
        elif "product" in prompt.lower() and "information" in prompt.lower():
            return "Please provide the product name or ID for more detailed information."
        else:
            # A generic fallback, possibly indicating a translation task
            # This highly simplified simulation assumes the LLM would translate and respond generally.
            # A real LLM would be much more sophisticated.
            return f"[Simulated LLM response based on context for: {prompt.split('Customer Query')[1].split(')')[1].strip()}]"


    def process_query(self, customer_query: str, customer_lang: str, target_lang: str = "English") -> str:
        """
        Processes a customer query by constructing a cross-lingual prompt and simulating an LLM response.

        Args:
            customer_query (str): The customer's query text.
            customer_lang (str): The language of the customer's query.
            target_lang (str, optional): The language for the AI's response. Defaults to "English".

        Returns:
            str: The simulated AI assistant's response in the target language.
        """
        prompt = self._construct_in_context_prompt(customer_query, customer_lang, target_lang)
        print("\n--- Constructed Prompt ---")
        print(prompt)
        print("--------------------------")
        response = self.simulate_llm_response(prompt)
        return response


# --- Example Usage ---
if __name__ == "__main__":
    # Define some cross-lingual examples for the AI Assistant
    # In a real system, these would be carefully curated and potentially many more.
    cross_lingual_examples_data = [
        {
            "source_language": "English",
            "target_language": "Spanish",
            "source_text": "How can I track my order?",
            "target_text": "¿Cómo puedo rastrear mi pedido?"
        },
        {
            "source_language": "Spanish",
            "target_language": "English",
            "source_text": "Mi paquete está retrasado.",
            "target_text": "My package is delayed."
        },
        {
            "source_language": "German",
            "target_language": "English",
            "source_text": "Wie ist die Rückgaberichtlinie?",
            "target_text": "What is the return policy?"
        },
        {
            "source_language": "English",
            "target_language": "French",
            "source_text": "I need help with my account settings.",
            "target_text": "J'ai besoin d'aide avec les paramètres de mon compte."
        },
        {
            "source_language": "French",
            "target_language": "German",
            "source_text": "Où est mon colis?",
            "target_text": "Wo ist mein Paket?"
        }
    ]

    # Initialize the GlobalCare AI Assistant
    ai_assistant = GlobalCareAIAssistant(cross_lingual_examples=cross_lingual_examples_data)

    print("\n--- Scenario 1: Spanish Customer Query, English Response ---")
    customer_query_spanish = "¿Cuál es su política de envío internacional?"
    response_spanish = ai_assistant.process_query(customer_query_spanish, "Spanish", "English")
    print(f"AI Assistant Final Response: {response_spanish}")

    print("\n--- Scenario 2: German Customer Query, English Response (using specific example) ---")
    customer_query_german = "Kann ich diesen Artikel zurückgeben?"
    response_german = ai_assistant.process_query(customer_query_german, "German", "English")
    print(f"AI Assistant Final Response: {response_german}")

    print("\n--- Scenario 3: English Customer Query, French Response ---")
    customer_query_english = "I have a question about my recent order."
    response_english = ai_assistant.process_query(customer_query_english, "English", "French")
    print(f"AI Assistant Final Response: {response_english}")

    print("\n--- Scenario 4: New Language (Italian) Customer Query, English Response (general fallback) ---")
    customer_query_italian = "Il mio ordine è stato annullato, perché?"
    response_italian = ai_assistant.process_query(customer_query_italian, "Italian", "English")
    print(f"AI Assistant Final Response: {response_italian}")

    print("\n--- Scenario 5: Spanish Customer Query (about shipping), English Response ---")
    customer_query_spanish_shipping = "Necesito información sobre el envío de mi producto."
    response_spanish_shipping = ai_assistant.process_query(customer_query_spanish_shipping, "Spanish", "English")
    print(f"AI Assistant Final Response: {response_spanish_shipping}")
