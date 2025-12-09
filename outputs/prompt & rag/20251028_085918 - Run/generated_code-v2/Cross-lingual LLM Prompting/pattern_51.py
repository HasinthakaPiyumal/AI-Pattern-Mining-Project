
class MultilingualChatbot:
    def __init__(self, llm_client, in_context_examples=None):
        """
        Initializes the MultilingualChatbot with an LLM client and in-context examples.

        Args:
            llm_client: An object that can interact with a multilingual LLM (e.g., OpenAI API client).
                        It should have a 'generate' method that takes a prompt string and returns a response string.
            in_context_examples (list): A list of dictionaries, where each dictionary represents an example.
                                        Each example should have:
                                        - 'source_lang_query': User query in the source language.
                                        - 'source_lang_response': Assistant response in the source language.
                                        - 'target_lang_query': User query in the target language.
                                        - 'target_lang_response': Assistant response in the target language.
                                        Example: [
                                            {
                                                "source_lang_query": "What is your return policy?",
                                                "source_lang_response": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
                                                "target_lang_query": "¿Cuál es su política de devolución?",
                                                "target_lang_response": "Nuestra política de devolución permite devoluciones dentro de los 30 días posteriores a la compra con un recibo válido."
                                            }
                                        ]
        """
        self.llm_client = llm_client
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.system_instruction = (
            "You are a helpful and polite multilingual customer support assistant. "
            "Please provide concise and accurate answers in the language of the user's query."
        )

    def _construct_in_clt_prompt(self, user_query: str, user_query_language_name: str) -> str:
        """
        Constructs the prompt using InCLT Crosslingual Transfer Prompting.
        It includes system instructions, in-context examples in both source and target languages,
        and the current user query.

        Args:
            user_query (str): The current query from the user.
            user_query_language_name (str): The name of the language of the user's query (e.g., "English", "Spanish").

        Returns:
            str: The fully constructed prompt string.
        """
        prompt_parts = [f"System: {self.system_instruction}\n"]

        for example in self.in_context_examples:
            prompt_parts.append(f"User ({example['source_lang_language_name']}): {example['source_lang_query']}")
            prompt_parts.append(f"Assistant ({example['source_lang_language_name']}): {example['source_lang_response']}\n")
            prompt_parts.append(f"User ({example['target_lang_language_name']}): {example['target_lang_query']}")
            prompt_parts.append(f"Assistant ({example['target_lang_language_name']}): {example['target_lang_response']}\n")

        prompt_parts.append(f"User ({user_query_language_name}): {user_query}")
        prompt_parts.append(f"Assistant ({user_query_language_name}):") # LLM will complete this

        return "\n".join(prompt_parts)

    def ask(self, user_query: str, user_query_language_name: str) -> str:
        """
        Sends a user query to the LLM and returns the response, utilizing InCLT prompting.

        Args:
            user_query (str): The customer's question.
            user_query_language_name (str): The name of the language the user's query is in.

        Returns:
            str: The LLM's response to the query.
        """
        if not self.llm_client:
            raise ValueError("LLM client is not initialized. Cannot send query.")

        prompt = self._construct_in_clt_prompt(user_query, user_query_language_name)
        print(f"--- Generated Prompt ---\n{prompt}\n------------------------") # For debugging
        response = self.llm_client.generate(prompt)
        return response.strip()


# --- Example Usage (requires a mock or actual LLM client) ---

# Mock LLM Client for demonstration purposes
class MockLLMClient:
    def generate(self, prompt: str) -> str:
        print(f"[MockLLMClient received prompt]:\n{prompt[:200]}...")
        # Simulate LLM response based on common customer support queries
        if "return policy" in prompt.lower() or "política de devolución" in prompt.lower():
            return "Our return policy allows returns within 30 days of purchase with a valid receipt. (This is a mock response)"
        elif "track my order" in prompt.lower() or "rastrear mi pedido" in prompt.lower():
            return "You can track your order using the tracking number provided in your shipping confirmation email. (This is a mock response)"
        elif "hello" in prompt.lower() or "hola" in prompt.lower():
            return "Hello! How can I assist you today? (Mock response)"
        else:
            return "I'm sorry, I don't have information on that topic. Please ask another question. (Mock response)"


if __name__ == "__main__":
    # Initialize mock LLM client
    mock_llm = MockLLMClient()

    # Define In-Context Learning examples with both source and target languages
    # (assuming English as source, Spanish as target for these examples)
    icl_examples = [
        {
            "source_lang_language_name": "English",
            "source_lang_query": "What is your return policy?",
            "source_lang_response": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
            "target_lang_language_name": "Spanish",
            "target_lang_query": "¿Cuál es su política de devolución?",
            "target_lang_response": "Nuestra política de devolución permite devoluciones dentro de los 30 días posteriores a la compra con un recibo válido."
        },
        {
            "source_lang_language_name": "English",
            "source_lang_query": "How can I track my order?",
            "source_lang_response": "You can track your order using the tracking number provided in your shipping confirmation email.",
            "target_lang_language_name": "Spanish",
            "target_lang_query": "¿Cómo puedo rastrear mi pedido?",
            "target_lang_response": "Puede rastrear su pedido utilizando el número de seguimiento proporcionado en el correo electrónico de confirmación de envío."
        }
    ]

    # Initialize the chatbot with the mock LLM and examples
    chatbot = MultilingualChatbot(llm_client=mock_llm, in_context_examples=icl_examples)

    print("\n--- Testing Chatbot ---\n")

    # Test query in English
    print("Customer (English): Hello, I have a question about my order.")
    response_en = chatbot.ask("Hello, I have a question about my order.", "English")
    print(f"Chatbot (English): {response_en}\n")

    # Test query in Spanish
    print("Customer (Spanish): Necesito saber sobre mi política de devoluciones.")
    response_es = chatbot.ask("Necesito saber sobre mi política de devoluciones.", "Spanish")
    print(f"Chatbot (Spanish): {response_es}\n")

    # Test query in English (related to example)
    print("Customer (English): I want to know your return policy.")
    response_en_icl = chatbot.ask("I want to know your return policy.", "English")
    print(f"Chatbot (English): {response_en_icl}\n")

    # Test query in an unlisted language (LLM should still try to respond based on its general multilingual capabilities)
    # Note: For robust cross-lingual behavior, the LLM itself must support the language.
    # The InCLT pattern specifically helps with transfer, assuming underlying LLM capability.
    print("Customer (French): Bonjour, quelle est votre politique de retour?")
    response_fr = chatbot.ask("Bonjour, quelle est votre politique de retour?", "French")
    print(f"Chatbot (French): {response_fr}\n")
