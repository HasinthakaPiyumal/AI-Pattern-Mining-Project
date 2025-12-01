class MultilingualChatbot:
    def __init__(self, in_context_examples):
        """
        Initializes the chatbot with in-context learning examples.

        Args:
            in_context_examples (list): A list of dictionaries, each containing
                                        source and target language queries and resolutions.
                                        Example:
                                        [
                                            {
                                                "source_lang_query": "How do I reset my password?",
                                                "source_lang_resolution": "You can reset your password by going to "
                                                                        "the settings page and clicking 'Forgot Password'.",
                                                "target_lang_query": "¿Cómo restablezco mi contraseña?",
                                                "target_lang_resolution": "Puede restablecer su contraseña yendo a la "
                                                                        "página de configuración y haciendo clic en "
                                                                        "'Olvidé mi contraseña'.",
                                                "language": "Spanish"
                                            }
                                        ]
        """
        self.in_context_examples = in_context_examples
        # In a real application, you would initialize an actual LLM client here
        # e.g., self.llm_client = openai.OpenAI()
        print("MultilingualChatbot initialized with InCLT examples.")

    def _mock_llm_call(self, prompt):
        """
        Mocks an LLM API call. In a real scenario, this would interact with an actual LLM.
        For demonstration, it just returns a generic response based on the prompt.
        """
        print(f"\n--- Mock LLM received prompt (truncated for display) ---\n{prompt[:500]}...\n--- End of Mock LLM Prompt ---\n")
        # Simple heuristic to simulate a response based on the last query
        lines = prompt.strip().split("\n")
        last_query_line = None
        for i in reversed(range(len(lines))):
            if lines[i].strip().startswith("User (") and lines[i].strip().endswith("):"):
                last_query_line = lines[i]
                break

        if last_query_line:
            language = last_query_line.split("User (")[1].split("):")[0]
            query_text = last_query_line.split("):")[1].strip()
            return f"Chatbot ({language}): Understood your query: '{query_text}'. " \
                   f"Based on our knowledge base and the provided examples, " \
                   f"I can assist you further."
        return "Chatbot: I received your query. How can I help?"


    def _construct_icl_prompt(self, customer_query: str, target_language: str) -> str:
        """
        Constructs the prompt using InCLT (In-Context Learning Transfer) methodology.
        This includes examples in both source and target languages.

        Args:
            customer_query (str): The customer's query in the target language.
            target_language (str): The language of the customer's query (e.g., "Spanish").

        Returns:
            str: The fully constructed prompt for the LLM.
        """
        prompt_parts = [
            "You are a helpful and multilingual customer support chatbot.",
            "Your goal is to provide accurate and relevant answers in the user's language.",
            "Here are some examples of past customer interactions and their resolutions, "
            "presented in both English (source) and the respective target language. "
            "Use these to understand the context and provide a good response:"
        ]

        # Add in-context examples
        for example in self.in_context_examples:
            # Add Source Language Example
            prompt_parts.append(f"User (English): {example['source_lang_query']}")
            prompt_parts.append(f"Chatbot (English): {example['source_lang_resolution']}")
            
            # Add Target Language Example
            prompt_parts.append(f"User ({example['language']}): {example['target_lang_query']}")
            prompt_parts.append(f"Chatbot ({example['language']}): {example['target_lang_resolution']}")
            prompt_parts.append("-" * 30) # Separator for clarity

        # Add the current customer's query
        prompt_parts.append(f"Now, please respond to the following customer query:")
        prompt_parts.append(f"User ({target_language}): {customer_query}")
        prompt_parts.append(f"Chatbot ({target_language}):") # Prompt the LLM to generate its response

        return "\n".join(prompt_parts)

    def get_response(self, customer_query: str, target_language: str) -> str:
        """
        Gets a response from the chatbot for a given customer query.

        Args:
            customer_query (str): The customer's query in the target language.
            target_language (str): The language of the customer's query.

        Returns:
            str: The chatbot's response.
        """
        prompt = self._construct_icl_prompt(customer_query, target_language)
        response = self._mock_llm_call(prompt) # In a real app, call self.llm_client.chat.completions.create(...)
        return response

# --- Example Usage ---
icl_examples = [
    {
        "source_lang_query": "How do I reset my password?",
        "source_lang_resolution": "You can reset your password by going to the settings page and clicking 'Forgot Password'.",
        "target_lang_query": "¿Cómo restablezco mi contraseña?",
        "target_lang_resolution": "Puede restablecer su contraseña yendo a la página de configuración y haciendo clic en 'Olvidé mi contraseña'.",
        "language": "Spanish"
    },
    {
        "source_lang_query": "My order is delayed. What should I do?",
        "source_lang_resolution": "Please provide your order number, and I will check the status for you.",
        "target_lang_query": "Mi pedido está retrasado. ¿Qué debo hacer?",
        "target_lang_resolution": "Por favor, proporcione su número de pedido y verificaré el estado por usted.",
        "language": "Spanish"
    },
    {
        "source_lang_query": "I want to update my shipping address.",
        "source_lang_resolution": "You can update your shipping address in your account profile under 'My Addresses'.",
        "target_lang_query": "Quiero actualizar mi dirección de envío.",
        "target_lang_resolution": "Puede actualizar su dirección de envío en el perfil de su cuenta en 'Mis direcciones'.",
        "language": "Spanish"
    }
]

chatbot = MultilingualChatbot(in_context_examples=icl_examples)

customer_query_spanish = "¿Dónde está mi paquete?"
target_lang = "Spanish"
print(f"\nCustomer ({target_lang}): {customer_query_spanish}")
chatbot_response = chatbot.get_response(customer_query_spanish, target_lang)
print(f"Chatbot ({target_lang}): {chatbot_response.split(f'Chatbot ({target_lang}): ')[-1].strip()}")

print("\n" + "="*50 + "\n")

customer_query_spanish_2 = "¿Cómo puedo cambiar mi contraseña?"
print(f"\nCustomer ({target_lang}): {customer_query_spanish_2}")
chatbot_response_2 = chatbot.get_response(customer_query_spanish_2, target_lang)
print(f"Chatbot ({target_lang}): {chatbot_response_2.split(f'Chatbot ({target_lang}): ')[-1].strip()}")