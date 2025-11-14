
class MultilingualLLMMock:
    """
    A mock Large Language Model to simulate multilingual responses.
    In a real application, this would be replaced by an actual LLM (e.g., from Hugging Face Transformers).
    """
    def generate_response(self, prompt: str) -> str:
        """
        Simulates an LLM generating a response based on the prompt.
        It attempts to extract the desired language and the user's query
        to provide a relevant, language-aware canned response.
        """
        # Very simplified extraction of desired response language and the user's actual query
        desired_response_lang = "English"  # Default
        if "in Spanish." in prompt:
            desired_response_lang = "Spanish"
        elif "in French." in prompt:
            desired_response_lang = "French"
        elif "in German." in prompt: # Example of another language that could be supported
            desired_response_lang = "German"

        # Extract the user's specific query from the prompt
        user_query_text = ""
        user_query_marker = "\nCustomer: "
        assistant_marker = "\nAssistant:"

        # Find the last occurrence of the customer query before the final "Assistant:" prompt
        last_user_query_start = prompt.rfind(user_query_marker, 0, prompt.rfind(assistant_marker))
        if last_user_query_start != -1:
            start_index = last_user_query_start + len(user_query_marker)
            end_index = prompt.find(assistant_marker, start_index)
            if end_index != -1:
                user_query_text = prompt[start_index:end_index].strip()
            else:
                user_query_text = prompt[start_index:].strip()


        # A very basic "intelligent" response mapping based on keywords and desired language
        responses_map = {
            "order status": {
                "english": "Your order #12345 is being processed. Expect shipment within 2 days.",
                "spanish": "Su pedido #12345 está siendo procesado. Envío esperado en 2 días.",
                "french": "Votre commande #12345 est en cours de traitement. Expédition prévue sous 2 jours."
            },
            "product features": {
                "english": "Our product offers a 10-hour battery life and is water-resistant.",
                "spanish": "Nuestro producto ofrece 10 horas de batería y es resistente al agua.",
                "french": "Notre produit offre une autonomie de 10 heures et est résistant à l'eau."
            },
            "hello": {
                "english": "Hello! How can I assist you today?",
                "spanish": "¡Hola! ¿Cómo puedo ayudarle hoy?",
                "french": "Bonjour ! Comment puis-je vous aider aujourd'hui ?" 
            },
            "default": {  # Fallback response
                "english": f"I received your query: \'{user_query_text}\'. How else can I assist?",
                "spanish": f"Recibí su consulta: \'{user_query_text}\'. ¿En qué más puedo ayudarle?",
                "french": f"J'ai reçu votre question : \'{user_query_text}\'. Comment puis-je vous aider davantage ?" 
            }
        }

        response = responses_map["default"][desired_response_lang.lower()]
        for keyword, lang_responses in responses_map.items():
            if keyword != "default" and keyword in user_query_text.lower():
                response = lang_responses.get(desired_response_lang.lower(), responses_map["default"][desired_response_lang.lower()])
                break

        return response


class PromptManager:
    """
    Manages the creation of prompts using the InCLT Crosslingual Transfer Prompting pattern.
    It stores cross-lingual examples and integrates them into the prompt for the LLM.
    """
    def __init__(self):
        self.examples = []

    def add_cross_lingual_example(self,
                                  source_lang: str,
                                  source_query: str,
                                  source_response: str,
                                  target_lang: str,
                                  target_query: str,
                                  target_response: str):
        """
        Adds a cross-lingual in-context learning example to the manager.
        Each example consists of a query-response pair presented in two different languages.
        These examples guide the LLM on how to perform cross-lingual transfer.
        """
        self.examples.append({
            "lang1": source_lang.lower(), "query1": source_query, "response1": source_response,
            "lang2": target_lang.lower(), "query2": target_query, "response2": target_response
        })

    def generate_icl_prompt(self, user_query: str, desired_response_lang: str) -> str:
        """
        Generates a comprehensive prompt for the LLM using the InCLT pattern.
        The prompt includes a system instruction, cross-lingual in-context examples,
        and finally the actual user query for the LLM to respond to.
        """
        desired_response_lang_lower = desired_response_lang.lower()

        prompt_parts = [
            "You are a helpful multilingual customer support assistant.",
            f"Please respond to the following query in {desired_response_lang}.",
            "Here are some examples of how to answer customer queries, demonstrating cross-lingual understanding and transfer:"
        ]

        # Add all stored cross-lingual examples to the prompt
        # This is the core of InCLT: showing the LLM the mapping and transfer capabilities
        for i, ex in enumerate(self.examples):
            prompt_parts.append(f"\n--- Cross-Lingual Example {i+1} ({ex['lang1'].capitalize()} to {ex['lang2'].capitalize()}) ---")
            prompt_parts.append(f"Customer ({ex['lang1']}): {ex['query1']}")
            prompt_parts.append(f"Assistant ({ex['lang1']}): {ex['response1']}")
            prompt_parts.append(f"Customer ({ex['lang2']}): {ex['query2']}")
            prompt_parts.append(f"Assistant ({ex['lang2']}): {ex['response2']}")

        # Add the actual user query at the end
        prompt_parts.append(f"\n--- Your Turn ({desired_response_lang}) ---")
        prompt_parts.append(f"Customer: {user_query}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)


class MultilingualChatbot:
    """
    A Multilingual Customer Support Chatbot that leverages InCLT Crosslingual Transfer Prompting.
    It uses a PromptManager to construct prompts for a multilingual LLM.
    """
    def __init__(self, llm_model, prompt_manager):
        """
        Initializes the chatbot with an LLM and a prompt manager.

        Args:
            llm_model: An instance of a multilingual LLM (or a mock).
            prompt_manager: An instance of PromptManager to handle example-based prompting.
        """
        self.llm = llm_model
        self.prompt_manager = prompt_manager

    def add_icl_example(self, source_lang: str, source_query: str, source_response: str,
                        target_lang: str, target_query: str, target_response: str):
        """
        Delegates adding a cross-lingual in-context learning example to the prompt manager.
        """
        self.prompt_manager.add_cross_lingual_example(
            source_lang, source_query, source_response,
            target_lang, target_query, target_response
        )

    def get_response(self, customer_query: str, response_language: str) -> str:
        """
        Generates a response to a customer query using the InCLT prompting pattern.

        Args:
            customer_query: The query from the customer.
            response_language: The language in which the response should be generated (e.g., "English", "Spanish").

        Returns:
            The generated response from the LLM.
        """
        # Generate the prompt with cross-lingual in-context examples
        full_prompt = self.prompt_manager.generate_icl_prompt(customer_query, response_language)
        
        # For debugging: print the full prompt to see how InCLT works
        # print("\n--- FULL PROMPT SENT TO LLM ---\n", full_prompt, "\n----------------------------\n")

        # Get response from the LLM
        llm_response = self.llm.generate_response(full_prompt)
        return llm_response


# Main execution for demonstration of the InCLT Crosslingual Transfer Prompting pattern
if __name__ == "__main__":
    print("Starting Multilingual Customer Support Chatbot Demonstration...")

    # 1. Initialize our mock LLM and prompt manager
    mock_llm = MultilingualLLMMock()
    prompt_manager = PromptManager()

    # 2. Add cross-lingual in-context examples (This is where the InCLT pattern is applied)
    # These examples teach the LLM how to transfer knowledge between languages for specific tasks.
    print("\nAdding cross-lingual in-context examples to the prompt manager...")
    prompt_manager.add_cross_lingual_example(
        source_lang="English",
        source_query="What is the status of my order?",
        source_response="Your order #XYZ is currently being processed and is expected to ship within 2 business days.",
        target_lang="Spanish",
        target_query="¿Cuál es el estado de mi pedido?",
        target_response="Su pedido #XYZ está siendo procesado y se espera que se envíe en un plazo de 2 días hábiles."
    )

    prompt_manager.add_cross_lingual_example(
        source_lang="English",
        source_query="Tell me about the product features.",
        source_response="The \'Awesome Gadget\' features a long-lasting battery and supports fast charging.",
        target_lang="French",
        target_query="Parlez-moi des fonctionnalités du produit.",
        target_response="Le \'Super Gadget\' dispose d'une batterie longue durée et prend en charge la charge rapide."
    )

    prompt_manager.add_cross_lingual_example(
        source_lang="Spanish",
        source_query="Necesito ayuda con la configuración de mi cuenta.",
        source_response="Claro, ¿podría proporcionarme su nombre de usuario o correo electrónico?",
        target_lang="English",
        target_query="I need help with my account settings.",
        target_response="Certainly, could you please provide me with your username or email?"
    )
    print("Examples added successfully.")

    # 3. Initialize the chatbot with the mock LLM and the prompt manager
    chatbot = MultilingualChatbot(llm_model=mock_llm, prompt_manager=prompt_manager)
    print("\nMultilingual Chatbot ready for interactions!")

    # 4. Simulate customer interactions in different languages
    print("\n--- Simulating Customer Query 1 (Spanish) ---")
    query1 = "¿Cuál es el estado de mi pedido?"
    response1 = chatbot.get_response(query1, "Spanish")
    print(f"Customer: {query1}")
    print(f"Chatbot (Spanish): {response1}")

    print("\n--- Simulating Customer Query 2 (French) ---")
    query2 = "Parlez-moi des fonctionnalités du produit."
    response2 = chatbot.get_response(query2, "French")
    print(f"Customer: {query2}")
    print(f"Chatbot (French): {response2}")

    print("\n--- Simulating Customer Query 3 (English - a query not directly in examples but related) ---")
    query3 = "What are the features of your new smartphone?"
    response3 = chatbot.get_response(query3, "English")
    print(f"Customer: {query3}")
    print(f"Chatbot (English): {response3}")

    print("\n--- Simulating Customer Query 4 (Spanish - a new greeting, expecting general response) ---")
    query4 = "¿Hola, como estas?" # A new greeting, should still trigger a Spanish response from mock LLM
    response4 = chatbot.get_response(query4, "Spanish")
    print(f"Customer: {query4}")
    print(f"Chatbot (Spanish): {response4}")

    print("\n--- Simulating Customer Query 5 (English - a new greeting, expecting general response) ---")
    query5 = "Hi there!" 
    response5 = chatbot.get_response(query5, "English")
    print(f"Customer: {query5}")
    print(f"Chatbot (English): {response5}")

    print("\nDemonstration complete.")
