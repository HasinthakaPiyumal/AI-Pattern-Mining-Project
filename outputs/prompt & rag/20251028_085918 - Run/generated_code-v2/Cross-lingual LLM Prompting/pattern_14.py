
class MultilingualChatbot:
    def __init__(self, target_language="English"):
        self.target_language = target_language

    def _construct_icl_prompt(self, customer_query: str, query_language: str, in_context_examples: list) -> str:
        """
        Constructs the InCLT (In-Context Learning Transfer) prompt.
        Examples are crafted to include both source and target language components.
        """
        prompt_parts = [
            "You are a multilingual customer support assistant. Below are examples of customer queries in various languages and their corresponding solutions, demonstrating cross-lingual understanding."
        ]

        for i, example in enumerate(in_context_examples):
            prompt_parts.append(
                f"\n---\nExample {i+1} (Source: {example['source_lang']}, Target: {self.target_language}/{example['source_lang']}):"
            )
            prompt_parts.append(f"Customer: {example['source_query']}")
            prompt_parts.append(f"Solution: {example['target_solution']} (Translated back to {example['source_lang']}: {example['target_solution_in_source_lang']})")

        prompt_parts.append(
            f"\n---\nCustomer Query in {query_language}: {customer_query}\nTarget Language ({self.target_language}) Solution:"
        )

        return "\n".join(prompt_parts)

    def get_response(self, customer_query: str, query_language: str) -> dict:
        """
        Simulates an LLM processing a customer query using InCLT prompting.
        In a real application, this would involve sending the prompt to an actual LLM.
        """

        # Define example interactions for InCLT
        # These examples demonstrate using both source and target languages.
        in_context_examples = [
            {
                "source_lang": "English",
                "source_query": "I need help with my billing statement.",
                "target_solution": "Please provide your account number and the date of the statement you are inquiring about.",
                "target_solution_in_source_lang": "Necesito ayuda con mi extracto de facturación."
            },
            {
                "source_lang": "Spanish",
                "source_query": "Mi internet no funciona en casa.",
                "target_solution": "Have you tried restarting your router and checking the cable connections?",
                "target_solution_in_source_lang": "¿Ha intentado reiniciar su router y revisar las conexiones de cable?"
            },
            {
                "source_lang": "French",
                "source_query": "Je voudrais changer mon abonnement.",
                "target_solution": "What kind of subscription change are you looking for? Upgrading or downgrading?",
                "target_solution_in_source_lang": "¿Qué tipo de cambio de suscripción está buscando? ¿Actualizar o degradar?"
            }
        ]

        prompt = self._construct_icl_prompt(customer_query, query_language, in_context_examples)

        # Simulate LLM response based on the query and examples.
        # In a real system, an actual LLM would generate this based on the prompt.
        simulated_llm_response = """
Based on your query, let me try to assist you. If you are asking for help with billing, please provide your account number. If it's about internet, please check your router. If it's about changing a subscription, please specify what you would like to do.

Simulated direct answer: (Placeholder - actual LLM would generate specific answer here based on query and examples)
"""

        if "internet" in customer_query.lower() or "router" in customer_query.lower() or "funciona" in customer_query.lower():
            simulated_llm_response = "Please try restarting your router and checking all cable connections. If the problem persists, our technical team can assist you further."
        elif "billing" in customer_query.lower() or "facturación" in customer_query.lower() or "statement" in customer_query.lower():
            simulated_llm_response = "Could you please provide your account number and the specific date of the billing statement in question?"
        elif "subscription" in customer_query.lower() or "abonnement" in customer_query.lower() or "cambiar" in customer_query.lower():
             simulated_llm_response = "To help you with your subscription change, please tell me if you're looking to upgrade, downgrade, or modify specific features."

        return {"prompt": prompt, "simulated_response": simulated_llm_response}

# --- Example Usage ---
if __name__ == "__main__":
    chatbot = MultilingualChatbot()

    # Spanish customer query
    spanish_query = "Mi televisión no tiene señal."
    spanish_response_data = chatbot.get_response(spanish_query, "Spanish")
    print("\n--- Spanish Query ---")
    print("Prompt:\n", spanish_response_data["prompt"])
    print("\nSimulated LLM Response:\n", spanish_response_data["simulated_response"])

    # English customer query
    english_query = "I want to upgrade my data plan."
    english_response_data = chatbot.get_response(english_query, "English")
    print("\n--- English Query ---")
    print("Prompt:\n", english_response_data["prompt"])
    print("\nSimulated LLM Response:\n", english_response_data["simulated_response"])

    # French customer query
    french_query = "J'ai un problème avec ma facture."
    french_response_data = chatbot.get_response(french_query, "French")
    print("\n--- French Query ---")
    print("Prompt:\n", french_response_data["prompt"])
    print("\nSimulated LLM Response:\n", french_response_data["simulated_response"])
