class MultilingualChatbot:
    """
    A simulated Multilingual Customer Support Chatbot leveraging
    Cross-Lingual In-Context Learning (InCLT) for enhanced performance
    in multilingual scenarios.
    """

    def __init__(self, llm_model_name="Simulated-LLM"):
        """
        Initializes the chatbot with a placeholder for an LLM.
        In a real application, this would load an actual multilingual LLM.
        """
        self.llm_model_name = llm_model_name
        print(f"Chatbot initialized using {self.llm_model_name} for cross-lingual processing.")

        # Predefined cross-lingual examples for in-context learning
        # These examples demonstrate how the LLM should handle a task (e.g., provide a solution)
        # using information from both source and target languages.
        self.cross_lingual_examples = {
            "customer_service_query": [
                {
                    "instruction": "Translate the user's issue and provide a solution in the target language. The issue is about 'order status'.",
                    "source_lang_issue": "My order number 12345 is late. Where is it?",
                    "source_lang_solution": "Your order #12345 is currently in transit and expected to arrive by next Tuesday. You can track it here: [tracking_link].",
                    "target_lang_issue_fr": "Ma commande numéro 12345 est en retard. Où est-elle ?",
                    "target_lang_solution_fr": "Votre commande #12345 est actuellement en transit et devrait arriver d'ici mardi prochain. Vous pouvez la suivre ici : [lien_de_suivi].",
                    "target_lang_issue_es": "Mi pedido número 12345 está retrasado. ¿Dónde está?",
                    "target_lang_solution_es": "Su pedido #12345 está actualmente en tránsito y se espera que llegue el próximo martes. Puede rastrearlo aquí: [enlace_de_seguimiento]."
                },
                {
                    "instruction": "Help the user with a 'refund request'. Explain the process.",
                    "source_lang_issue": "I want a refund for item X. How do I do that?",
                    "source_lang_solution": "To request a refund for item X, please navigate to your 'Order History', select the item, and click 'Request Refund'. Our team will review it within 2 business days.",
                    "target_lang_issue_fr": "Je souhaite un remboursement pour l'article X. Comment faire ?",
                    "target_lang_solution_fr": "Pour demander un remboursement pour l'article X, veuillez vous rendre dans votre 'Historique de commandes', sélectionner l'article et cliquer sur 'Demander un remboursement'. Notre équipe l'examinera dans les 2 jours ouvrables.",
                    "target_lang_issue_es": "Quiero un reembolso por el artículo X. ¿Cómo lo hago?",
                    "target_lang_solution_es": "Para solicitar un reembolso por el artículo X, vaya a su 'Historial de pedidos', seleccione el artículo y haga clic en 'Solicitar reembolso'. Nuestro equipo lo revisará en un plazo de 2 días hábiles."
                }
            ]
        }

    def _generate_in_context_prompt(self, user_query: str, target_language: str) -> str:
        """
        Generates a prompt incorporating cross-lingual in-context examples.
        This simulates the InCLT pattern by providing examples in both source
        (English, for simplicity) and the target language.
        """
        prompt_parts = [
            f"You are a helpful multilingual customer support assistant.",
            f"Your goal is to understand the user's query, which might be in English or another language, and provide a helpful response in {target_language}. Leverage the following examples to understand how to transfer knowledge across languages for customer support tasks."
        ]

        # Add cross-lingual examples
        for example in self.cross_lingual_examples["customer_service_query"]:
            prompt_parts.append("\n--- Example ---")
            prompt_parts.append(f"Instruction: {example['instruction']}")
            prompt_parts.append(f"Source (English) Issue: {example['source_lang_issue']}")
            prompt_parts.append(f"Source (English) Solution: {example['source_lang_solution']}")
            if target_language == "French":
                prompt_parts.append(f"Target (French) Issue: {example['target_lang_issue_fr']}")
                prompt_parts.append(f"Target (French) Solution: {example['target_lang_solution_fr']}")
            elif target_language == "Spanish":
                prompt_parts.append(f"Target (Spanish) Issue: {example['target_lang_issue_es']}")
                prompt_parts.append(f"Target (Spanish) Solution: {example['target_lang_solution_es']}")
            # Add more languages if needed

        prompt_parts.append("\n--- User Query ---")
        prompt_parts.append(f"User: {user_query}")
        prompt_parts.append(f"Please respond in {target_language}.")
        prompt_parts.append(f"Assistant:")

        return "\n".join(prompt_parts)

    def get_response(self, user_query: str, target_language: str = "English") -> str:
        """
        Simulates getting a response from a multilingual LLM based on the
        user's query and target language, using the InCLT prompting pattern.
        """
        if target_language not in ["English", "French", "Spanish"]:
            return "Sorry, I can only provide support in English, French, or Spanish at the moment."

        full_prompt = self._generate_in_context_prompt(user_query, target_language)

        # In a real application, you would send `full_prompt` to an actual LLM API.
        # For this simulation, we'll provide a simplified, rule-based response
        # that attempts to reflect the cross-lingual intent.

        print("\n--- Sending to Simulated LLM ---")
        print(full_prompt)
        print("--- End Simulated LLM Input ---")

        # Simulate LLM's understanding and response based on query and target language
        if "order" in user_query.lower() or "commande" in user_query.lower() or "pedido" in user_query.lower():
            if "status" in user_query.lower() or "statut" in user_query.lower() or "estado" in user_query.lower() or "late" in user_query.lower() or "retard" in user_query.lower() or "retrasado" in user_query.lower():
                if target_language == "French":
                    return "Votre commande est en transit et devrait arriver bientôt. Vous pouvez vérifier le statut sur votre page de commande."
                elif target_language == "Spanish":
                    return "Su pedido está en tránsito y debería llegar pronto. Puede verificar el estado en su página de pedido."
                else:
                    return "Your order is in transit and should arrive soon. You can check the status on your order page."
            elif "cancel" in user_query.lower() or "annuler" in user_query.lower() or "cancelar" in user_query.lower():
                if target_language == "French":
                    return "Pour annuler une commande, veuillez visiter votre historique de commandes et sélectionner l'option d'annulation."
                elif target_language == "Spanish":
                    return "Para cancelar un pedido, visite su historial de pedidos y seleccione la opción de cancelación."
                else:
                    return "To cancel an order, please visit your order history and select the cancellation option."
        elif "refund" in user_query.lower() or "remboursement" in user_query.lower() or "reembolso" in user_query.lower():
            if target_language == "French":
                return "Pour demander un remboursement, veuillez suivre les étapes décrites dans notre politique de retour sur le site web."
            elif target_language == "Spanish":
                return "Para solicitar un reembolso, siga los pasos descritos en nuestra política de devoluciones en el sitio web."
            else:
                return "To request a refund, please follow the steps outlined in our return policy on the website."
        elif "hello" in user_query.lower() or "bonjour" in user_query.lower() or "hola" in user_query.lower():
            if target_language == "French":
                return "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
            elif target_language == "Spanish":
                return "¡Hola! ¿Cómo puedo ayudarte hoy?"
            else:
                return "Hello! How can I assist you today?"
        
        # Fallback for other queries
        if target_language == "French":
            return "Je ne suis pas sûr de comprendre votre demande. Pourriez-vous reformuler ou poser une question différente ?"
        elif target_language == "Spanish":
            return "No estoy seguro de entender su solicitud. ¿Podría reformular o hacer una pregunta diferente?"
        else:
            return "I'm not sure I understand your request. Could you please rephrase or ask a different question?"


# --- Example Usage ---
if __name__ == "__main__":
    chatbot = MultilingualChatbot()

    print("\n--- English Interaction ---")
    response_en = chatbot.get_response("My order 54321 is delayed.", "English")
    print(f"Assistant (English): {response_en}")

    print("\n--- French Interaction ---")
    response_fr = chatbot.get_response("Je veux annuler ma commande.", "French")
    print(f"Assistant (French): {response_fr}")

    print("\n--- Spanish Interaction ---")
    response_es = chatbot.get_response("Quiero un reembolso por mi compra.", "Spanish")
    print(f"Assistant (Spanish): {response_es}")

    print("\n--- Mixed Language Query (English input, Spanish output) ---")
    response_mixed = chatbot.get_response("Hello, I need help with my subscription.", "Spanish")
    print(f"Assistant (Spanish): {response_mixed}")

    print("\n--- Unknown Language Target ---")
    response_unknown_lang = chatbot.get_response("Hello", "German")
    print(f"Assistant (German): {response_unknown_lang}")