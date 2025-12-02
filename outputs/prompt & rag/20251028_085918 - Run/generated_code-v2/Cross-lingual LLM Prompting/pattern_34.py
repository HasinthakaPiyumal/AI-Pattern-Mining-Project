import random

class MultilingualChatbot:
    def __init__(self, icl_examples):
        self.icl_examples = icl_examples

    def _get_icl_prompt_examples(self, target_language, num_examples=2):
        selected_examples = []
        available_keys = list(self.icl_examples.keys())
        random.shuffle(available_keys)
        for key in available_keys[:num_examples]:
            ex = self.icl_examples[key]
            if f"{target_language.lower()}_query" in ex and f"{target_language.lower()}_response" in ex:
                selected_examples.append(ex)
        return selected_examples

    def _construct_in_clt_prompt(self, user_query, target_language):
        prompt_parts = []
        selected_examples = self._get_icl_prompt_examples(target_language)

        for ex in selected_examples:
            prompt_parts.append(f"English: '{ex['english_query']}' -> '{ex['english_response']}'")
            prompt_parts.append(f"{target_language.capitalize()}: '{ex[f'{target_language.lower()}_query']}' -> '{ex[f'{target_language.lower()}_response']}'")
        
        prompt_parts.append(f"User Query ({target_language.capitalize()}): '{user_query}'")
        return "\n\n".join(prompt_parts)

    def _simulate_llm_response(self, prompt):
        # This is a highly simplified simulation of an LLM response.
        # In a real application, this would be an API call to a multilingual LLM.
        
        # Look for keywords from the ICL examples to simulate a relevant response
        if "devolver" in prompt.lower() or "return" in prompt.lower():
            return "Para devolver un artículo, por favor, visite nuestra página de devoluciones y siga las instrucciones. (To return an item, please visit our returns page and follow the instructions.)"
        elif "dañado" in prompt.lower() or "damaged" in prompt.lower() or "reembolso" in prompt.lower() or "refund" in prompt.lower():
            return "Sí, por favor, proporcione su número de pedido y los detalles, y procesaremos un reembolso. (Yes, please provide your order number and details, and we will process a refund.)"
        else:
            return "Disculpe, no entendí su consulta. ¿Podría reformularla? (Sorry, I didn't understand your query. Could you rephrase it?)"

    def get_chatbot_response(self, user_query, target_language="Spanish"):
        in_clt_prompt = self._construct_in_clt_prompt(user_query, target_language)
        print("\n--- Generated InCLT Prompt ---")
        print(in_clt_prompt)
        print("------------------------------")
        simulated_response = self._simulate_llm_response(in_clt_prompt)
        return simulated_response

# --- Example Usage ---
if __name__ == "__main__":
    # Define your In-Context Learning examples with both source (English) and target (Spanish) languages
    icl_data = {
        "returns_process": {
            "english_query": "How do I return an item?",
            "english_response": "To return an item, please visit our returns page and follow the instructions.",
            "spanish_query": "¿Cómo devuelvo un artículo?",
            "spanish_response": "Para devolver un artículo, por favor, visite nuestra página de devoluciones y siga las instrucciones."
        },
        "damaged_order_refund": {
            "english_query": "My order arrived damaged, can I get a refund?",
            "english_response": "Yes, please provide your order number and details, and we will process a refund.",
            "spanish_query": "Mi pedido llegó dañado, ¿puedo obtener un reembolso?",
            "spanish_response": "Sí, por favor, proporcione su número de pedido y los detalles, y procesaremos un reembolso."
        },
        "shipping_cost": {
            "english_query": "What are the shipping costs?",
            "english_response": "Shipping costs vary based on your location and selected shipping method. You can see the exact cost at checkout.",
            "spanish_query": "¿Cuáles son los gastos de envío?",
            "spanish_response": "Los gastos de envío varían según su ubicación y el método de envío seleccionado. Puede ver el costo exacto al finalizar la compra."
        }
    }

    chatbot = MultilingualChatbot(icl_data)

    # Simulate a customer query in Spanish
    user_query_spanish_1 = "Quiero devolver un producto, ¿cuál es el proceso?"
    response_1 = chatbot.get_chatbot_response(user_query_spanish_1, "Spanish")
    print(f"\nCustomer: {user_query_spanish_1}")
    print(f"Chatbot: {response_1}")

    user_query_spanish_2 = "Mi pedido llegó dañado, necesito un reembolso."
    response_2 = chatbot.get_chatbot_response(user_query_spanish_2, "Spanish")
    print(f"\nCustomer: {user_query_spanish_2}")
    print(f"Chatbot: {response_2}")

    user_query_spanish_3 = "¿Cuánto cuesta el envío a México?"
    response_3 = chatbot.get_chatbot_response(user_query_spanish_3, "Spanish")
    print(f"\nCustomer: {user_query_spanish_3}")
    print(f"Chatbot: {response_3}")

    # Simulate a query that might not perfectly match ICL examples
    user_query_spanish_4 = "Tengo una pregunta general sobre mi cuenta."
    response_4 = chatbot.get_chatbot_response(user_query_spanish_4, "Spanish")
    print(f"\nCustomer: {user_query_spanish_4}")
    print(f"Chatbot: {response_4}")

